"""The engine: one core shared by proxy mode and inline mode.

    jit = TreeJIT("treejit.db")
    res = jit.handle("anthropic", body, headers)   # replay, subcall, or a body to forward
    if res.kind == "subcall":                       # T2/T3: send res.body upstream, non-streaming
        res = jit.resume(res, response_json, status) # -> replay, or forward (never another subcall)
    ...forward res.body upstream if res.kind == "forward"...
    jit.complete(res, response_info, status, latency_ms)
    jit.outcome(run_id, "pass")                     # verifier signal; rebuilds the tree
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import compaction, dialects, families, subcalls
from .builder import build_family
from .config import Config
from .dialects import Dialect
from .model import Episode, NormRequest, ResponseInfo, ToolCall
from .replay import Plan, decide, hints, materialize
from .store import Store, dumps
from .tree import TreeView
from .util import h

log = logging.getLogger("treejit")

RUN_HEADER = "x-treejit-run"
TRUNCATED = ("max_tokens", "length", "pause_turn", "content_filter", "refusal")  # stops that don't end a task


@dataclass
class _Pending:
    request_id: int
    family: str | None
    task: str
    task_hash: str
    run_id: str | None
    started: float
    n_steps: int = -1                 # steps in the episode when forwarded (to record where the model ended it)


@dataclass
class _Sub(_Pending):
    """State carried from handle() to resume() across a T2/T3 subcall."""

    req: NormRequest | None = None
    view: TreeView | None = None
    plan: Plan | None = None


@dataclass
class Result:
    kind: str                         # "replay" | "forward" | "subcall"
    body: dict | None                 # replay: response JSON (non-stream); forward/subcall: request body to send
    stream: bool = False
    sse: list[bytes] | None = None    # replay + stream: SSE chunks
    run_id: str | None = None
    tier: str = "T4"
    plan: Plan | None = None
    headers: dict[str, str] = field(default_factory=dict)
    ctx: _Pending | None = None


class TreeJIT:
    def __init__(self, db: str | None = None, config: Config | None = None, **overrides: Any) -> None:
        self.cfg = config or Config.load(**overrides)
        if db is not None:
            self.cfg.db = db
        self.store = Store(self.cfg.db)
        # family -> (view, the family's built_at it was loaded at). One tuple, swapped atomically.
        self._views: dict[str, tuple[TreeView, float]] = {}
        self._build_lock = threading.RLock()  # one build at a time per instance (sync and background)
        self._rebuilder: _Rebuilder | None = None
        self.rebuild_mode = "sync"
        self.set_rebuild_mode("sync" if self.cfg.rebuild == "auto" else self.cfg.rebuild)

    # ------------------------------------------------------------------ tree
    def set_rebuild_mode(self, mode: str) -> None:
        """"sync": outcome() rebuilds before it returns. "background": a worker thread rebuilds
        (coalescing outcomes per family) and swaps the live view in when done; requests keep the
        previous view meanwhile. The proxy uses background unless `rebuild = "sync"`."""
        if mode not in ("sync", "background"):
            raise ValueError(f"rebuild mode must be sync|background|auto, got {mode!r}")
        self.rebuild_mode = mode
        if mode == "background" and self._rebuilder is None:
            self._rebuilder = _Rebuilder(self)

    def view(self, family: str) -> TreeView:
        # one query per request: `dirty` asks for a build, `built_at` says whether the cached view is
        # stale (another instance or process, e.g. `treejit outcome`, rebuilt the family)
        row = self.store.q1("SELECT dirty, built_at FROM families WHERE id=?", (family,))
        if row is not None and row["dirty"]:
            rb = self._rebuilder if self.rebuild_mode == "background" else None
            if rb is None:
                with self._build_lock:  # another thread may have built it while we waited
                    again = self.store.q1("SELECT dirty FROM families WHERE id=?", (family,))
                    if again is not None and again["dirty"]:
                        self.rebuild(family)
                row = None
                if family not in self._views:
                    self._views[family] = self._load(self.store, family)
            elif not rb.pending(family):
                rb.request(family)  # serve the current view meanwhile
        cached = self._views.get(family)
        if cached is None or (row is not None and (row["built_at"] or 0.0) != cached[1]):
            cached = self._load(self.store, family)
            self._views[family] = cached
        return cached[0]

    @staticmethod
    def _load(store: Store, family: str) -> tuple[TreeView, float]:
        with store.transaction():  # one snapshot: the tables and built_at agree
            row = store.q1("SELECT built_at FROM families WHERE id=?", (family,))
            view = TreeView.load(store, family)
        return view, ((row["built_at"] or 0.0) if row is not None else 0.0)

    def rebuild(self, family: str | None = None) -> list[dict]:
        """Rebuild now, on the calling thread (any mode)."""
        fams = [family] if family else [r["id"] for r in self.store.q("SELECT id FROM families")]
        return [self._build(self.store, f) for f in fams]

    def _build(self, store: Store, family: str) -> dict:
        with self._build_lock, _no_gc():
            out = build_family(store, self.cfg, family)
            self._views[family] = self._load(store, family)
        return out

    def wait_rebuilds(self, timeout: float | None = None) -> bool:
        """Block until the background rebuilds requested so far are done (True) or `timeout` passes."""
        rb = self._rebuilder
        return True if rb is None else rb.wait_all(timeout)

    def rebuild_dirty(self) -> list[dict]:
        return [r for f in self.store.q("SELECT id FROM families WHERE dirty=1") for r in self.rebuild(f["id"])]

    # ------------------------------------------------------------------ requests
    def handle(self, dialect: str, body: dict, headers: dict | None = None) -> Result:
        t0 = time.perf_counter()
        d = dialects.get(dialect)
        hdrs = {k.lower(): v for k, v in (headers or {}).items()}
        req = d.parse_request(body)
        if not req.tools:
            rid = self.store.log_request(dialect=dialect, tier="pass", note="no_tools")
            return Result("forward", body, req.stream, ctx=_Pending(rid, None, "", "", None, t0))

        fam = families.resolve(self.store, req.system, req.tools, dialect)
        ep = req.episode
        task_hash = h(ep.task)
        run_id = self._run_id(hdrs, fam, ep, task_hash)
        if run_id and ep.steps:
            self._record(run_id, fam, ep, task_hash)

        view = self.view(fam)
        plan = decide(view, self.cfg, req, d)
        if plan.calls:
            return self._replay(d, req, fam, ep.task, task_hash, run_id, plan, t0)
        if plan.sub is not None:
            sub_body = subcalls.build(d.name, plan.sub, req, self.cfg)
            if sub_body is not None:
                sub = plan.sub
                labels = " | ".join(o.edge.label for o in sub.options)
                rid = self.store.log_request(family=fam, run_id=run_id, dialect=dialect, tier=sub.tier, node=sub.node,
                                             note=f"{sub.reason}@{sub.used}: {labels}"[:500])
                return Result("subcall", sub_body, req.stream, None, run_id, sub.tier, plan, self._headers(run_id, sub.tier),
                              _Sub(rid, fam, ep.task, task_hash, run_id, t0, len(ep.steps), req, view, plan))
        return self._forward(d, req, view, plan, fam, ep.task, task_hash, run_id, t0)

    def resume(self, result: Result, response: dict | None, status: int = 200, latency_ms: float | None = None) -> Result:
        """Finish a T2/T3 subcall: `response` is the upstream JSON (None on transport failure).
        Returns a replay, or the T4 forward the request would have been without the subcall."""
        ctx = result.ctx
        if result.kind != "subcall" or not isinstance(ctx, _Sub) or ctx.plan is None or ctx.plan.sub is None:
            raise ValueError("resume() expects a subcall Result from handle()")
        req, plan, sub = ctx.req, ctx.plan, ctx.plan.sub
        d = dialects.get(req.dialect)
        ms = latency_ms if latency_ms is not None else (time.perf_counter() - ctx.started) * 1000
        kw: dict[str, Any] = {"status": status, "latency_ms": ms}
        call, why, label = None, f"http_{status}", ""
        if response is not None and status < 400:
            try:
                u = d.parse_response(response).usage
                kw.update(input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read=u.cache_read,
                          cache_write=u.cache_write)
            except (AttributeError, TypeError, ValueError):
                pass
            opt, vals, why = subcalls.resolve(sub, subcalls.tool_input(d.name, sub, response))
            if opt is not None:
                args, why = (opt.args, "") if not opt.holes else materialize(ctx.view, self.cfg, opt, vals)
                if args is not None:
                    via = "t3" if sub.kind == "fill" else "ck" if sub.reason == "budget" else "t2"
                    conf = opt.conf if sub.kind == "fill" else 1.0
                    call, label = ToolCall(d.new_call_id(sub.node, conf, via), opt.edge.tool, args), opt.edge.label
        row = self.store.q1("SELECT note FROM requests WHERE id=?", (ctx.request_id,))
        note = row["note"] if row is not None else ""
        if call is None:
            self.store.update_request(ctx.request_id, note=f"{note}; failed:{why}"[:500], **kw)
            plan.reason = f"{sub.tier}_failed:{why}"
            return self._forward(d, req, ctx.view, plan, ctx.family, ctx.task, ctx.task_hash, ctx.run_id, ctx.started)
        plan.calls, plan.nodes, plan.tier, plan.reason = [call], [sub.node], sub.tier, ""
        plan.detail.append(f"{sub.tier}@{sub.used} {label} ({sub.reason})")
        res = self._replay(d, req, ctx.family, ctx.task, ctx.task_hash, ctx.run_id, plan, ctx.started, log=False)
        self.store.update_request(ctx.request_id, run_id=res.run_id, n_calls=1, call_ids=json.dumps([call.id]),
                                  note=f"{note}; ok: {label}"[:500], **kw)
        return res

    def _replay(self, d: Dialect, req: NormRequest, fam: str, task: str, task_hash: str, run_id: str | None,
                plan: Plan, t0: float, log: bool = True) -> Result:
        if run_id is None:
            run_id = "r_" + h(fam, task_hash, plan.calls[0].id)
            self.store.upsert_run(run_id, fam, task, task_hash)
        for nid in plan.nodes:
            self.store.hit(nid)
        if req.stream:
            sse, resp = d.build_sse(req.model, plan.calls, req.raw), None
        else:
            sse, resp = None, d.build_response(req.model, plan.calls)
        if log:
            ms = (time.perf_counter() - t0) * 1000
            self.store.log_request(family=fam, run_id=run_id, dialect=d.name, tier=plan.tier, node=plan.nodes[0],
                                   n_calls=len(plan.calls), latency_ms=ms, status=200,
                                   call_ids=json.dumps([c.id for c in plan.calls]), note="; ".join(plan.detail)[:500])
        return Result("replay", resp, req.stream, sse, run_id, plan.tier, plan, self._headers(run_id, plan.tier))

    def _forward(self, d: Dialect, req: NormRequest, view: TreeView, plan: Plan, fam: str, task: str, task_hash: str,
                 run_id: str | None, t0: float) -> Result:
        fwd = d.prepare_forward(req)
        comp = compaction.apply(self.store, view, self.cfg, req, fwd) if self.cfg.compact else None
        if comp is not None:
            fwd = comp.body
        hint = hints(view, self.cfg, plan.node)
        if hint:
            fwd = d.inject_hint(fwd, hint)
        if plan.node:
            self.store.hit(plan.node)
        note = plan.reason + ("; " + "; ".join(plan.detail) if plan.detail else "") + ("; hints" if hint else "")
        if comp is not None and comp.n:
            note += "; " + comp.note
        rid = self.store.log_request(family=fam, run_id=run_id, dialect=d.name, tier="T4", node=plan.node, note=note[:500],
                                     compacted_chars=comp.chars if comp is not None else 0)
        return Result("forward", fwd, req.stream, None, run_id, "T4", plan, self._headers(run_id, "T4"),
                      _Pending(rid, fam, task, task_hash, run_id, t0, len(req.episode.steps)))

    def complete(self, result: Result, info: ResponseInfo | None, status: int = 200, latency_ms: float | None = None) -> None:
        """Record usage/latency of a forwarded request once the upstream response is known."""
        ctx = result.ctx
        if ctx is None or result.kind != "forward":
            return
        run_id = ctx.run_id
        if run_id is None and ctx.family and info is not None and info.calls and status < 400:
            run_id = "r_" + h(ctx.family, ctx.task_hash, info.calls[0].id)
            self.store.upsert_run(run_id, ctx.family, ctx.task, ctx.task_hash)
            result.run_id = run_id
        ms = latency_ms if latency_ms is not None else (time.perf_counter() - ctx.started) * 1000
        kw: dict[str, Any] = {"status": status, "latency_ms": ms, "run_id": run_id}
        if info is not None:
            u = info.usage
            kw.update(input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read=u.cache_read,
                      cache_write=u.cache_write, n_calls=len(info.calls), call_ids=json.dumps([c.id for c in info.calls]))
        self.store.update_request(ctx.request_id, **kw)
        if (run_id and info is not None and status < 400 and not info.calls and ctx.n_steps > 0
                and info.stop_reason not in TRUNCATED):
            # the model ended the episode here: a final answer, no tool call. The builder turns this
            # into an END choice at the contexts after the last step (see tree.END).
            self.store.set_ended(run_id, ctx.n_steps)

    def outcome(self, run_id: str, result: str | bool, reason: str | None = None, wait: bool = False) -> list[str]:
        """Verifier signal. result: pass|fail (or True/False); 'error' (timeout, 429...) is recorded but ignored.
        Rebuilds the run's family: before returning in sync mode; in background mode on the worker
        (`wait=True` blocks until that build is done)."""
        if isinstance(result, bool):
            result = "pass" if result else "fail"
        result = {"passed": "pass", "success": "pass", "ok": "pass", "failed": "fail", "failure": "fail"}.get(result, result)
        if result not in ("pass", "fail", "error"):
            raise ValueError(f"outcome must be pass|fail|error, got {result!r}")
        ids = self.store.set_outcome(run_id, result, reason)
        fams = sorted({r["family"] for r in (self.store.run(i) for i in ids) if r is not None})
        rb = self._rebuilder if self.rebuild_mode == "background" else None
        if rb is None:
            for f in fams:
                self.rebuild(f)
        else:
            gens = [(f, rb.request(f)) for f in fams]
            if wait:
                for f, g in gens:
                    rb.wait(f, g)
        return ids

    # ------------------------------------------------------------------ helpers
    def _run_id(self, hdrs: dict, fam: str, ep: Episode, task_hash: str) -> str | None:
        hdr = hdrs.get(RUN_HEADER)
        if hdr:
            row = self.store.run(hdr)
            if row is None or row["task_hash"] == task_hash:
                return hdr
            return f"{hdr}.{task_hash[:8]}"
        if ep.steps:
            return "r_" + h(fam, task_hash, ep.steps[0].call.id)
        return None

    def _record(self, run_id: str, fam: str, ep: Episode, task_hash: str) -> None:
        """Log the episode's steps. A T2 pick counts as a model choice (the model chose among
        known children), so it is recorded as not replayed and feeds purity and decision lists."""
        self.store.upsert_run(run_id, fam, ep.task, task_hash)
        start = max(0, self.store.n_steps(run_id) - 1)
        rows = []
        for i in range(start, len(ep.steps)):
            st = ep.steps[i]
            rows.append((i, st.call.id, st.call.name, dumps(st.call.args), st.obs.text if st.obs else None,
                         int(st.obs.is_error) if st.obs else 0, int(bool(st.replayed_node) and st.replayed_via != "t2")))
        if rows:
            self.store.write_steps(run_id, start, rows)

    @staticmethod
    def _headers(run_id: str | None, tier: str) -> dict[str, str]:
        out = {"x-treejit-tier": tier}
        if run_id:
            out["x-treejit-run"] = run_id
        return out

    def wrap(self, client: Any, run_id: str | None = None, dialect: str | None = None) -> Any:
        from .inline import wrap_client

        return wrap_client(self, client, run_id=run_id, dialect=dialect)

    def stats(self, family: str | None = None) -> dict[str, Any]:
        where, args = ("WHERE family=?", (family,)) if family else ("WHERE family IS NOT NULL", ())
        rows = self.store.q(f"SELECT tier, COUNT(*) n, SUM(n_calls) calls, SUM(input_tokens+output_tokens+cache_read+cache_write) tok "
                            f"FROM requests {where} GROUP BY tier", args)
        return {r["tier"]: {"requests": r["n"], "tool_calls": r["calls"] or 0, "tokens": r["tok"] or 0} for r in rows}

    def close(self) -> None:
        if self._rebuilder is not None:
            self._rebuilder.stop()
        self.store.close()


_gc_lock = threading.Lock()
_gc_holds = 0


@contextlib.contextmanager
def _no_gc() -> Iterator[None]:
    """The cyclic GC paused during a build. A build allocates ~10^5 short-lived, acyclic objects
    (freed by reference counting), so it only triggers full collections that pause every thread
    for 50-150 ms, stalling the proxy's event loop, and make the build ~25% slower."""
    global _gc_holds
    with _gc_lock:
        if _gc_holds == 0 and not gc.isenabled():
            _gc_holds = -1  # someone else disabled it: leave it alone
        if _gc_holds >= 0:
            _gc_holds += 1
            gc.disable()
    try:
        yield
    finally:
        with _gc_lock:
            if _gc_holds > 0:
                _gc_holds -= 1
                if _gc_holds == 0:
                    gc.enable()
            elif _gc_holds == -1:
                _gc_holds = 0


class _Rebuilder:
    """Background rebuilds for one TreeJIT: a single worker thread, coalescing per family.

    Each request bumps the family's wanted generation; the worker builds the family at the newest
    generation wanted when it starts, so any number of outcomes during a build cost one more build.
    The worker uses its own SQLite connection (WAL: requests keep reading while it builds) and swaps
    the family's view in once the build is committed."""

    def __init__(self, jit: TreeJIT) -> None:
        self.jit = jit
        self.cv = threading.Condition()
        self.want: dict[str, int] = {}
        self.done: dict[str, int] = {}
        self.thread: threading.Thread | None = None
        self.stopping = False
        self.store: Store | None = None

    def pending(self, family: str) -> bool:
        return self.want.get(family, 0) > self.done.get(family, 0)

    def request(self, family: str) -> int:
        with self.cv:
            gen = self.want[family] = self.want.get(family, 0) + 1
            if self.thread is None or not self.thread.is_alive():
                self.stopping = False
                self.thread = threading.Thread(target=self._run, name="treejit-rebuild", daemon=True)
                self.thread.start()
            self.cv.notify_all()
            return gen

    def wait(self, family: str, gen: int, timeout: float | None = None) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.done.get(family, 0) >= gen or self.stopping, timeout)

    def wait_all(self, timeout: float | None = None) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.stopping or not any(self.pending(f) for f in self.want), timeout)

    def stop(self) -> None:
        """Finish the build in progress (if any) and end the worker; pending families stay dirty."""
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join()
        if self.store is not None:
            self.store.close()
            self.store = None

    def _run(self) -> None:
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.stopping or any(self.pending(f) for f in self.want))
                if self.stopping:
                    return
                family = next(f for f in self.want if self.pending(f))
                gen = self.want[family]
            try:
                self.jit._build(self._store(), family)
            except Exception:  # the family stays dirty; the next request for it asks again
                log.exception("treejit: background rebuild of %s failed", family)
            with self.cv:
                self.done[family] = max(self.done.get(family, 0), gen)
                self.cv.notify_all()

    def _store(self) -> Store:
        main = self.jit.store
        if main.path == ":memory:":
            return main  # a single connection; the store lock serializes its users
        if self.store is None:
            self.store = Store(main.path)
        return self.store
