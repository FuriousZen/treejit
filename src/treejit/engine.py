"""The engine: one core shared by proxy mode and inline mode.

    jit = TreeJIT("treejit.db")
    res = jit.handle("anthropic", body, headers)   # replay, or a body to forward
    ...forward res.body upstream if res.kind == "forward"...
    jit.complete(res, response_info, status, latency_ms)
    jit.outcome(run_id, "pass")                     # verifier signal; rebuilds the tree
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import dialects, families
from .builder import build_family
from .config import Config
from .model import Episode, ResponseInfo
from .replay import Plan, decide, hints
from .store import Store, dumps
from .tree import TreeView
from .util import h

RUN_HEADER = "x-treejit-run"


@dataclass
class _Pending:
    request_id: int
    family: str | None
    task: str
    task_hash: str
    run_id: str | None
    started: float


@dataclass
class Result:
    kind: str                         # "replay" | "forward"
    body: dict | None                 # replay: response JSON (non-stream); forward: request body to send
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
        self._views: dict[str, TreeView] = {}

    # ------------------------------------------------------------------ tree
    def view(self, family: str) -> TreeView:
        row = self.store.q1("SELECT dirty FROM families WHERE id=?", (family,))
        if row is not None and row["dirty"]:
            self.rebuild(family)
        if family not in self._views:
            self._views[family] = TreeView.load(self.store, family)
        return self._views[family]

    def rebuild(self, family: str | None = None) -> list[dict]:
        fams = [family] if family else [r["id"] for r in self.store.q("SELECT id FROM families")]
        out = []
        for f in fams:
            out.append(build_family(self.store, self.cfg, f))
            self._views.pop(f, None)
        return out

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
            if run_id is None:
                run_id = "r_" + h(fam, task_hash, plan.calls[0].id)
                self.store.upsert_run(run_id, fam, ep.task, task_hash)
            for nid in plan.nodes:
                self.store.hit(nid)
            if req.stream:
                sse, resp = d.build_sse(req.model, plan.calls, body), None
            else:
                sse, resp = None, d.build_response(req.model, plan.calls)
            ms = (time.perf_counter() - t0) * 1000
            self.store.log_request(family=fam, run_id=run_id, dialect=dialect, tier=plan.tier, node=plan.nodes[0],
                                   n_calls=len(plan.calls), latency_ms=ms, status=200,
                                   call_ids=json.dumps([c.id for c in plan.calls]), note="; ".join(plan.detail)[:500])
            return Result("replay", resp, req.stream, sse, run_id, plan.tier, plan, self._headers(run_id, plan.tier))

        fwd = d.prepare_forward(req)
        hint = hints(view, self.cfg, plan.node)
        if hint:
            fwd = d.inject_hint(fwd, hint)
        if plan.node:
            self.store.hit(plan.node)
        note = plan.reason + ("; " + "; ".join(plan.detail) if plan.detail else "") + ("; hints" if hint else "")
        rid = self.store.log_request(family=fam, run_id=run_id, dialect=dialect, tier="T4", node=plan.node, note=note[:500])
        return Result("forward", fwd, req.stream, None, run_id, "T4", plan, self._headers(run_id, "T4"),
                      _Pending(rid, fam, ep.task, task_hash, run_id, t0))

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

    def outcome(self, run_id: str, result: str | bool, reason: str | None = None) -> list[str]:
        """Verifier signal. result: pass|fail (or True/False); 'error' (timeout, 429...) is recorded but ignored."""
        if isinstance(result, bool):
            result = "pass" if result else "fail"
        result = {"passed": "pass", "success": "pass", "ok": "pass", "failed": "fail", "failure": "fail"}.get(result, result)
        if result not in ("pass", "fail", "error"):
            raise ValueError(f"outcome must be pass|fail|error, got {result!r}")
        ids = self.store.set_outcome(run_id, result, reason)
        fams = {r["family"] for r in (self.store.run(i) for i in ids) if r is not None}
        for f in fams:
            self.rebuild(f)
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
        self.store.upsert_run(run_id, fam, ep.task, task_hash)
        start = max(0, self.store.n_steps(run_id) - 1)
        rows = []
        for i in range(start, len(ep.steps)):
            st = ep.steps[i]
            rows.append((i, st.call.id, st.call.name, dumps(st.call.args), st.obs.text if st.obs else None,
                         int(st.obs.is_error) if st.obs else 0, int(bool(st.replayed_node))))
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
        self.store.close()
