"""Frontier prefix compaction (opt-in, `Config.compact`).

When a request goes to the model (T4), treejit owns the replayed part of the
prefix. In the forwarded copy only (the harness's history is never touched) the
raw observation of a *verified replayed step* is replaced by a one/two-line
deterministic digest:

    [treejit: replayed & verified step — Bash(python -m pytest -q) → ok, 21 lines, 1001 chars; first line: "…… [ 34%]"; last line: "233 passed, 5 warnings in 6.93s"]
    [output elided by treejit; call the tool again if you need it]

Eligibility (all modes)
  1. the call id carries the replay marker, the edge is known at the node encoded
     in the id, the observation satisfies that edge's (non-empty) learned
     postcondition, and its features show no error;
  2. the last `compact_keep_last` observations of the request are sent in full;
  3. observations a decision depends on are kept (see `depended_on`):
       a. "current": every observation a binding rule, guard or decision list of
          any child of a frontier context reads (["x", ["obs", k], ...], "case",
          guards/stumps read the last observation). Only matters when
          `compact_keep_last` < 3, the binding reach;
       b. "path" (opt-in, `compact_keep_path`): every observation a binding rule of
          an edge on the path read to render its step's arguments. Off by default:
          decisions are made on the harness's uncompacted body, so this protects
          nothing (PLAN C2);
  4. observations shorter than `compact_min_chars` are left alone.

When (`compact_mode`)
  first_sight (default): append-only. A step may be compacted only in the *first*
      forwarded request that contains it. Once it went upstream in full it stays
      full, and once compacted it stays compacted, so consecutive forwards of one
      conversation are byte-identical up to the previous request's last message and
      the provider's prompt cache keeps the whole prefix. Every decision is stored
      per call id in `compactions` (a NULL digest = "sent in full"). When no row
      exists (pruned, another instance, compaction just enabled) the decision is
      reconstructed from the conversation itself: the first forward that contained
      step i is the request just before the first model-chosen step after i (or
      this request), so step i was inside that request's keep-last window iff
      i >= f - keep. In practice first_sight compacts the replayed steps of a
      burst (more than `compact_keep_last` steps replayed between two frontier
      calls) and nothing else.
  epoch: first_sight, plus a re-compaction when the conversation's cache is cold:
      if the previous forward of this conversation is older than
      `compact_epoch_ttl` seconds (the cache entry has expired, so the next
      request rewrites the whole prefix anyway), the request is compacted as in
      `window` mode and the result becomes the new sticky state.
  window: the pre-C1 behaviour. The keep-last window moves with every request and
      the step that leaves it is compacted, which changes an earlier message on
      every forward. Fewest raw tokens, but it defeats prompt caching (+123% billed
      in the C1 model): only for providers without prompt caching.

Preserved thinking (Claude Fable 5.1, Opus 5.5): a thinking block is bound to the exact
prefix that produced it, and on accounts created on or after 2026-08-31 a request whose
earlier history changed is a 400. first_sight is compatible (append-only, byte-stable).
`window` is not (it edits an earlier message on every forward), and neither is `epoch`:
its re-compaction of a cold conversation rewrites earlier tool results, which invalidates
every later thinking block even though the cache is cold anyway. Use first_sight with
those models.

Sticky hints (`sticky_hints`, below) follow the same rule for frontier hints: a hint given
after history item i is re-inserted at i, identically, in every later forward of the
conversation.

Determinism: the digest is a pure function of (call, observation) and decisions are
sticky across tree rebuilds and restarts (keyed by call id + observation hash).
Steps of earlier episodes of the same conversation (before the last user text
message) reuse their stored digests too. See tests/cache_model.py for the prompt
cache cost model these modes were chosen with.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .config import Config
from .dialects import Dialect, _text_of
from .features import guard_holds, obs_features, parse_json
from .model import NormRequest, Observation, Step, ToolCall
from .store import Store
from .tree import NodeEdge, TreeView, contexts, node_id
from .util import h, now

MODES = ("first_sight", "epoch", "window")
PRUNE_EVERY = 3600.0   # opportunistic pruning of old compaction rows, at most once an hour per store

HEAD_CHARS = 120
LABEL_CHARS = 80
JSON_CHARS = 160


@dataclass
class Compaction:
    body: dict
    n: int = 0          # observations compacted in this request
    chars: int = 0      # characters removed (original minus digest)
    epoch: bool = False  # epoch mode: this request re-compacted a cold conversation

    @property
    def note(self) -> str:
        return (f"compacted {self.n} obs/{self.chars} chars" + (" (epoch)" if self.epoch else "")) if self.n else ""


# ------------------------------------------------------------------ digest


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def call_label(call: ToolCall, cfg: Config) -> str:
    cmd = call.args.get("command") if call.name in cfg.shell_tools else None
    if isinstance(cmd, str):
        inner = " ".join(cmd.split())
    else:
        inner = ", ".join(f"{k}={json.dumps(call.args[k], ensure_ascii=False, sort_keys=True)}" for k in sorted(call.args))
    return f"{call.name}({_clip(inner, LABEL_CHARS)})"


def _scalar(v: Any) -> str:
    if isinstance(v, dict):
        return "{…}" if v else "{}"
    if isinstance(v, list):
        return f"[{len(v)}]"
    if isinstance(v, str):
        return json.dumps(_clip(v, 40), ensure_ascii=False)
    return json.dumps(v)


def digest(call: ToolCall, obs: Observation, cfg: Config) -> str:
    """One/two-line deterministic summary of a verified replayed step."""
    text = obs.text or ""
    f = obs_features(obs)
    status = "error" if f.get("err") else "ok"
    if "exit" in f:
        status += f" (exit {f['exit']})"
    js = parse_json(text)
    if isinstance(js, dict):
        detail = "json {" + ", ".join(f"{json.dumps(str(k), ensure_ascii=False)}: {_scalar(v)}" for k, v in js.items()) + "}"
        detail = _clip(detail, JSON_CHARS)
    elif isinstance(js, list):
        detail = f"json list of {len(js)}"
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        detail = "first line: " + json.dumps(_clip(lines[0] if lines else "", HEAD_CHARS), ensure_ascii=False)
        if len(lines) > 1:
            detail += "; last line: " + json.dumps(_clip(lines[-1], HEAD_CHARS), ensure_ascii=False)
    n_lines = len(text.splitlines())
    return (f"[treejit: replayed & verified step — {call_label(call, cfg)} → {status}, {n_lines} lines, {len(text)} chars; {detail}]\n"
            "[output elided by treejit; call the tool again if you need it]")


# ------------------------------------------------------------------ eligibility


def obs_reads(rule: Any) -> set[int]:
    """Observation offsets k (obs[-k]) a binding rule reads."""
    if not isinstance(rule, list) or not rule:
        return set()
    kind = rule[0]
    if kind == "x":
        src = rule[1]
        return {int(src[1])} if isinstance(src, list) and src and src[0] == "obs" else set()
    if kind == "case":
        return {1}
    if kind == "fmt":
        out: set[int] = set()
        for p in rule[1]:
            out |= obs_reads(p)
        return out
    return set()


def _edge_reads(ne: NodeEdge, guard: bool) -> set[int]:
    out: set[int] = set()
    for rule in ne.bindings.values():
        out |= obs_reads(rule)
    if guard and ne.guard:
        out.add(1)
    return out


def depended_on(view: TreeView, cfg: Config, eids: list[str | None], replayed: list[str | None]) -> tuple[set[int], set[int]]:
    """(current, path): step indices whose observation the current decision (rule 3a)
    or a path edge's bindings (rule 3b) read."""
    n = len(eids)
    current: set[int] = {n - 1} if n else set()
    for kind, ctx in contexts(eids, n, cfg):
        nid = node_id(view.family, kind, ctx)
        for ne in view.children.get(nid, []):
            current |= {n - k for k in _edge_reads(ne, guard=True) if 0 < k <= n}
    path: set[int] = set()
    for i, e in enumerate(eids):
        if e is None:
            continue
        nodes = [node_id(view.family, kind, ctx) for kind, ctx in contexts(eids, i, cfg)]
        if replayed[i]:
            nodes.append(replayed[i])
        for nid in nodes:
            ne = view.child(nid, e)
            if ne is not None:
                path |= {i - k for k in _edge_reads(ne, guard=False) if 0 < k <= i}
    return current, path


def _verified(view: TreeView, st: Any, eid: str | None) -> bool:
    if not st.replayed_node or st.obs is None or eid is None:
        return False
    ne = view.child(st.replayed_node, eid)
    if ne is None or not ne.post:
        return False
    f = obs_features(st.obs)
    return not f.get("err") and guard_holds(ne.post, f)


# ------------------------------------------------------------------ apply


def apply(store: Store, view: TreeView, cfg: Config, req: NormRequest, body: dict, clock: float | None = None) -> Compaction:
    """Compact a forwarded request body. `body` is not mutated (neither is req.raw).
    `clock` overrides the current time (epoch mode and pruning; tests and the cost model)."""
    t = now() if clock is None else clock
    maybe_prune(store, cfg, t)
    mode = cfg.compact_mode
    if mode not in MODES:
        raise ValueError(f"compact_mode must be one of {MODES}, got {mode!r}")
    steps = req.episode.steps
    n = len(steps)
    keep = max(0, cfg.compact_keep_last)
    results = _results(req, body)
    ep_ids = {st.call.id for st in steps}
    cold = False
    if mode == "epoch":
        # a conversation is identified by its first tool call id (earlier episodes included). Every
        # forward refreshes its timestamp, so "cold" means: no forward of it within the cache TTL.
        conv = next(iter(results), None) or (steps[0].call.id if steps else None)
        if conv is not None:
            last = store.touch_conversation(conv, t)
            cold = last is not None and t - last > cfg.compact_epoch_ttl
    # tool results of replayed calls from earlier episodes of this conversation (before the last user text)
    prior = {cid: v for cid, v in results.items() if cid not in ep_ids and Step(ToolCall(cid, "", {})).replayed_node}
    cand = [i for i, st in enumerate(steps) if st.replayed_node and st.obs is not None]
    if not cand and not prior:
        return Compaction(body)
    rows = store.compactions([steps[i].call.id for i in cand] + list(prior))
    out: dict[str, str] = {}
    lens: dict[str, int] = {}
    # earlier episodes: reuse their stored digests verbatim (never decide anew)
    for cid, (text, err) in prior.items():
        r = rows.get(cid)
        if r is not None and r[1] is not None and r[0] == h(text, int(err)):
            out[cid], lens[cid] = r[1], len(text)
    new_rows: dict[str, tuple] = {}
    if cand:
        eids, _ = view.recognize(req.episode)
        replayed = [s.replayed_node for s in steps]
        deps: dict[int, tuple[set[int], set[int]]] = {}

        def kept_for(i: int, f: int) -> bool:
            """Rule 3 as seen by the forward of the first f steps."""
            if f not in deps:
                deps[f] = depended_on(view, cfg, eids[:f], replayed[:f])
            current, path = deps[f]
            return i in current or (cfg.compact_keep_path and i in path)

        fwd_points = [i for i, st in enumerate(steps) if st.replayed_node is None] + [n]
        for i in cand:
            st = steps[i]
            cid, text = st.call.id, st.obs.text
            oh = h(text, int(st.obs.is_error))
            prev = rows.get(cid)
            prev = prev if prev is not None and prev[0] == oh else None
            if prev is not None and prev[1] is not None and (mode != "window" or not (i >= n - keep or kept_for(i, n))):
                out[cid], lens[cid] = prev[1], len(text)       # sticky: compacted once, compacted for good
                continue
            if mode == "window" or cold:
                f, record = n, cold                           # decide as of this request
            elif prev is not None:
                continue                                      # sticky: sent in full once, full for good
            else:
                f, record = fwd_points[bisect.bisect_right(fwd_points, i)], True
            if len(text) < cfg.compact_min_chars:
                continue                                      # never compacted: no row needed
            d = None
            if i < f - keep and not kept_for(i, f) and _verified(view, st, eids[i]):
                d = digest(st.call, st.obs, cfg)
                if len(d) >= len(text):
                    continue
                out[cid], lens[cid] = d, len(text)
            if record:
                new_rows[cid] = (cid, oh, d, len(text) - len(d) if d else 0)
            elif d is not None:
                new_rows[cid] = (cid, oh, d, len(text) - len(d))
    if out:
        new_body, done = _rewrite(req.dialect, body, out)
    else:
        new_body, done = body, set()
    # a digest counts only if it was actually spliced in (text-only content). In first_sight/epoch a
    # step whose digest could not be spliced went upstream in full, and that is what gets recorded.
    save = [r if r[2] is None or cid in done else (cid, r[1], None, 0) for cid, r in new_rows.items()
            if r[2] is None or cid in done or mode != "window"]
    if save:
        store.save_compactions(save, replace=mode == "epoch" or mode == "window")
    return Compaction(new_body, len(done), sum(lens[c] - len(out[c]) for c in done), cold and bool(done))


def maybe_prune(store: Store, cfg: Config, t: float) -> int:
    """Drop compaction rows older than `compact_retention_days` (at most once per PRUNE_EVERY)."""
    if cfg.compact_retention_days <= 0 or t < getattr(store, "_compact_prune_at", 0.0):
        return 0
    store._compact_prune_at = t + PRUNE_EVERY  # type: ignore[attr-defined]
    return store.prune_compactions(t - cfg.compact_retention_days * 86400)


def _output_text(item: dict) -> str:
    out = item.get("output")
    return out if isinstance(out, str) else _text_of(out) if isinstance(out, list) else json.dumps(out)


def _results(req: NormRequest, body: dict) -> dict[str, tuple[str, bool]]:
    """Every tool result in the body, in order: {call_id: (text, is_error)}."""
    out: dict[str, tuple[str, bool]] = {}
    if req.dialect == "responses":
        for it in body.get("input") if isinstance(body.get("input"), list) else []:
            if isinstance(it, dict) and it.get("type") in _OUTPUTS and isinstance(it.get("call_id"), str):
                out.setdefault(it["call_id"], (_output_text(it), False))
        return out
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if req.dialect == "openai":
            cid = m.get("tool_call_id")
            if m.get("role") == "tool" and isinstance(cid, str):
                out.setdefault(cid, (_text_of(m.get("content")), False))
            continue
        content = m.get("content")
        if m.get("role") != "user" or not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("tool_use_id"), str):
                out.setdefault(b["tool_use_id"], (_text_of(b.get("content")), bool(b.get("is_error"))))
    return out


def _text_blocks_only(content: Any) -> bool:
    return isinstance(content, str) or (isinstance(content, list) and all(
        isinstance(b, str) or (isinstance(b, dict) and b.get("type", "text") == "text") for b in content))


def _replace(content: Any, text: str) -> Any:
    if isinstance(content, str) or content is None:
        return text
    block = {"type": "text", "text": text}
    # a prompt-cache breakpoint on an inner block moves to the digest block (the last one wins, as
    # the breakpoint closest to the end of the original content is the one that covered it)
    marks = [b["cache_control"] for b in content if isinstance(b, dict) and "cache_control" in b]
    if marks:
        block["cache_control"] = marks[-1]
    return [block]


_OUTPUTS = ("function_call_output", "custom_tool_call_output")


def _rewrite(dialect: str, body: dict, out: dict[str, str]) -> tuple[dict, set[str]]:
    """Copy-on-write replacement of tool results by call id. Returns (body, ids replaced)."""
    if dialect == "responses":
        items = body.get("input") if isinstance(body.get("input"), list) else []
        new_items, done = list(items), set()
        for i, it in enumerate(items):
            cid = it.get("call_id") if isinstance(it, dict) else None
            if (isinstance(cid, str) and it.get("type") in _OUTPUTS and cid in out and cid not in done
                    and (isinstance(it.get("output"), str) or (isinstance(it.get("output"), list) and all(
                        isinstance(p, dict) and p.get("type") in ("input_text", "text") for p in it["output"])))):
                o = it.get("output")
                new_items[i] = dict(it, output=out[cid] if isinstance(o, str) else [{"type": "input_text", "text": out[cid]}])
                done.add(cid)
        return (dict(body, input=new_items), done) if done else (body, done)
    msgs = body.get("messages") or []
    new_msgs = list(msgs)
    done: set[str] = set()
    for mi, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        if dialect == "openai":
            cid = m.get("tool_call_id")
            if m.get("role") == "tool" and cid in out and cid not in done and _text_blocks_only(m.get("content")):
                new_msgs[mi] = dict(m, content=_replace(m.get("content"), out[cid]))
                done.add(cid)
            continue
        content = m.get("content")
        if m.get("role") != "user" or not isinstance(content, list):
            continue
        blocks, changed = list(content), False
        for bi, b in enumerate(content):
            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                continue
            cid = b.get("tool_use_id")
            if cid in out and cid not in done and _text_blocks_only(b.get("content")):
                blocks[bi] = dict(b, content=_replace(b.get("content"), out[cid]))
                done.add(cid)
                changed = True
        if changed:
            new_msgs[mi] = dict(m, content=blocks)
    if not done:
        return body, done
    return dict(body, messages=new_msgs), done


# ------------------------------------------------------------------ sticky hints (PLAN H1 / X2)
#
# A frontier hint (replay.hints) used to be appended to the forwarded copy of one request only. The next
# request no longer had it: the provider saw an earlier message change, which moves the prompt-cache prefix
# (+156-247% billed under automatic caching with hints=always) and, on models with preserved thinking,
# invalidates every later thinking block (a 400 on enforced accounts). Now a hint is part of the
# conversation from the moment it is first forwarded: it is stored under an *anchor* (a chained hash of
# the dialect, system prompt, tool names and every history item up to and including the one it follows,
# cache_control markers ignored) and re-inserted, byte-identical, at the same position in every later
# forward whose history has that prefix. A new hint is only ever given after the last item; if that
# position already has one (a retried request, or another conversation with the very same prefix) the
# stored one is reused. Hints unused for `compact_retention_days` are pruned.


def _no_cc(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _no_cc(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_no_cc(v) for v in x]
    return x


def hint_anchors(req: NormRequest, items: list) -> list[str]:
    """Anchor of each history position (see above)."""
    acc = hashlib.sha256(json.dumps(["hints", req.dialect, req.system, [t.get("name") for t in req.tools]],
                                    ensure_ascii=False).encode()).digest()
    out = []
    for it in items:
        acc = hashlib.sha256(acc + json.dumps(_no_cc(it), sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")).encode()).digest()
        out.append(acc.hex()[:32])
    return out


def sticky_hints(store: Store, d: Dialect, req: NormRequest, body: dict, hint: str | None,
                 retention_days: float = 7.0) -> tuple[dict, str | None]:
    """`body` with every hint this conversation was given re-inserted, plus `hint` (if any, and if this
    position has none yet) after its last item. Returns (body, the hint after the last item or None)."""
    items = d.history(req.raw)
    if not items:
        return body, None
    t = now()
    if retention_days > 0 and t >= getattr(store, "_hint_prune_at", 0.0):
        store._hint_prune_at = t + PRUNE_EVERY  # type: ignore[attr-defined]
        store.prune_hints(t - retention_days * 86400)
    anchors = hint_anchors(req, items)
    rows = store.hints(anchors)
    last = anchors[-1]
    if hint and last not in rows and d.inject_hint(body, hint) is not body:
        store.save_hint(last, len(anchors) - 1, hint)
        rows = store.hints(anchors)  # the first hint stored at a position wins
    used = [i for i, a in enumerate(anchors) if a in rows]
    for i in reversed(used):  # from the end: an inserted item never shifts a later insertion point
        body = d.inject_hint(body, rows[anchors[i]], at=i)
    if used:
        store.touch_hints([anchors[i] for i in used], t)
    return body, rows.get(last)
