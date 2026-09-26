"""Frontier prefix compaction (opt-in, `Config.compact`).

When a request goes to the model (T4), treejit owns the replayed part of the
prefix. In the forwarded copy only (the harness's history is never touched) the
raw observation of a *verified replayed step* is replaced by a one/two-line
deterministic digest:

    [treejit: replayed & verified step — Bash(python -m pytest -q) → ok, 21 lines, 1001 chars; first line: "…… [ 34%]"; last line: "233 passed, 5 warnings in 6.93s"]
    [output elided by treejit; call the tool again if you need it]

Rules
  1. eligible: the call id carries the replay marker, the edge is known at the
     node encoded in the id, the observation satisfies that edge's (non-empty)
     learned postcondition, and its features show no error;
  2. the last `compact_keep_last` observations are always sent in full;
  3. observations a decision depends on are kept (see `depended_on`):
       a. every observation a binding rule, guard or decision list of any child
          of a *current* frontier context reads (["x", ["obs", k], ...], "case",
          guards/stumps read the last observation);
       b. every observation a binding rule of an edge *on the path* read to render
          its step's arguments (the rule of the step's edge at each context of
          that step, and at the node encoded in a replayed id);
     "arg" rules read call arguments, which are never compacted;
  4. observations shorter than `compact_min_chars` are left alone.

Determinism: the digest is a pure function of (call, observation). Decisions are
sticky: once a call id has been compacted the digest is stored (keyed by call id +
observation hash) and reused verbatim on every later request, even if the tree was
rebuilt in between, so a compacted step never flips back to full and the provider's
prompt cache keeps the prefix. The only exception is rule 2/3a (the current
decision needs it), which can only happen if the harness rewinds the conversation
or `compact_keep_last` is smaller than the binding reach (3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import Config
from .features import guard_holds, obs_features, parse_json
from .model import NormRequest, Observation, ToolCall
from .store import Store
from .tree import NodeEdge, TreeView, contexts, node_id
from .util import h

HEAD_CHARS = 120
LABEL_CHARS = 80
JSON_CHARS = 160


@dataclass
class Compaction:
    body: dict
    n: int = 0          # observations compacted in this request
    chars: int = 0      # characters removed (original minus digest)

    @property
    def note(self) -> str:
        return f"compacted {self.n} obs/{self.chars} chars" if self.n else ""


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


def apply(store: Store, view: TreeView, cfg: Config, req: NormRequest, body: dict) -> Compaction:
    """Compact a forwarded request body. `body` is not mutated (neither is req.raw)."""
    steps = req.episode.steps
    n = len(steps)
    keep = max(0, cfg.compact_keep_last)
    if n <= keep:
        return Compaction(body)
    eids, _ = view.recognize(req.episode)
    current, path = depended_on(view, cfg, eids, [s.replayed_node for s in steps])
    sticky = store.compactions([s.call.id for s in steps[: n - keep] if s.replayed_node])
    out: dict[str, str] = {}
    new_rows: dict[str, tuple] = {}
    lens: dict[str, int] = {}
    for i, st in enumerate(steps[: n - keep]):
        if i in current or st.obs is None or not st.replayed_node:
            continue
        oh = h(st.obs.text, int(st.obs.is_error))
        prev = sticky.get(st.call.id)
        if prev is not None and prev[0] == oh:
            out[st.call.id] = prev[1]
        elif i not in path and len(st.obs.text) >= cfg.compact_min_chars and _verified(view, st, eids[i]):
            d = digest(st.call, st.obs, cfg)
            if len(d) >= len(st.obs.text):
                continue
            out[st.call.id] = d
            new_rows[st.call.id] = (st.call.id, oh, d, len(st.obs.text) - len(d))
        else:
            continue
        lens[st.call.id] = len(st.obs.text)
    if not out:
        return Compaction(body)
    new_body, done = _rewrite(req.dialect, body, out)
    rows = [r for cid, r in new_rows.items() if cid in done]
    if rows:
        store.save_compactions(rows)
    return Compaction(new_body, len(done), sum(lens[c] - len(out[c]) for c in done))


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


def _rewrite(dialect: str, body: dict, out: dict[str, str]) -> tuple[dict, set[str]]:
    """Copy-on-write replacement of tool results by call id. Returns (body, ids replaced)."""
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
