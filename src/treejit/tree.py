"""The live tree as seen at request time: contexts, node ids, and recognition.

A *context* is what a decision is conditioned on:
  ("r", edges)  the exact edge path from the episode root (depth <= D)
  ("g", edges)  the last k edges, for each k in cfg.ngram (macros)
Root contexts give a tree; n-gram contexts make recurring sub-paths (git add ->
git commit -> git push) reachable from any task whose recent history matches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .model import Episode, ToolCall
from .store import Store
from .templates import call_slots, shape_of
from .util import h


def node_id(family: str, kind: str, ctx: tuple) -> str:
    return h(family, kind, list(ctx), n=12)


def contexts(eids: list[str | None], i: int, cfg: Config) -> list[tuple[str, tuple]]:
    """Contexts for the decision at step i, most specific first."""
    out: list[tuple[str, tuple]] = []
    prefix = eids[:i]
    if i <= cfg.max_depth and all(e is not None for e in prefix):
        out.append(("r", tuple(prefix)))
    for k in cfg.ngram:
        if 0 < k <= i:
            g = eids[i - k : i]
            if all(e is not None for e in g):
                out.append(("g", tuple(g)))
    return out


@dataclass
class EdgeInfo:
    id: str
    tool: str
    shape: str
    template: dict
    label: str


@dataclass
class NodeEdge:
    node: str
    edge: str
    n: int
    pass_runs: int
    fail_runs: int
    pass_n: int
    blamed: float
    tomb: bool
    live: bool
    replayable: bool
    tier: str
    purity: float
    success: float
    conf: float
    bindings: dict
    holes: list
    guard: dict
    post: dict
    ref: dict
    reasons: list
    commit_point: bool
    savings: float
    latency_ms: float
    score: float

    @classmethod
    def from_row(cls, r: Any) -> "NodeEdge":
        return cls(
            r["node"], r["edge"], r["n"], r["pass_runs"], r["fail_runs"], r["pass_n"], r["blamed"], bool(r["tomb"]),
            bool(r["live"]), bool(r["replayable"]), r["tier"], r["purity"], r["success"], r["conf"],
            json.loads(r["bindings"] or "{}"), json.loads(r["holes"] or "[]"), json.loads(r["guard"] or "{}"),
            json.loads(r["post"] or "{}"), json.loads(r["ref"] or "{}"), json.loads(r["reasons"] or "[]"),
            bool(r["commit_point"]), r["savings"] or 0.0, r["latency_ms"] or 0.0, r["score"] or 0.0,
        )


@dataclass
class TreeView:
    family: str
    edges: dict[str, EdgeInfo] = field(default_factory=dict)
    by_shape: dict[str, str] = field(default_factory=dict)
    children: dict[str, list[NodeEdge]] = field(default_factory=dict)
    stumps: dict[str, dict] = field(default_factory=dict)
    node_pass: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, store: Store, family: str) -> "TreeView":
        v = cls(family)
        for r in store.q("SELECT * FROM edges WHERE family=?", (family,)):
            e = EdgeInfo(r["id"], r["tool"], r["shape"], json.loads(r["template"]), r["label"])
            v.edges[e.id] = e
            v.by_shape[e.shape] = e.id
        for r in store.q("SELECT * FROM node_edges WHERE family=?", (family,)):
            v.children.setdefault(r["node"], []).append(NodeEdge.from_row(r))
        for r in store.q("SELECT id, stump, n_pass FROM nodes WHERE family=?", (family,)):
            v.node_pass[r["id"]] = r["n_pass"] or 0
            if r["stump"]:
                v.stumps[r["id"]] = json.loads(r["stump"])
        return v

    def match(self, call: ToolCall) -> tuple[str | None, dict | None]:
        eid = self.by_shape.get(shape_of(call))
        if eid is None:
            return None, None
        slots = call_slots(self.edges[eid].template, call)
        return (eid, slots) if slots is not None else (None, None)

    def recognize(self, ep: Episode) -> tuple[list[str | None], list[dict | None]]:
        eids, slots = [], []
        for st in ep.steps:
            e, s = self.match(st.call)
            eids.append(e)
            slots.append(s)
        return eids, slots

    def child(self, node: str, edge: str) -> NodeEdge | None:
        for ne in self.children.get(node, []):
            if ne.edge == edge:
                return ne
        return None
