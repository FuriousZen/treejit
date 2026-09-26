"""The replay decision for one request.

Climb only as high as needed:
  T0  one confident child whose guard holds          -> replay, zero model calls
  T1  several children, a learned predicate picks one -> replay, zero model calls
  T4  anything else                                   -> forward to the model (with node-local hints)
(T2 checkpoints and T3 hole-filling are not implemented yet; those cases go to T4.)

Bounds, always on: side exits when a replayed step's postcondition fails, a
confidence budget (product of edge confidences since the last model call must
stay above theta), a hard cap K on consecutive replayed steps, the read-only
allowlist, and commit points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bindings import Sources, eval_rule
from .config import Config
from .dialects import Dialect
from .features import eval_decision_list, guard_holds, obs_features, task_words
from .model import NormRequest, ToolCall
from .policy import is_commit_point, is_readonly
from .templates import call_slots, render
from .tree import NodeEdge, TreeView, contexts, node_id


@dataclass
class Plan:
    calls: list[ToolCall] = field(default_factory=list)
    tier: str = "T4"
    node: str | None = None           # node the (first) decision was made at
    nodes: list[str] = field(default_factory=list)
    reason: str = ""
    conf: float = 1.0
    detail: list[str] = field(default_factory=list)


def _choose(view: TreeView, cfg: Config, nid: str, kids: list[NodeEdge], feats: dict | None, text: str,
            words: set, pending: bool) -> tuple[NodeEdge | None, str, float, str]:
    """Returns (edge, tier, confidence, why-not)."""
    best = max(kids, key=lambda ne: (ne.pass_n, ne.conf, ne.edge))
    if best.purity >= cfg.purity:
        if best.tomb:
            return None, "", 0.0, "tombstoned"
        if not best.replayable:
            return None, "", 0.0, f"not_replayable:{best.tier}" + (":holes" if best.holes else "")
        if feats is not None and best.guard and not guard_holds(best.guard, feats):
            return None, "", 0.0, "guard"
        return best, "T0", best.conf, ""
    if pending:
        return None, "", 0.0, "ambiguous"
    dl = view.stumps.get(nid)
    if dl:
        leaf = eval_decision_list(dl, feats or {}, text, words)
        if leaf and leaf["purity"] >= cfg.purity and leaf["n"] >= 2:
            ne = next((k for k in kids if k.edge == leaf["edge"]), None)
            if ne is not None and ne.replayable and not ne.tomb:
                if feats is not None and ne.guard and not guard_holds(ne.guard, feats):
                    return None, "", 0.0, "guard"
                return ne, "T1", leaf["purity"] * ne.success, ""
            return None, "", 0.0, "branch_not_replayable"
    return None, "", 0.0, "ambiguous"


def decide(view: TreeView, cfg: Config, req: NormRequest, dialect: Dialect) -> Plan:
    ep = req.episode
    plan = Plan()
    if not ep.ready:
        plan.reason = "not_ready"
        return plan
    eids, slots = view.recognize(ep)
    steps = ep.steps

    # trailing run of replayed steps: side exits and the confidence budget
    trail = 0
    while trail < len(steps) and steps[len(steps) - 1 - trail].replayed_node:
        trail += 1
    budget = 1.0
    for idx in range(len(steps) - trail, len(steps)):
        st = steps[idx]
        ne = view.child(st.replayed_node, eids[idx]) if eids[idx] else None
        if ne is None:
            plan.reason = "replayed_edge_unknown"
            return plan
        if st.obs is not None and ne.post and not guard_holds(ne.post, obs_features(st.obs)):
            plan.reason, plan.node = "side_exit", st.replayed_node
            plan.detail.append(f"postcondition {ne.post} failed after step {idx}")
            return plan
        c = st.replayed_conf
        budget *= ne.conf if c is None else c

    words = task_words(ep.task)
    calls = [s.call for s in steps]
    obs = [s.obs for s in steps]
    count = trail
    while True:
        i = len(calls)
        pending = bool(plan.calls)  # the previous call in this batch has no observation yet
        ctxs = contexts(eids, i, cfg)
        feats = None if pending else (obs_features(obs[-1]) if i > 0 else {})
        text = "" if pending or i == 0 or obs[-1] is None else obs[-1].text
        choice, why, nid, used = None, "off_tree" if not ctxs else "no_evidence", None, ""
        seen_here: list[set[str]] = []  # model choices at more specific contexts we backed off from
        for kind, ctx in ctxs:
            cand = node_id(view.family, kind, ctx)
            kids = view.children.get(cand)
            if not kids:
                continue
            if plan.node is None and not plan.calls:
                plan.node = cand
            if view.node_pass.get(cand, 0) < cfg.promote_runs:
                # too little evidence here; a less specific context may know more,
                # as long as what it proposes doesn't contradict what little we saw here
                chosen = {k.edge for k in kids if k.pass_n > 0}
                if chosen:
                    seen_here.append(chosen)
                continue
            nid, used = cand, f"{kind}{len(ctx)}"
            ne, tier, conf, why = _choose(view, cfg, cand, kids, feats, text, words, pending)
            if ne is not None:
                if any(ne.edge not in chosen for chosen in seen_here):
                    why = "backoff_disagrees"
                else:
                    choice = (ne, tier, conf)
            break
        if choice is None:
            plan.reason = plan.reason or (why + (f"@{used}" if used else ""))
            break
        ne, tier, conf = choice
        edge = view.edges[ne.edge]
        S = Sources(ep.task, calls, obs, slots)
        values = {}
        for slot, rule in ne.bindings.items():
            v = eval_rule(rule, S) if rule is not None else None
            if v is None:
                why = f"unbound:{slot}"
                break
            values[slot] = v
        else:
            why = ""
        if why:
            plan.reason = plan.reason or why
            break
        try:
            args = render(edge.template, ne.ref, values)
        except (KeyError, ValueError) as e:
            plan.reason = plan.reason or f"render:{e}"
            break
        if is_readonly(edge.tool, ne.ref, cfg) and not is_readonly(edge.tool, args, cfg):
            plan.reason = plan.reason or "unsafe_args"
            break
        if is_commit_point(edge.tool, args, cfg) and not ne.commit_point:
            plan.reason = plan.reason or "commit_point"
            break
        if count >= cfg.hard_cap:
            plan.reason = plan.reason or "hard_cap"
            break
        if budget * conf < cfg.theta:
            plan.reason = plan.reason or "budget"
            break
        call = ToolCall(dialect.new_call_id(nid, conf), edge.tool, args)
        plan.calls.append(call)
        plan.nodes.append(nid)
        plan.tier = tier if len(plan.calls) == 1 else plan.tier
        plan.detail.append(f"{tier}@{used} {edge.label} conf={conf:.2f}")
        budget *= conf
        plan.conf = budget
        count += 1
        calls = calls + [call]
        obs = obs + [None]
        eids = eids + [ne.edge]
        slots = slots + [call_slots(edge.template, call)]
        if not cfg.batch or len(plan.calls) >= cfg.max_batch or not is_readonly(edge.tool, args, cfg):
            break
    if plan.calls:
        plan.reason = ""
    return plan


def hints(view: TreeView, cfg: Config, nid: str | None) -> str | None:
    """Node-local frontier hints: this node's failures (with reasons) and known-good children."""
    if cfg.hints == "off" or not nid:
        return None
    kids = view.children.get(nid, [])
    bad = sorted([k for k in kids if k.tomb or k.blamed > 0], key=lambda k: -k.blamed)
    if cfg.hints == "failures" and not bad:
        return None
    lines = []
    for k in bad[: cfg.hint_max]:
        reason = f" (reason: {k.reasons[0]})" if k.reasons else ""
        lines.append(f"- AVOID {view.edges[k.edge].label}: failed in {k.fail_runs} earlier run(s){reason}")
    good = sorted([k for k in kids if k.pass_runs and not k.tomb], key=lambda k: -k.pass_runs)
    for k in good[: max(1, cfg.hint_max - len(lines))]:
        lines.append(f"- worked before: {view.edges[k.edge].label} ({k.pass_runs} successful run(s))")
    if not lines:
        return None
    return "<treejit-hints>\nAt this step in similar earlier tasks:\n" + "\n".join(lines) + "\n</treejit-hints>"

