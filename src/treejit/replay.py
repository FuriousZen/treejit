"""The replay decision for one request.

Climb only as high as needed:
  T0  one confident child whose guard holds          -> replay, zero model calls
  T1  several children, a learned predicate picks one -> replay, zero model calls
  T3  structure known (T0/T1), some values are holes  -> one small call fills them, then replay
  T2  ambiguous node, or confidence budget exhausted  -> one small call picks a known child (or "new")
  T4  anything else                                   -> forward to the model (with node-local hints)

decide() never calls a model: a T2/T3 opportunity comes back as `plan.sub`, and the
engine turns it into a subcall. Subcalls are only made for the first step of a request.

Bounds, always on: side exits when a replayed step's postcondition fails, a
confidence budget (product of edge confidences since the last model call or T2
checkpoint must stay above theta), a hard cap K on consecutive replayed steps,
the read-only allowlist, and commit points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bindings import Sources, eval_rule
from .config import Config
from .dialects import Dialect
from .features import eval_decision_list, guard_holds, obs_features, task_words
from .model import NormRequest, ToolCall
from .policy import is_commit_point, is_readonly
from .templates import Val, call_slots, render
from .tree import EdgeInfo, NodeEdge, TreeView, contexts, node_id

MAX_OPTIONS = 6       # known children offered at a T2 call
MAX_FILL = 8000       # longest value accepted for one hole
MIN_COVER = 0.5       # T2 only if the replayable options account for this share of the model's choices here


@dataclass
class Option:
    """A candidate next call: an edge at a node, with the values the tree could bind."""

    ne: NodeEdge
    edge: EdgeInfo
    values: dict[str, Val]
    holes: list[str]                  # slots the model must supply (no rule, or the rule abstained)
    conf: float = 1.0
    tier: str = ""                    # how the structure was chosen (T0/T1), for fills
    args: dict | None = None          # rendered, when there are no holes


@dataclass
class Subcall:
    kind: str                         # "fill" (T3) | "choose" (T2)
    node: str
    used: str
    options: list[Option]
    reason: str = ""                  # holes | ambiguous | budget

    @property
    def tier(self) -> str:
        return "T3" if self.kind == "fill" else "T2"


@dataclass
class Plan:
    calls: list[ToolCall] = field(default_factory=list)
    tier: str = "T4"
    node: str | None = None           # node the (first) decision was made at
    nodes: list[str] = field(default_factory=list)
    reason: str = ""
    conf: float = 1.0
    detail: list[str] = field(default_factory=list)
    sub: Subcall | None = None        # a T2/T3 opportunity, for the engine to turn into a subcall


def _usable(ne: NodeEdge, holes_ok: bool) -> bool:
    return not ne.tomb and (ne.replayable or (holes_ok and ne.fillable))


def _choose(view: TreeView, cfg: Config, nid: str, kids: list[NodeEdge], feats: dict | None, text: str,
            words: set, pending: bool, holes_ok: bool = False) -> tuple[NodeEdge | None, str, float, str]:
    """Returns (edge, tier, confidence, why-not)."""
    best = max(kids, key=lambda ne: (ne.pass_n, ne.conf, ne.edge))
    if best.purity >= cfg.purity:
        if best.tomb:
            return None, "", 0.0, "tombstoned"
        if not _usable(best, holes_ok):
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
            if ne is not None and _usable(ne, holes_ok):
                if feats is not None and ne.guard and not guard_holds(ne.guard, feats):
                    return None, "", 0.0, "guard"
                return ne, "T1", leaf["purity"] * ne.success, ""
            return None, "", 0.0, "branch_not_replayable"
    return None, "", 0.0, "ambiguous"


def materialize(view: TreeView, cfg: Config, opt: Option, filled: dict[str, Val] | None = None) -> tuple[dict | None, str]:
    """Render an option's call and re-check safety. `filled`: model-supplied values for its holes."""
    values = dict(opt.values)
    if filled is not None:
        for slot in opt.holes:
            v = filled.get(slot)
            if v is None or not v.cooked.strip() or len(v.cooked) > MAX_FILL:
                return None, f"bad_value:{slot}"
            values[slot] = v
    try:
        args = render(opt.edge.template, opt.ne.ref, values)
    except (KeyError, ValueError) as e:
        return None, f"render:{e}"
    tool = opt.edge.tool
    if is_readonly(tool, opt.ne.ref, cfg) and not is_readonly(tool, args, cfg):
        return None, "unsafe_args"
    if is_commit_point(tool, args, cfg) and not opt.ne.commit_point:
        return None, "commit_point"
    if filled is not None and view.match(ToolCall("", tool, args))[0] != opt.edge.id:
        return None, "shape_changed"
    return args, ""


def option(view: TreeView, cfg: Config, ne: NodeEdge, S: Sources, holes_ok: bool,
           borrow: list[NodeEdge] | None = None) -> tuple[Option | None, str]:
    """Bind an edge's variables. Unbound slots become holes when holes_ok, else a reason to go to T4.
    `borrow`: the same edge at more specific contexts, whose rules may bind what this one can't."""
    values: dict[str, Val] = {}
    holes = []
    for slot in list(ne.bindings) + [h for h in ne.holes if h not in ne.bindings]:
        rule = ne.bindings.get(slot)
        v = eval_rule(rule, S) if rule is not None else None
        for alt in borrow or []:
            if v is not None:
                break
            r = alt.bindings.get(slot)
            v = eval_rule(r, S) if r is not None else None
        if v is not None:
            values[slot] = v
        else:
            holes.append(slot)
    opt = Option(ne, view.edges[ne.edge], values, holes)
    if holes:
        return (opt, "") if holes_ok else (None, f"unbound:{holes[0]}")
    opt.args, why = materialize(view, cfg, opt)
    return (opt, "") if opt.args is not None else (None, why)


def _alternatives(view: TreeView, cfg: Config, kids: list[NodeEdge], feats: dict | None, S: Sources,
                  skip: str = "") -> list[Option]:
    """Known children the model may pick at a T2 call: chosen before, usable, guard holds."""
    out = []
    for k in sorted(kids, key=lambda k: (-k.pass_n, -k.conf, k.edge)):
        if k.edge == skip or k.pass_n <= 0 or not _usable(k, cfg.t3):
            continue
        if feats is not None and k.guard and not guard_holds(k.guard, feats):
            continue
        opt, _ = option(view, cfg, k, S, cfg.t3)
        if opt is not None:
            opt.conf = k.conf
            out.append(opt)
    return out


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
        if st.replayed_via in ("t2", "ck"):
            budget = 1.0  # the model looked at this step: a T2 checkpoint restarts the budget
        c = st.replayed_conf
        budget *= ne.conf if c is None else c

    words = task_words(ep.task)
    calls = [s.call for s in steps]
    obs = [s.obs for s in steps]
    count = trail
    while True:
        i = len(calls)
        first = not plan.calls
        pending = not first  # the previous call in this batch has no observation yet
        holes_ok = cfg.t3 and first
        ctxs = contexts(eids, i, cfg)
        feats = None if pending else (obs_features(obs[-1]) if i > 0 else {})
        text = "" if pending or i == 0 or obs[-1] is None else obs[-1].text
        S = Sources(ep.task, calls, obs, slots)
        choice, why, nid, used = None, "off_tree" if not ctxs else "no_evidence", None, ""
        kids: list[NodeEdge] = []
        seen_here: list[set[str]] = []  # model choices at more specific contexts we backed off from
        starved: list[str] = []          # ...and those contexts
        for kind, ctx in ctxs:
            cand = node_id(view.family, kind, ctx)
            here = view.children.get(cand)
            if not here:
                continue
            if plan.node is None and first:
                plan.node = cand
            if view.node_pass.get(cand, 0) < cfg.promote_runs:
                # too little evidence here; a less specific context may know more,
                # as long as what it proposes doesn't contradict what little we saw here
                chosen = {k.edge for k in here if k.pass_n > 0}
                if chosen:
                    seen_here.append(chosen)
                starved.append(cand)
                continue
            nid, used, kids = cand, f"{kind}{len(ctx)}", here
            ne, tier, conf, why = _choose(view, cfg, cand, kids, feats, text, words, pending, holes_ok)
            if ne is not None:
                if any(ne.edge not in chosen for chosen in seen_here):
                    why = "backoff_disagrees"
                else:
                    choice = (ne, tier, conf)
            break
        if choice is None:
            if (why == "ambiguous" and first and cfg.t2 and nid and count < cfg.hard_cap
                    and sum(1 for k in kids if k.pass_n > 0 and not k.tomb) >= 2):
                opts = _alternatives(view, cfg, kids, feats, S)[:MAX_OPTIONS]
                if sum(o.ne.purity for o in opts) >= MIN_COVER:
                    plan.sub = Subcall("choose", nid, used, opts, "ambiguous")
            plan.reason = plan.reason or (why + (f"@{used}" if used else ""))
            break
        ne, tier, conf = choice
        # Value back-off (with T3 on): once T3 serves a hole at a general context, the more specific
        # contexts stop collecting model-chosen evidence, but their value rules (learned from every
        # passing instance) still apply. Without T3 those steps went to the model and fed them.
        borrow = [c for c in (view.child(n, ne.edge) for n in starved) if c is not None and c.live and not c.tomb] \
            if holes_ok else []
        opt, why = option(view, cfg, ne, S, holes_ok, borrow)
        if opt is None:
            plan.reason = plan.reason or why
            break
        opt.conf, opt.tier = conf, tier
        if count >= cfg.hard_cap:
            plan.reason = plan.reason or "hard_cap"
            break
        if budget * conf < cfg.theta:
            if first and cfg.t2:
                alts = _alternatives(view, cfg, kids, feats, S, skip=ne.edge)
                plan.sub = Subcall("choose", nid, used, [opt] + alts[: MAX_OPTIONS - 1], "budget")
            plan.reason = plan.reason or "budget"
            break
        if opt.holes:
            plan.sub = Subcall("fill", nid, used, [opt], "holes")
            plan.reason = plan.reason or "holes:" + ",".join(opt.holes)
            break
        edge, args = opt.edge, opt.args
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
        plan.sub = None
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

