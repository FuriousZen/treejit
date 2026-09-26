"""Build a family's tree from its cold trace log.

The build is a deterministic function of (traces, outcomes, operator state), so it
can simply be re-run whenever an outcome arrives:

1. group calls by shape and anti-unify each group into an edge template
2. place every step under its contexts (root path + last-k n-grams)
3. assign blame for failed runs to the first edge after the divergence point
4. per (node, edge): promotion, bindings/holes, guard, postcondition,
   confidence, tombstone, tier, savings and priority score
5. per node: a decision list over observation/task predicates (T1)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .bindings import Sources, find_rule
from .config import Config
from .features import excess_negatives, guard_holds, guard_of, learn_decision_list, obs_features, postcondition, task_words
from .model import Observation, Step, ToolCall, user_ended
from .policy import commit_reason, is_readonly
from .store import Store, dumps
from .templates import anti_unify, call_slots, edge_id, label, shape_of, var_slots
from .tree import END, contexts, node_id
from .util import decay, now

MAX_INSTANCES = 40  # most recent passing instances used for bindings/guards
NE_COLS = ("node", "edge", "family", "n", "pass_runs", "fail_runs", "pass_n", "blamed", "tomb", "live", "replayable",
           "tier", "purity", "success", "conf", "bindings", "holes", "guard", "post", "ref", "reasons", "commit_point",
           "savings", "latency_ms", "score", "fillable", "blocked", "commit_reason", "approved")


@dataclass
class RunData:
    id: str
    outcome: str | None
    reason: str | None
    task: str
    task_hash: str
    created: float
    updated: float
    outcome_at: float
    ended_after: int | None = None    # the model gave its final answer after this many steps
    inherited: int = 0                # leading steps copied from a finished run this one forks: context only
    steps: list[Step] = field(default_factory=list)
    replayed: list[bool] = field(default_factory=list)
    eids: list[str] = field(default_factory=list)
    slots: list[dict | None] = field(default_factory=list)
    feats: list[dict] = field(default_factory=list)
    _words: set | None = None

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"

    @property
    def failed(self) -> bool:
        return self.outcome == "fail"

    def words(self) -> set:
        if self._words is None:
            self._words = task_words(self.task)
        return self._words

    def sources(self, i: int) -> Sources:
        return Sources(self.task, [s.call for s in self.steps[:i]], [s.obs for s in self.steps[:i]], self.slots[:i])


def load_runs(store: Store, family: str, limit: int = 1_000_000) -> list[RunData]:
    runs: dict[str, RunData] = {}
    rows = store.q("SELECT * FROM runs WHERE family=? ORDER BY created DESC LIMIT ?", (family, limit))
    for r in reversed(rows):
        runs[r["id"]] = RunData(r["id"], r["outcome"], r["reason"], r["task"] or "", r["task_hash"] or "",
                                r["created"], r["updated"], r["outcome_at"] or r["updated"], r["ended_after"],
                                r["inherited"] or 0)
    if not runs:
        return []
    for s in store.q("SELECT s.* FROM steps s JOIN runs r ON r.id=s.run_id WHERE r.family=? ORDER BY s.run_id, s.idx", (family,)):
        rd = runs.get(s["run_id"])
        if rd is None or s["idx"] != len(rd.steps):
            continue  # gap in the log; ignore the rest of this run
        obs = Observation(s["obs"], bool(s["is_error"])) if s["obs"] is not None else None
        rd.steps.append(Step(ToolCall(s["call_id"], s["tool"], json.loads(s["args"])), obs))
        rd.replayed.append(bool(s["replayed"]))
    return [r for r in runs.values() if r.steps]


def _usage_by_call(store: Store, family: str) -> dict[str, tuple[float, float]]:
    """Cost of the full (T4) model call that produced each call id. Savings are measured
    against T4 only; T2/T3 subcalls are logged with their own usage but never count as savings."""
    out: dict[str, tuple[float, float]] = {}
    for r in store.q("SELECT call_ids, input_tokens, output_tokens, cache_read, cache_write, latency_ms, n_calls "
                     "FROM requests WHERE family=? AND tier='T4' AND call_ids IS NOT NULL", (family,)):
        ids = json.loads(r["call_ids"] or "[]")
        if not ids:
            continue
        tokens = (r["input_tokens"] or 0) + (r["output_tokens"] or 0) + (r["cache_read"] or 0) + (r["cache_write"] or 0)
        for cid in ids:
            out[cid] = (tokens / len(ids), (r["latency_ms"] or 0.0) / len(ids))
    return out


def build_family(store: Store, cfg: Config, family: str) -> dict[str, Any]:
    t_now = now()
    runs = load_runs(store, family, cfg.max_runs)
    pins = store.pins()
    approvals = store.approvals()
    not_commit = store.not_commit()
    evicted = store.evictions()
    usage = _usage_by_call(store, family)

    # 1. edge templates by shape
    by_shape: dict[str, list[ToolCall]] = defaultdict(list)
    for rd in runs:
        for st in rd.steps:
            by_shape[shape_of(st.call)].append(st.call)
    templates = {s: anti_unify(calls) for s, calls in by_shape.items()}
    eid_of = {s: edge_id(family, s) for s in by_shape}
    edge_tpl = {eid_of[s]: templates[s] for s in by_shape}

    for rd in runs:
        for st in rd.steps:
            s = shape_of(st.call)
            rd.eids.append(eid_of[s])
            rd.slots.append(call_slots(templates[s], st.call))
            rd.feats.append(obs_features(st.obs))

    # 2. contexts
    inst: dict[tuple[str, str], list[tuple[RunData, int]]] = defaultdict(list)
    node_meta: dict[str, tuple[str, tuple]] = {}
    node_runs: dict[str, set] = defaultdict(set)
    node_seen: dict[str, float] = defaultdict(float)
    step_nodes: dict[tuple[str, int], list[str]] = {}
    # User steps (M1) are context, never choices: a later user turn is not something the model (or
    # replay) produces. When it answers a finished agent turn, the model chose to stop and talk there,
    # which is END evidence at that context (below). Inherited steps (a fork's copied prefix) were
    # counted in the run they came from.
    user_stops: list[tuple[RunData, int]] = []
    for rd in runs:
        for i in range(len(rd.steps)):
            if rd.steps[i].is_user or i < rd.inherited:
                step_nodes[(rd.id, i)] = []
                if rd.steps[i].is_user and user_ended(rd.steps[i].call.name) and (i > rd.inherited or not rd.inherited):
                    user_stops.append((rd, i))
                continue
            nids = []
            for kind, ctx in contexts(rd.eids, i, cfg):
                nid = node_id(family, kind, ctx)
                if evicted.get(nid, 0) >= rd.updated:
                    continue
                node_meta[nid] = (kind, ctx)
                inst[(nid, rd.eids[i])].append((rd, i))
                node_runs[nid].add(rd.id)
                node_seen[nid] = max(node_seen[nid], rd.updated)
                nids.append(nid)
            step_nodes[(rd.id, i)] = nids

    pass_pairs = {pair for pair, lst in inst.items() if any(rd.passed for rd, _ in lst)}
    # What the model would choose here is learned only from steps the model chose:
    # counting replayed steps would let replay reinforce its own guesses.
    node_pass_n: dict[str, int] = defaultdict(int)
    for (nid, _), lst in inst.items():
        node_pass_n[nid] += sum(1 for rd, i in lst if rd.passed and not rd.replayed[i])

    # The model's other choice: ending the episode. A passing run whose final answer came right
    # after its last step puts an END choice at the contexts after that step. It counts toward
    # the node's evidence (so purity and decision lists can say "the model stops here") but is
    # never replayed: the final answer always comes from the model.
    node_end: dict[str, int] = defaultdict(int)
    ends: dict[str, list[tuple[RunData, int]]] = defaultdict(list)
    stops = [(rd, len(rd.steps)) for rd in runs
             if rd.passed and rd.ended_after == len(rd.steps) and len(rd.steps) > rd.inherited]
    stops += [(rd, i) for rd, i in user_stops if rd.passed]
    for rd, at in stops:
        for kind, ctx in contexts(rd.eids, at, cfg):
            nid = node_id(family, kind, ctx)
            if evicted.get(nid, 0) >= rd.updated:
                continue
            node_meta[nid] = (kind, ctx)
            node_runs[nid].add(rd.id)
            node_seen[nid] = max(node_seen[nid], rd.updated)
            node_end[nid] += 1
            node_pass_n[nid] += 1
            ends[nid].append((rd, at))

    # Side exits teach. A replayed step whose result broke the edge's postcondition is
    # a miss against the context that chose it; if the model then took over and the run
    # passed, its choice is what should have happened at that context (a correction).
    post_cache: dict[tuple[str, str], dict] = {}

    def post_of(pair: tuple[str, str]) -> dict:
        if pair not in post_cache:
            fs = [rd.feats[i] for rd, i in inst.get(pair, []) if rd.passed and not rd.replayed[i] and rd.steps[i].obs is not None]
            post_cache[pair] = postcondition(fs) if len(fs) >= 2 else {}
        return post_cache[pair]

    misses: dict[tuple[str, str], int] = defaultdict(int)
    corr: dict[tuple[str, str], list[tuple[RunData, int]]] = defaultdict(list)
    for rd in runs:
        for i in range(len(rd.steps)):
            used = rd.steps[i].replayed_node
            if not used or rd.steps[i].obs is None:
                continue
            post = post_of((used, rd.eids[i]))
            if not post or guard_holds(post, rd.feats[i]):
                continue
            misses[(used, rd.eids[i])] += 1
            if rd.passed and i + 1 < len(rd.steps) and not rd.replayed[i + 1] and not rd.steps[i + 1].is_user:
                for nid in step_nodes.get((rd.id, i), []):
                    corr[(nid, rd.eids[i + 1])].append((rd, i))
                    node_pass_n[nid] += 1

    # Failed replays teach too. A replayed step in a failed run is evidence against the decision
    # that chose it, at the node that chose it: this input, this edge, bad outcome. Edge blame
    # can't say that when the edge is right for other inputs (it is in pass_pairs), and replayed
    # steps never become examples, so a wrong T0/T1 choice used to keep firing (seed 3: a
    # task-word rule sent typo tasks down the delete-module branch in 28 runs). Negatives lower
    # the edge's purity at that node and the decision-list rules predicting it on such inputs.
    # A failed run also fails every *other* replayed step in it, so negatives only count beyond
    # the failure rate tolerated among the same replays in passing runs (`excess_negatives`).
    negs: dict[str, list[tuple[RunData, int, str]]] = defaultdict(list)
    confs: dict[str, list[tuple[RunData, int, str]]] = defaultdict(list)
    neg_n: dict[tuple[str, str], int] = defaultdict(int)
    conf_n: dict[tuple[str, str], int] = defaultdict(int)
    for rd in runs:
        if rd.passed or rd.failed:
            for i, st in enumerate(rd.steps):
                if not rd.replayed[i] or not st.replayed_node:
                    continue
                for nid in step_nodes.get((rd.id, i), []):  # the deciding context and its siblings
                    (negs if rd.failed else confs)[nid].append((rd, i, rd.eids[i]))
                    (neg_n if rd.failed else conf_n)[(nid, rd.eids[i])] += 1

    # 3. credit assignment: first edge after the divergence point
    blame: dict[tuple[str, str], list[tuple[float, str, str | None, str]]] = defaultdict(list)
    for rd in runs:
        if not rd.failed:
            continue
        w = decay(t_now - rd.outcome_at, cfg.half_life_days)
        pairs: list[tuple[str, str]] = []
        for i in range(len(rd.steps)):
            nids = step_nodes.get((rd.id, i), [])
            if not nids:
                continue  # a user step, an inherited step, or an evicted context: not a choice to blame
            if (nids[0], rd.eids[i]) not in pass_pairs:
                pairs = [(nid, rd.eids[i]) for nid in nids if (nid, rd.eids[i]) not in pass_pairs]
                break
        else:
            last = next((j for j in range(len(rd.steps) - 1, -1, -1)
                         if not rd.steps[j].is_user and j >= rd.inherited), -1)
            nids = step_nodes.get((rd.id, last), []) if last >= 0 else []
            pairs = [(nids[0], rd.eids[last])] if nids else []
        for pair in pairs:
            blame[pair].append((w, rd.task_hash, rd.reason, rd.id))

    # 4. per node-edge statistics
    ne_rows = []
    for (nid, eid) in list(inst) + [k for k in corr if k not in inst]:
        lst = inst.get((nid, eid), [])
        tpl = edge_tpl[eid]
        passing = [(rd, i) for rd, i in lst if rd.passed][-MAX_INSTANCES:]
        pass_runs = len({rd.id for rd, _ in passing})
        fail_runs = len({rd.id for rd, _ in lst if rd.failed})
        pass_n = sum(1 for rd, i in lst if rd.passed and not rd.replayed[i]) + len(corr.get((nid, eid), []))
        # share of the model's choices here, discounted by failed replays of this edge here
        neg = excess_negatives(neg_n.get((nid, eid), 0), conf_n.get((nid, eid), 0), cfg.purity)
        purity = pass_n / (node_pass_n[nid] + neg) if node_pass_n[nid] else 0.0
        bl = blame.get((nid, eid), [])
        f = sum(b[0] for b in bl)
        distinct = len({b[1] for b in bl})
        success = (pass_runs + 1) / (pass_runs + f + misses.get((nid, eid), 0) + 2)
        tomb = distinct >= cfg.tomb_k and f >= 1.0 and f / (pass_runs + f) >= cfg.tomb_prob
        pinned = (nid, eid) in pins or (nid, "") in pins
        live = pass_runs >= cfg.promote_runs or pinned

        bindings: dict[str, list | None] = {}
        holes: list[str] = []
        guard: dict = {}
        post: dict = {}
        basis = passing or lst[-MAX_INSTANCES:]
        usable = [(rd, i) for rd, i in basis if rd.slots[i] is not None]
        if usable:
            srcs = [rd.sources(i) for rd, i in usable]
            for slot in var_slots(tpl):
                vals = [rd.slots[i][slot] for rd, i in usable]
                rule = find_rule(vals, srcs, shell_slot="#" in slot)
                bindings[slot] = rule
                if rule is None:
                    holes.append(slot)
        else:
            holes = var_slots(tpl)
        if passing:
            if all(i > 0 for _, i in passing):
                guard = guard_of([rd.feats[i - 1] for rd, i in passing])
            with_obs = [rd.feats[i] for rd, i in passing if rd.steps[i].obs is not None]
            post = postcondition(with_obs) if len(with_obs) == len(passing) else {}
        ref_rd, ref_i = (passing or lst or [(rd, i + 1) for rd, i in corr[(nid, eid)]])[-1]
        ref = ref_rd.steps[ref_i].call.args
        why_commit = commit_reason(tpl["tool"], ref, cfg)
        # `treejit approve EDGE --not-commit` is per edge; `approve '*'` never implies it
        commit = bool(why_commit) and eid not in not_commit
        approved = (eid, nid) in approvals or (eid, "") in approvals or ("*", "") in approvals
        safe = is_readonly(tpl["tool"], ref, cfg) or approved
        fillable = live and not tomb and safe and (not commit or (approved and pass_runs >= cfg.promote_runs + 1))
        replayable = fillable and not holes
        tier = "tomb" if tomb else "hot" if replayable else "live" if live else "warm" if pass_runs else "cold"
        blocked = ("" if replayable else "tomb" if tomb else "not_live" if not live else "holes" if holes
                   else "needs_approval" if not safe or (commit and not approved) else "commit_point_needs_evidence")
        costs = [usage[rd.steps[i].call.id] for rd, i in lst if not rd.replayed[i] and rd.steps[i].call.id in usage]
        savings = sum(c[0] for c in costs) / len(costs) if costs else 0.0
        latency = sum(c[1] for c in costs) / len(costs) if costs else 0.0
        nvars = len(var_slots(tpl))
        templatability = 1.0 - (len(holes) / nvars if nvars else 0.0)
        reasons = []
        for b in sorted(bl, key=lambda b: -b[0]):
            if b[2] and b[2] not in reasons:
                reasons.append(b[2])
        ne_rows.append((
            nid, eid, family, len(lst), pass_runs, fail_runs, pass_n, round(f, 4), int(tomb), int(live), int(replayable),
            tier, round(purity, 4), round(success, 4), round(purity * success, 4), dumps(bindings), dumps(holes),
            dumps(guard), dumps(post), dumps(ref), dumps(reasons[:3]), int(commit), round(savings, 1), round(latency, 1),
            round(pass_runs * max(savings, 1.0) * templatability, 2), int(fillable), blocked, why_commit, int(approved),
        ))

    # 5. decision lists
    node_rows = []
    by_node: dict[str, list[tuple[RunData, int, str]]] = defaultdict(list)
    for (nid, eid), lst in inst.items():
        for rd, i in lst:
            if rd.passed and not rd.replayed[i]:
                by_node[nid].append((rd, i, eid))
    for (nid, eid), lst in corr.items():
        by_node[nid].extend((rd, i, eid) for rd, i in lst)
    for nid, lst in ends.items():
        by_node[nid].extend((rd, at, END) for rd, at in lst)

    def example(rd: RunData, i: int, eid: str) -> tuple[str, dict, str, set]:
        text = rd.steps[i - 1].obs.text if i > 0 and rd.steps[i - 1].obs else ""
        return (eid, rd.feats[i - 1] if i > 0 else {}, text, rd.words())

    for nid, (kind, ctx) in node_meta.items():
        examples = [example(rd, i, eid) for rd, i, eid in by_node.get(nid, [])[-200:]]
        negatives = [example(rd, i, eid) for rd, i, eid in negs.get(nid, [])[-200:]]
        confirmed = [example(rd, i, eid) for rd, i, eid in confs.get(nid, [])[-200:]] if negatives else []
        stump = learn_decision_list(examples, cfg.purity, negatives=negatives, confirmed=confirmed, class_sets=True) \
            if len({e[0] for e in examples}) > 1 else None
        parent = via = None
        if kind == "r" and ctx:
            parent, via = node_id(family, "r", ctx[:-1]), ctx[-1]
        node_rows.append((nid, family, kind, dumps(list(ctx)), len(ctx), parent, via, len(node_runs[nid]),
                          node_pass_n[nid], dumps(stump) if stump else None, node_seen[nid], node_end[nid]))

    edge_rows = [(eid_of[s], family, templates[s]["tool"], s, dumps(templates[s]), len(by_shape[s]), label(templates[s]))
                 for s in by_shape]

    with store.transaction():
        db = store.db
        for table in ("edges", "nodes", "node_edges"):
            db.execute(f"DELETE FROM {table} WHERE family=?", (family,))
        db.executemany("INSERT INTO edges VALUES(?,?,?,?,?,?,?)", edge_rows)
        db.executemany("INSERT INTO nodes(id, family, kind, ctx, depth, parent, via, n_runs, n_pass, stump, last_seen, n_end) "
                       "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", node_rows)
        db.executemany(f"INSERT INTO node_edges({','.join(NE_COLS)}) VALUES({','.join('?' * len(NE_COLS))})", ne_rows)
        db.execute("UPDATE families SET built_at=?, dirty=0 WHERE id=?", (t_now, family))
    return {"family": family, "runs": len(runs), "edges": len(edge_rows), "nodes": len(node_rows),
            "hot": sum(1 for r in ne_rows if r[11] == "hot"), "tomb": sum(1 for r in ne_rows if r[11] == "tomb")}
