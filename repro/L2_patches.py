"""Prototype (monkeypatch) of a per-input-class guard on T1 task-word rules, for L1 and L2.

knn: the builder stores with every task-word rule the task-word sets of
   ex  supporting model-chosen examples + confirmed (passing) replays of the rule
   nx  failed replays of the rule (negatives)
At replay, T1 on the rule requires: max Jaccard(input, ex) >= THR and > max Jaccard(input, nx).
Otherwise 'unproven_rule' -> T2 (the model's pick becomes an example, so the class is learned in one call).
"""
import treejit.builder as B
import treejit.replay as R
from treejit.features import eval_pred, eval_decision_list

THR = 0.5


def jac(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def apply(variant):
    if variant == "base":
        return
    _orig_ldl = B.learn_decision_list

    def ldl(examples, purity, **kw):
        dl = _orig_ldl(examples, purity, **kw)
        if dl:
            for r in dl["rules"]:
                if r["pred"][0] != "task":
                    continue
                m = lambda pool: [sorted(e[3]) for e in (pool or []) if e[0] == r["edge"] and eval_pred(r["pred"], e[1], e[2], e[3])]
                ex = m(examples) + (m(kw.get("confirmed")) if variant == "knn" else [])  # knn0: examples only
                r["ex"] = [list(s) for s in {tuple(x) for x in ex}][:40]
                if variant in ("knn", "knn0"):
                    r["nx"] = [list(s) for s in {tuple(x) for x in m(kw.get("negatives"))}][:40]
        return dl
    B.learn_decision_list = ldl
    # confirmed replays are only passed to the learner when there are negatives; pass them always for knn
    _orig_choose = R._choose

    def choose(view, cfg, nid, kids, feats, text, words, pending, holes_ok=False):
        out = _orig_choose(view, cfg, nid, kids, feats, text, words, pending, holes_ok)
        if out[1] == "T1":
            leaf = eval_decision_list(view.stumps.get(nid) or {}, feats or {}, text, words)
            if leaf and leaf["pred"][0] == "task" and leaf.get("ex"):
                pos = max(jac(words, set(e)) for e in leaf["ex"])
                neg = max((jac(words, set(e)) for e in leaf.get("nx", [])), default=0.0)
                if pos < THR or neg >= pos:
                    return None, "", 0.0, "unproven_rule"
        return out
    R._choose = choose
