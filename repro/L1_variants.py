"""Prototype fixes for L1 by monkeypatching (repo untouched), measured on the bench and the scripted repro.

variants:
  base      current code
  dedup     rule support counts distinct task word-sets (identical/duplicate tasks count once)
  novel     T1 on a task-word rule replays only if the input resembles a supporting example
            (max Jaccard of task word sets >= THR); otherwise 'unproven_rule' -> T2. The T2 pick then
            becomes an example, so each new input class costs one small call, once.
  sup10     task_rule_support=10 (a stricter count; stands in for a Wilson/sequential bound)
usage: L1_variants.py <variant> <seed|repro> [tasks]
"""
import json, sys
import treejit.builder as B
import treejit.replay as R
from treejit.features import eval_pred, eval_decision_list

THR = 0.5
variant, what = sys.argv[1], sys.argv[2]
overrides = {}

if variant in ("dedup", "novel", "novel+dedup"):
    _orig_ldl = B.learn_decision_list

    def ldl(examples, purity, **kw):
        dl = _orig_ldl(examples, purity, **kw)
        if dl:
            for r in dl["rules"]:
                if r["pred"][0] != "task":
                    continue
                sup = [e for e in examples if e[0] == r["edge"] and eval_pred(r["pred"], e[1], e[2], e[3])]
                sets = {frozenset(e[3]) for e in sup}
                if "dedup" in variant:
                    r["support"] = len(sets)
                if "novel" in variant:
                    r["ex"] = [sorted(s) for s in list(sets)[-20:]]
        return dl
    B.learn_decision_list = ldl

if "novel" in variant:
    _orig_choose = R._choose

    def choose(view, cfg, nid, kids, feats, text, words, pending, holes_ok=False):
        out = _orig_choose(view, cfg, nid, kids, feats, text, words, pending, holes_ok)
        if out[1] == "T1":
            leaf = eval_decision_list(view.stumps.get(nid) or {}, feats or {}, text, words)
            if leaf and leaf["pred"][0] == "task" and leaf.get("ex"):
                sim = max(len(words & set(e)) / len(words | set(e)) for e in leaf["ex"])
                if sim < THR:
                    return None, "", 0.0, "unproven_rule"
        return out
    R._choose = choose

if variant == "sup10":
    overrides["task_rule_support"] = 10

if what == "repro":
    import runpy, treejit
    # run the scripted repro with patched modules; TreeJIT picks up overrides via env
    import os
    for k, v in overrides.items():
        os.environ["TREEJIT_" + k.upper()] = str(v)
    runpy.run_path("L1_repro.py", run_name="__main__")
    sys.exit()

from treejit_bench.runner import run_suite
seed = int(what)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
res = run_suite(n, seed=seed, mode="treejit+ok", **overrides)
tail = [r for r in res if r.index >= 150]
out = {
    "variant": variant, "seed": seed,
    "success": sum(r.success for r in res),
    "fails": [(r.index, r.kind, r.reason) for r in res if not r.success],
    "small_per_task_all": round(sum(r.small_calls for r in res) / len(res), 3),
    "full_per_task_all": round(sum(r.model_calls for r in res) / len(res), 3),
    "tokens_per_task_all": round(sum(r.tokens for r in res) / len(res)),
    "small_per_task_151_200": round(sum(r.small_calls for r in tail) / len(tail), 3),
    "tokens_per_task_151_200": round(sum(r.tokens for r in tail) / len(tail)),
}
print(json.dumps(out))
