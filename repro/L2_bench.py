"""Bench a L2_patches variant: usage L2_bench.py <variant> <seed>"""
import json, sys
from L2_patches import apply
apply(sys.argv[1])
from treejit_bench.runner import run_suite
seed = int(sys.argv[2])
res = run_suite(200, seed=seed, mode="treejit+ok")
tail = [r for r in res if r.index >= 150]
print(json.dumps({"variant": sys.argv[1], "seed": seed, "success": sum(r.success for r in res),
                  "fails": [(r.index, r.kind, r.reason) for r in res if not r.success],
                  "small_per_task_all": round(sum(r.small_calls for r in res) / len(res), 3),
                  "tokens_per_task_all": round(sum(r.tokens for r in res) / len(res)),
                  "small_per_task_151_200": round(sum(r.small_calls for r in tail) / len(tail), 3),
                  "tokens_per_task_151_200": round(sum(r.tokens for r in tail) / len(tail))}))
