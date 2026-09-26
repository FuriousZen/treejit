"""Run the synthetic bench (treejit, treejit+ok) with the current policy or the prototype (argv[1] == 'proto')."""
import sys, os
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "/home/user/treejit/src", "/home/user/treejit/bench/src"]
if sys.argv[1] == "proto":
    import S_proto_policy; S_proto_policy.install()
from treejit_bench.__main__ import main
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S1_bench_" + sys.argv[1])
main(["--tasks", "200", "--modes", "treejit,treejit+ok", "--out", out])
import json
s = json.load(open(os.path.join(out, "summary.json")))
for m, w in s.items():
    print(m, {k: {x: round(v, 3) for x, v in w[k].items() if x in ("calls_per_task", "tokens_per_task", "served_pct", "success_pct")} for k in ("mid", "last")})
