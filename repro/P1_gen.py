"""Generate a coding-family trace log with the bench's SimModel and time TreeJIT.outcome (the synchronous rebuild).

usage: P1_gen.py <db> <n_tasks> <rebuild_every>   (rebuild_every=1: stock behaviour, every outcome rebuilds)
Writes per-outcome latency to <db>.outcome_ms.csv (index, runs_in_family, ms).
"""
import sys, time
import treejit.engine as E
from treejit_bench.runner import run_suite

db, n, every = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
orig_outcome = E.TreeJIT.outcome
orig_rebuild = E.TreeJIT.rebuild
log = []
state = {"k": 0}


def outcome(self, run_id, result, reason=None):
    state["k"] += 1
    if every > 1 and state["k"] % every:
        # debounced: record the outcome, mark dirty, skip the rebuild (view() would rebuild lazily, so also stop that)
        ids = self.store.set_outcome(run_id, "pass" if result is True else "fail" if result is False else result, reason)
        self.store.x("UPDATE families SET dirty=0")
        return ids
    t = time.perf_counter()
    out = orig_outcome(self, run_id, result, reason)
    ms = (time.perf_counter() - t) * 1000
    nr = self.store.q1("SELECT COUNT(*) n FROM runs")["n"]
    log.append((state["k"], nr, round(ms, 1)))
    return out


E.TreeJIT.outcome = outcome
t0 = time.time()
res = run_suite(n, seed=7, family="coding", mode="treejit+ok", db=db)
print(f"{n} tasks in {time.time() - t0:.0f}s; success {sum(r.success for r in res)}/{n}; "
      f"replayed share {sum(r.replayed_calls for r in res) / max(1, sum(r.tool_calls for r in res)):.0%}")
with open(db + ".outcome_ms.csv", "w") as f:
    f.write("k,runs,ms\n")
    for row in log:
        f.write(",".join(map(str, row)) + "\n")
for k, nr, ms in log[:: max(1, len(log) // 15)]:
    print(f"outcome #{k:5d} runs={nr:5d} {ms:8.1f} ms")
