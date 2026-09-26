"""C1 on the simulated bench: bill every task's forwarded full-model requests with the prompt-cache cost
model of tests/cache_model.py (one conversation per task, 10 s between requests, warm cache).

usage (from the repo root):
  PYTHONPATH=$PWD/src:$PWD/bench/src python3 repro/C1_bench_billed.py --seed 0 [--tasks 200] [--bigsys]
(seeds 0-5, both variants: repro/C1_bench_billed_out.txt)

Modes: treejit+ok (compaction off), +compact with compact_mode first_sight (default), and window (pre-C1).
T2/T3 subcalls are separate short bodies with their own prefix and are left out. --bigsys prepends a
62.5k-char static block to every system prompt (Claude Code-sized) to show the effect with a big cached prefix.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from cache_model import bill  # noqa: E402

from treejit_bench import runner, sim  # noqa: E402


def capture(mode, seed, tasks, bigsys, **overrides):
    convs, traj = {}, []
    orig = sim.SimModel.__call__
    pad = "static instructions line\n" * (62500 // 24) if bigsys else ""

    def call(self, body):
        if not (body.get("tool_choice") or {}).get("name", "").startswith("treejit_"):
            b = copy.deepcopy(body)
            if pad:
                b["system"] = pad + (b.get("system") or "")
            c = body["messages"][0]["content"]
            convs.setdefault(c if isinstance(c, str) else c[0]["text"], []).append(b)
        return orig(self, body)

    sim.SimModel.__call__ = call
    try:
        res = runner.run_suite(tasks, seed=seed, mode=mode, **overrides)
    finally:
        sim.SimModel.__call__ = orig
    for r in res:
        traj.append((r.success, r.tool_calls, tuple(r.tiers)))
    total = raw = 0.0
    for bodies in convs.values():
        b = bill([(10.0 * i, x) for i, x in enumerate(bodies)])
        total += b.billed
        raw += b.tokens
    comp = sum(r.compacted_chars for r in res) / len(res)
    return traj, raw / len(res), total / len(res), comp, sum(r.success for r in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--bigsys", action="store_true")
    a = ap.parse_args()
    rows = [("off", capture("treejit+ok", a.seed, a.tasks, a.bigsys)),
            ("first_sight", capture("treejit+ok+compact", a.seed, a.tasks, a.bigsys)),
            ("window", capture("treejit+ok+compact", a.seed, a.tasks, a.bigsys, compact_mode="window"))]
    base = rows[0][1]
    print(f"seed={a.seed} tasks={a.tasks} bigsys={a.bigsys}")
    print(f"   {'mode':12s} {'success':>7s} {'input tok/task':>15s} {'billed/task':>12s} {'vs off':>8s} {'compacted chars/task':>21s} traj==off")
    for name, (traj, raw, billed, comp, succ) in rows:
        print(f"   {name:12s} {succ:7d} {raw:15,.0f} {billed:12,.0f} {100 * (billed - base[2]) / base[2]:+7.1f}% {comp:21,.0f} {traj == base[0]}")


if __name__ == "__main__":
    main()
