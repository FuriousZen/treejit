"""python -m treejit_bench --tasks 200 --out bench_out

Runs the synthetic suite with treejit off and on and writes:
  results.csv          one row per (mode, task)
  summary.json         windowed metrics per mode
  learning_curve.html  model calls / tokens / replay share vs task index
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from .chart import report_html
from .runner import run_suite, run_suite_proxy, to_rows


def window_stats(rows: list[dict], a: int, b: int) -> dict:
    w = rows[a:b]
    if not w:
        return {}
    tool = sum(r["tool_calls"] for r in w) or 1
    return {
        "range": f"{a + 1}-{a + len(w)}", "n": len(w),
        "calls_per_task": sum(r["model_calls"] for r in w) / len(w),
        "small_calls_per_task": sum(r["small_calls"] for r in w) / len(w),
        "tokens_per_task": sum(r["tokens"] for r in w) / len(w),
        "served_pct": 100 * sum(r["replayed_calls"] for r in w) / tool,
        "success_pct": 100 * sum(1 for r in w if r["success"]) / len(w),
        "wall_s_per_task": sum(r["wall_ms"] for r in w) / len(w) / 1000,
        "side_exits": sum(r["side_exits"] for r in w),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="treejit_bench")
    p.add_argument("--tasks", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--family", choices=["mixed", "coding", "retail"], default="mixed")
    p.add_argument("--noise", type=float, default=0.06, help="probability the simulated model takes a known-bad shortcut")
    p.add_argument("--modes", default="baseline,treejit,treejit+ok")
    p.add_argument("--via-proxy", action="store_true", help="run treejit modes through the ASGI proxy (SSE streaming)")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--out", default="bench_out")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    series: dict[str, list[dict]] = {}
    for mode in a.modes.split(","):
        t0 = time.time()
        if a.via_proxy and mode != "baseline":
            res = run_suite_proxy(a.tasks, a.seed, a.family, mode, a.noise, db=os.path.join(a.out, f"{mode}.db"))
            key = mode + "@proxy"
        else:
            res = run_suite(a.tasks, a.seed, a.family, mode, a.noise,
                            db=None if mode == "baseline" else os.path.join(a.out, f"{mode}.db"))
            key = mode
        series[key] = to_rows(res)
        print(f"{key:<18} {a.tasks} tasks in {time.time() - t0:.1f}s")

    n = a.tasks
    summary = {m: {"first": window_stats(rows, 0, min(10, n)),
                   "mid": window_stats(rows, max(0, min(40, n - 10)), min(50, n)),
                   "last": window_stats(rows, max(0, n - 50), n)} for m, rows in series.items()}
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        rows = [r for rs in series.values() for r in rs]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    meta = {"tasks": a.tasks, "seed": a.seed, "family": a.family, "noise": a.noise, "via_proxy": a.via_proxy}
    with open(os.path.join(a.out, "learning_curve.html"), "w") as f:
        f.write(report_html(series, summary, meta, a.window))

    print(f"\n{'mode':<18} {'window':<9} {'calls/task':>10} {'small/task':>10} {'tokens/task':>12} {'replayed':>9} {'success':>8} "
          f"{'side exits':>10}")
    for m, s in summary.items():
        for k in ("first", "mid", "last"):
            x = s[k]
            print(f"{m:<18} {x['range']:<9} {x['calls_per_task']:>10.2f} {x['small_calls_per_task']:>10.2f} "
                  f"{x['tokens_per_task']:>12,.0f} {x['served_pct']:>8.0f}% "
                  f"{x['success_pct']:>7.0f}% {x['side_exits']:>10}")
    print(f"\nwrote {a.out}/results.csv, summary.json, learning_curve.html")


if __name__ == "__main__":
    main()
