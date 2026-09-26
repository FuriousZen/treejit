"""python -m treejit_bench --tasks 200 --out bench_out

Runs a task suite with treejit off and on and writes:
  results.csv          one row per (mode, task)
  summary.json         windowed metrics per mode
  learning_curve.html  model calls / tokens / replay share / small calls vs task index

Suites
  sim       (default) the synthetic coding + retail suite driven by a simulated model
  taubench  real tau-bench tasks (TAUBENCH_PATH=<clone>), oracle-with-noise agent or a real model:
            python -m treejit_bench --suite taubench --tau-env retail --tau-split test --modes baseline,treejit,treejit+ok
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from . import runner
from .chart import report_html
from .runner import run_suite, run_suite_proxy, to_rows


def window_stats(rows: list[dict], a: int, b: int) -> dict:
    w = rows[a:b]
    if not w:
        return {}
    tool = sum(r["tool_calls"] for r in w) or 1
    out = {
        "range": f"{a + 1}-{a + len(w)}", "n": len(w),
        "calls_per_task": sum(r["model_calls"] for r in w) / len(w),
        "small_calls_per_task": sum(r["small_calls"] for r in w) / len(w),
        "tokens_per_task": sum(r["tokens"] for r in w) / len(w),
        "served_pct": 100 * sum(r["replayed_calls"] for r in w) / tool,
        "success_pct": 100 * sum(1 for r in w if r["success"]) / len(w),
        "wall_s_per_task": sum(r["wall_ms"] for r in w) / len(w) / 1000,
        "side_exits": sum(r["side_exits"] for r in w),
        "compacted_chars_per_task": sum(r["compacted_chars"] for r in w) / len(w),
    }
    if "cost_tokens" in w[0]:
        full_calls = sum(r["model_calls"] for r in w)
        small_calls = sum(r["small_calls"] for r in w)
        small_cost = sum(r["small_cost"] for r in w)
        full_cost = sum(r["cost_tokens"] for r in w) - small_cost
        out.update({
            "cost_per_task": sum(r["cost_tokens"] for r in w) / len(w),
            "cache_read_per_task": sum(r["cache_read"] for r in w) / len(w),
            "cache_write_per_task": sum(r["cache_write"] for r in w) / len(w),
            "cost_per_full_call": full_cost / full_calls if full_calls else 0.0,
            "cost_per_small_call": small_cost / small_calls if small_calls else 0.0,
        })
        out["small_full_cost_ratio"] = (out["cost_per_small_call"] / out["cost_per_full_call"]
                                        if small_calls and out["cost_per_full_call"] else None)
    if "reward" in w[0]:
        out["reward"] = sum(r["reward"] for r in w) / len(w)
    return out


def _parse_weights(s: str) -> dict:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--cost-weights takes write,read,output (e.g. 1.25,0.1,5)")
    return {"write": parts[0], "read": parts[1], "output": parts[2]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="treejit_bench")
    p.add_argument("--suite", choices=["sim", "taubench"], default="sim")
    p.add_argument("--tasks", type=int, default=None, help="number of tasks (sim default 200; taubench default: the whole split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--family", choices=["mixed", "coding", "retail"], default="mixed", help="sim suite only")
    p.add_argument("--noise", type=float, default=None,
                   help="sim: probability of a known-bad shortcut (default 0.06); taubench oracle: per-write slip probability (default 0.05)")
    p.add_argument("--modes", default="baseline,treejit,treejit+ok",
                   help="comma-separated: baseline, treejit, treejit+ok; append +compact for prefix compaction")
    p.add_argument("--via-proxy", action="store_true", help="sim: run treejit modes through the ASGI proxy (SSE streaming)")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--out", default="bench_out")
    g = p.add_argument_group("cost model (E2)")
    g.add_argument("--payload", choices=["none", "tau", "claude-code"], default="none",
                   help="virtual harness system+tools tokens added to every full call (none | tau ~5k | claude-code ~24k)")
    g.add_argument("--cache", action="store_true", help="model Anthropic prompt caching (cache_read / cache_write columns)")
    g.add_argument("--cost-weights", type=_parse_weights, default=None, metavar="W,R,O",
                   help="cost_tokens weights for cache write, cache read, output (default 1.25,0.1,5)")
    t = p.add_argument_group("tau-bench")
    t.add_argument("--tau-env", choices=["retail", "airline"], default="retail")
    t.add_argument("--tau-split", choices=["test", "train", "dev"], default="test")
    t.add_argument("--tau-start", type=int, default=0, help="first task index")
    t.add_argument("--tau-path", default=None, help="tau-bench clone (default $TAUBENCH_PATH)")
    t.add_argument("--tau-user", choices=["single", "confirm"], default="single", help="scripted user: one turn, or confirm agent questions")
    t.add_argument("--agent", choices=["oracle", "claude"], default="oracle", help="claude needs ANTHROPIC_API_KEY and the anthropic SDK")
    t.add_argument("--claude-model", default="claude-opus-5")
    p.add_argument("--rebuild-every", type=int, default=1, metavar="K",
                   help="rebuild the tree after every K-th outcome instead of every one (faster on tau-bench; "
                        "changes learning dynamics: the last K-1 runs are not yet visible)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cost_weights:
        runner.COST_WEIGHTS.update(a.cost_weights)
    tau = a.suite == "taubench"
    noise = a.noise if a.noise is not None else (0.05 if tau else 0.06)
    n_tasks = a.tasks if a.tasks is not None or tau else 200
    cost = a.cache or a.payload != "none" or tau

    series: dict[str, list[dict]] = {}
    for mode in a.modes.split(","):
        t0 = time.time()
        db = None if mode == "baseline" else os.path.join(a.out, f"{mode}.db")
        key = mode
        if tau:
            from .taubench import run_taubench

            res = run_taubench(n_tasks, a.tau_env, a.tau_split, mode, noise, a.seed, db=db, start=a.tau_start, agent=a.agent,
                               payload=a.payload, cache=a.cache, rebuild_every=a.rebuild_every, path=a.tau_path,
                               user=a.tau_user, claude_model=a.claude_model)
        elif a.via_proxy and mode != "baseline":
            res = run_suite_proxy(n_tasks, a.seed, a.family, mode, noise, db=db, payload=a.payload, cache=a.cache)
            key = mode + "@proxy"
        else:
            res = run_suite(n_tasks, a.seed, a.family, mode, noise, db=db, payload=a.payload, cache=a.cache,
                            rebuild_every=a.rebuild_every)
        series[key] = to_rows(res, cost=cost)
        print(f"{key:<26} {len(res)} tasks in {time.time() - t0:.1f}s")

    n = max(len(rows) for rows in series.values())
    summary = {m: {"first": window_stats(rows, 0, min(10, n)),
                   "mid": window_stats(rows, max(0, min(40, n - 10)), min(50, n)),
                   "last": window_stats(rows, max(0, n - 50), n)} for m, rows in series.items()}
    if tau:
        for m, rows in series.items():
            summary[m]["all"] = window_stats(rows, 0, n)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        rows = [r for rs in series.values() for r in rs]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    meta = {"suite": a.suite, "tasks": n, "seed": a.seed, "noise": noise}
    if tau:
        meta.update({"env": a.tau_env, "split": a.tau_split, "agent": a.agent})
    else:
        meta.update({"family": a.family, "via_proxy": a.via_proxy})
    if cost:
        meta.update({"payload": a.payload, "cache": a.cache})
    if a.rebuild_every > 1:
        meta["rebuild_every"] = a.rebuild_every
    with open(os.path.join(a.out, "learning_curve.html"), "w") as f:
        f.write(report_html(series, summary, meta, a.window))

    extra = ""
    if cost:
        extra += f" {'cost/task':>10} {'cache r/w per task':>19} {'small/full cost':>15}"
    if tau:
        extra += f" {'reward':>7}"
    print(f"\n{'mode':<26} {'window':<9} {'calls/task':>10} {'small/task':>10} {'tokens/task':>12} {'replayed':>9} {'success':>8} "
          f"{'side exits':>10} {'compacted/task':>15}" + extra)
    for m, s in summary.items():
        for k in ("first", "mid", "last", "all"):
            if k not in s or (k == "all" and s[k]["range"] == s["last"]["range"]):
                continue
            x = s[k]
            line = (f"{m:<26} {x['range']:<9} {x['calls_per_task']:>10.2f} {x['small_calls_per_task']:>10.2f} "
                    f"{x['tokens_per_task']:>12,.0f} {x['served_pct']:>8.0f}% "
                    f"{x['success_pct']:>7.0f}% {x['side_exits']:>10} {x['compacted_chars_per_task']:>15,.0f}")
            if cost:
                ratio = x.get("small_full_cost_ratio")
                rw = f"{x['cache_read_per_task']:,.0f}/{x['cache_write_per_task']:,.0f}"
                line += f" {x['cost_per_task']:>10,.0f} {rw:>19} {('%.3f' % ratio) if ratio is not None else '–':>15}"
            if tau:
                line += f" {x['reward']:>7.3f}"
            print(line)
    print(f"\nwrote {a.out}/results.csv, summary.json, learning_curve.html")


if __name__ == "__main__":
    main()
