"""E2: measure prompt composition of simulated full (T4) vs small (T2/T3) calls, then re-price
the same trajectories under realistic harness payloads and prompt caching.

usage: python E2_measure.py [--tasks 200] [--seed 0] [--mode treejit+ok]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys

sys.path[:0] = ["/home/user/treejit/src", "/home/user/treejit/bench/src"]
from treejit_bench import runner, sim  # noqa: E402

LOG: list[dict] = []
_orig = sim.SimModel.__call__


def _spy(self, body):
    forced = (body.get("tool_choice") or {}).get("name", "")
    sysc = len(body.get("system", "") if isinstance(body.get("system"), str) else json.dumps(body.get("system")))
    toolc = len(json.dumps(body.get("tools", [])))
    msgc = len(json.dumps(body["messages"]))
    fam = "coding" if any(t["name"] == "Bash" for t in body.get("tools", [])) or re.search(r'"tool": "(Bash|Read|Edit)"', str(body["messages"][0]["content"])) else "retail"
    resp = _orig(self, body)
    LOG.append({"small": forced.startswith("treejit_"), "fam": fam, "sys": sysc, "tools": toolc, "msgs": msgc,
                "n_msgs": len(body["messages"]), "parent_msgs": _LAST["msgs"], "in": resp["usage"]["input_tokens"], "out": resp["usage"]["output_tokens"],
                "task": body["messages"][0]["content"][:40] if isinstance(body["messages"][0]["content"], str) else ""})
    return resp


sim.SimModel.__call__ = _spy
from treejit.engine import TreeJIT  # noqa: E402
_LAST = {"msgs": 0}
_oh = TreeJIT.handle


def _h(self, dialect, body, headers=None):
    _LAST["msgs"] = len(json.dumps(body["messages"]))
    return _oh(self, dialect, body, headers)


TreeJIT.handle = _h


def price(rows, sys_tok, tool_tok, cache: bool, write_mult=1.25, read_mult=0.1):
    """Re-price each logged call. Full calls carry the harness's real system+tools (sys_tok+tool_tok tokens)
    instead of the sim's; small calls keep their own short prompt (treejit builds it, no harness payload).
    With cache: the static prefix (tools+system) is a cache read after the first call of a family,
    the conversation prefix seen in the previous full call of the same task is a cache read, and only the
    new suffix is written. Returns effective input-token-equivalents (uncached input = 1.0)."""
    total = 0.0
    prev_msgs: dict[str, int] = {}
    warm = set()
    for r in rows:
        out = r["out"] * 5  # output tokens cost 5x input (Sonnet/Opus list-price ratio)
        if r["small"]:
            total += r["in"] + out  # short prompt, different prefix: never cached
            continue
        msgs_tok = r["msgs"] // 4
        static = sys_tok + tool_tok
        if not cache:
            total += static + msgs_tok + out
            continue
        static_cost = static * (read_mult if r["fam"] in warm else write_mult)
        warm.add(r["fam"])
        seen = prev_msgs.get(r["task"], 0)
        seen = min(seen, msgs_tok)
        total += static_cost + seen * read_mult + (msgs_tok - seen) * write_mult + out
        prev_msgs[r["task"]] = msgs_tok
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    results = {}
    for mode in ("baseline", "treejit+ok"):
        LOG.clear()
        runner.run_suite(a.tasks, a.seed, "mixed", mode)
        results[mode] = list(LOG)
    rows = results["treejit+ok"]
    full = [r for r in rows if not r["small"]]
    small = [r for r in rows if r["small"]]
    for name, rs in (("full(T4)", full), ("small(T2/T3)", small), ("baseline", results["baseline"])):
        for fam in ("coding", "retail"):
            x = [r for r in rs if r["fam"] == fam]
            if not x:
                continue
            print(f"{name:<13} {fam:<7} n={len(x):<5} sys={st.mean(r['sys'] for r in x):6.0f}ch tools={st.mean(r['tools'] for r in x):6.0f}ch "
                  f"msgs={st.mean(r['msgs'] for r in x):6.0f}ch (max {max(r['msgs'] for r in x)}) in_tok={st.mean(r['in'] for r in x):6.0f} "
                  f"out_tok={st.mean(r['out'] for r in x):4.0f}")
    print(f"\nsmall/full input ratio (sim): {st.mean(r['in'] + r['out'] for r in small) / st.mean(r['in'] + r['out'] for r in full):.2f}")
    print("\nRe-priced totals over all tasks (input-token equivalents, output x5):")
    print(f"{'scenario':<44} {'baseline':>12} {'treejit+ok':>12} {'cut':>6} {'small share':>11} {'small/full':>10}")
    for label, s, t, c in (("sim as-is (sys+tools from sim)", None, None, False),
                           ("tau-bench-like (1.5k sys + 3.5k tools)", 1500, 3500, False),
                           ("tau-bench-like, cached", 1500, 3500, True),
                           ("Claude Code (12k sys + 12k tools)", 12000, 12000, False),
                           ("Claude Code, cached", 12000, 12000, True)):
        tot = {}
        for mode, rs in results.items():
            if s is None:
                tot[mode] = sum(r["in"] + r["out"] * 5 for r in rs)
            else:
                tot[mode] = price(rs, s, t, c)
        sm = sum(r["in"] + r["out"] * 5 for r in small)
        per_full = (tot["treejit+ok"] - sm) / max(1, len(full))
        per_small = sm / max(1, len(small))
        print(f"{label:<44} {tot['baseline'] / a.tasks:12,.0f} {tot['treejit+ok'] / a.tasks:12,.0f} "
              f"{100 * (1 - tot['treejit+ok'] / tot['baseline']):5.1f}% {100 * sm / tot['treejit+ok']:10.1f}% {per_small / per_full:10.3f}")
    # What would these small calls have cost as full calls at the same point (the T4 they replaced)?
    for label, s_, t_ in (("sim", None, None), ("tau-bench-like", 1500, 3500), ("Claude Code", 12000, 12000)):
        ratios = []
        for r in small:
            stat = (r["sys"] + r["tools"]) // 4 if s_ is None else 0
            stat = stat if s_ is None else s_ + t_
            if s_ is None:  # sim's own full-call static for that family
                f = next(x for x in full if x["fam"] == r["fam"])
                stat = (f["sys"] + f["tools"]) // 4
            ratios.append((r["in"] + r["out"]) / (stat + r["parent_msgs"] // 4 + 60))
        print(f"small call / the T4 it replaced, {label:<15}: mean {st.mean(ratios):.3f}  median {st.median(ratios):.3f}")
    print(f"\nfull calls={len(full)} small calls={len(small)} over {a.tasks} tasks")
