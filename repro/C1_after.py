"""C1: frontier compaction vs the Anthropic prompt cache (after the fix).

The cost model and the scenarios live in tests/cache_model.py (see its docstring for the assumptions:
read 0.1x, write 1.25x, one breakpoint after `system`, one at the end of the conversation, 5-minute TTL,
tokens = chars/4). Every compaction mode here is the real `compaction.apply`, selected with
`compact_mode`; nothing is monkeypatched except the clock.

usage (from the repo root):
  PYTHONPATH=$PWD/src:$PWD/bench/src python3 repro/C1_after.py           > repro/C1_after_small.txt
  BIG_SYSTEM=62500 PYTHONPATH=... python3 repro/C1_after.py              > repro/C1_after_bigsys.txt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from cache_model import PATTERNS, SYSTEM, append_only, bill, scenario  # noqa: E402

BIG = int(os.environ.get("BIG_SYSTEM", "0"))  # extra system chars (Claude Code ~ 60k chars)
SYS = SYSTEM + "static instructions line\n" * (BIG // 24)
MODES = (None, "window", "first_sight", "epoch")


def name(m):
    return m or "off"


def table(title, rows):
    print(f"\n=== {title} ===")
    print(f"   {'mode':12s} {'reqs':>4s} {'input tok':>10s} {'cache read':>11s} {'billed':>9s} {'vs off (raw)':>13s} "
          f"{'vs off (billed)':>16s}  prefix breaks")
    base = rows[0][1]
    for m, b, breaks in rows:
        print(f"   {name(m):12s} {b.requests:4d} {b.tokens:10,.0f} {b.read:11,.0f} {b.billed:9,.0f} "
              f"{100 * (b.tokens - base.tokens) / base.tokens:+12.1f}% {100 * b.vs(base):+15.1f}%  {breaks}")


def main():
    print(f"system chars={len(SYS)}")
    for p in PATTERNS:
        rows = []
        for m in MODES:
            bs = scenario(p, m, system=SYS)
            rows.append((m, bill(bs), append_only([b for _, b in bs])))
        table(f"{p}: warm cache (10 s between forwards)", rows)
    for p in PATTERNS:
        for gap_at in (2, 6):
            gaps = {gap_at: 600.0}
            rows = []
            for m in MODES:
                bs = scenario(p, m, system=SYS, gaps=gaps)
                rows.append((m, bill(bs), append_only([b for _, b in bs])))
            table(f"{p}: 10-minute pause before forward #{gap_at} (cache expired, 5-min TTL)", rows)
    print("\n=== hints and the thinking drop (compaction off vs on, first_sight) ===")
    print(f"   {'scenario':34s} {'breakpoints':>11s} {'off':>8s} {'off+X':>8s} {'X cost':>8s} {'first_sight+X':>14s} {'vs off+X':>9s}")
    for p, kw, label in [("interleaved", {"hints": "always"}, "interleaved, hints=always"),
                         ("bursty", {"hints": "always"}, "bursty, hints=always"),
                         ("lead", {"thinking": True}, "lead (model step 0), thinking on")]:
        for bp in ("harness", "auto"):
            off = bill(scenario(p, None, system=SYS), breakpoints=bp)
            offx = bill(scenario(p, None, system=SYS, **kw), breakpoints=bp)
            fs = bill(scenario(p, "first_sight", system=SYS, **kw), breakpoints=bp)
            print(f"   {label:34s} {bp:>11s} {off.billed:8,.0f} {offx.billed:8,.0f} {100 * offx.vs(off):+7.1f}% "
                  f"{fs.billed:14,.0f} {100 * fs.vs(offx):+8.1f}%")


if __name__ == "__main__":
    main()
