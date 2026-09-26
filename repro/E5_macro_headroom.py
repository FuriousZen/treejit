"""E5: headroom for macros-as-tools. For each T4 request that produced a tool call, what served
the next requests of the same run? If the tail is already T0/T1 (often via g-contexts = last-k macros),
a macro tool could only save the entry T4 itself (which it cannot: the model still has to call the macro)."""
import sqlite3, sys, collections
for db in sys.argv[1:]:
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    runs = collections.defaultdict(list)
    for r in con.execute("SELECT run_id, tier, note, n_calls, call_ids FROM requests WHERE run_id IS NOT NULL ORDER BY id"):
        runs[r["run_id"]].append(r)
    order = [r[0] for r in con.execute("SELECT id FROM runs ORDER BY created")]
    late = order[len(order) // 2:]
    t4_tool = t4_final = 0
    after = collections.Counter()
    tail_len = []
    for rid in late:
        rs = runs[rid]
        for i, r in enumerate(rs):
            if r["tier"] != "T4":
                continue
            if not r["n_calls"]:
                t4_final += 1
                continue
            t4_tool += 1
            k = 0
            for nxt in rs[i + 1:]:
                if nxt["tier"] in ("T0", "T1"):
                    k += 1
                    after["g" if "@g" in (nxt["note"] or "") else "r"] += 1
                else:
                    after["stop:" + nxt["tier"]] += 1
                    break
            tail_len.append(k)
    n = len(late)
    print(f"{db}: late runs={n} T4 with tool call/run={t4_tool / n:.2f} T4 final answer/run={t4_final / n:.2f}")
    print(f"   after a tool-producing T4: mean replayed tail={sum(tail_len) / max(1, len(tail_len)):.2f} steps; "
          f"tail served via g-context {after['g']} vs root path {after['r']}; tails ended by {dict((k, v) for k, v in after.items() if k.startswith('stop'))}")
