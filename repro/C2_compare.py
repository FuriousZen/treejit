import csv, sys
KEYS = ["success", "model_calls", "small_calls", "tool_calls", "replayed_calls", "tiers"]
print(f"{'seed':>4} {'succ off/cmp/nopath':>22} {'tok/task all200 off/cmp/nopath':>34} {'compacted/task cmp/nopath':>27} {'traj diffs cmp vs nopath':>25}")
for s in range(6):
    rows = {}
    for f, key in ((f"C2_out_s{s}/results.csv", None), (f"C2_out_nopath_s{s}/results.csv", "nopath")):
        for r in csv.DictReader(open(f)):
            rows.setdefault(key or r["mode"], []).append(r)
    off, cmp_, nop = rows["treejit+ok"], rows["treejit+ok+compact"], rows["nopath"]
    diffs = [i for i, (a, b) in enumerate(zip(cmp_, nop)) if any(a[k] != b[k] for k in KEYS)]
    diffs_off = [i for i, (a, b) in enumerate(zip(off, nop)) if any(a[k] != b[k] for k in KEYS)]
    tok = lambda rs: sum(int(r["tokens"]) for r in rs) / len(rs)
    cc = lambda rs: sum(int(r["compacted_chars"]) for r in rs) / len(rs)
    succ = lambda rs: sum(r["success"] == "True" for r in rs)
    print(f"{s:>4} {succ(off):>6}/{succ(cmp_)}/{succ(nop):<10} {tok(off):>10,.0f}/{tok(cmp_):,.0f}/{tok(nop):,.0f}{'':>8} {cc(cmp_):>10,.0f}/{cc(nop):,.0f}{'':>8} {len(diffs):>6} (vs off: {len(diffs_off)}) {diffs[:5]}")
