"""build_family timings as a running process sees them.

usage: P1_timing.py <db> [n ...]
For each n: a copy of the db holding the family's first n-1 runs; "cold" = the first build in a fresh
process; then the n-th run (with its outcome) is added and "outcome" = the rebuild that follows (what
TreeJIT.outcome pays; caches warm). Best of 3 for the outcome rebuild.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import treejit.builder as B
from treejit import TreeJIT

src = sys.argv[1]
ns = [int(x) for x in sys.argv[2:]] or [100, 300, 1000, 2000]
con = sqlite3.connect(src)
fam = con.execute("SELECT family, COUNT(*) n FROM runs GROUP BY family ORDER BY n DESC").fetchone()[0]
all_runs = con.execute("SELECT * FROM runs WHERE family=? ORDER BY created", (fam,)).fetchall()
con.close()
for n in ns:
    if n > len(all_runs):
        break
    tmp = os.path.join(tempfile.mkdtemp(), "p.db")
    shutil.copy(src, tmp)
    jit = TreeJIT(tmp)
    jit.store.x("DELETE FROM runs")
    ins = f"INSERT OR REPLACE INTO runs VALUES({','.join('?' * len(all_runs[0]))})"
    jit.store.xmany(ins, all_runs[: n - 1])
    if hasattr(B, "clear_caches"):
        B.clear_caches()
    t = time.perf_counter()
    B.build_family(jit.store, jit.cfg, fam)
    cold = time.perf_counter() - t
    best = []
    for _ in range(3):
        jit.store.x("DELETE FROM runs WHERE id=?", (all_runs[n - 1][0],))
        B.build_family(jit.store, jit.cfg, fam)
        jit.store.x(ins, all_runs[n - 1])
        t = time.perf_counter()
        r = B.build_family(jit.store, jit.cfg, fam)
        best.append(time.perf_counter() - t)
    print(f"runs={r['runs']:5d} cold {cold * 1000:7.0f} ms   per-outcome rebuild {min(best) * 1000:7.0f} ms")
    jit.close()
