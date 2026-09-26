"""Clean cProfile (no wrappers) of build_family at n runs, sorted by cumulative time. usage: P1_cumprofile.py <db> <n>"""
import cProfile, io, pstats, shutil, sys, tempfile, os, time
import treejit.builder as B
from treejit import TreeJIT
tmp = os.path.join(tempfile.mkdtemp(), "p.db"); shutil.copy(sys.argv[1], tmp)
jit = TreeJIT(tmp); jit.cfg.max_runs = int(sys.argv[2])
fam = jit.store.q1("SELECT family, COUNT(*) n FROM runs GROUP BY family ORDER BY n DESC")["family"]
B.build_family(jit.store, jit.cfg, fam)  # warm
t = time.perf_counter(); B.build_family(jit.store, jit.cfg, fam); print(f"build (unprofiled) {1000*(time.perf_counter()-t):.0f} ms")
pr = cProfile.Profile(); pr.enable(); B.build_family(jit.store, jit.cfg, fam); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
print("\n".join(l for l in s.getvalue().splitlines() if l.strip()))
