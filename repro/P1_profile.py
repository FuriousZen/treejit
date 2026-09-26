"""Profile build_family at several run counts on a generated trace log.

usage: P1_profile.py <db> [n ...]    (uses cfg.max_runs = n, i.e. the most recent n runs)
Prints wall time (best of 3), a per-phase breakdown (wrapped helpers in builder's namespace), and cProfile top-N.
"""
import cProfile, io, pstats, shutil, sys, tempfile, time, os, collections
import treejit.builder as B
from treejit import TreeJIT

src = sys.argv[1]
ns = [int(x) for x in sys.argv[2:]] or [100, 300, 1000, 2000]
tmp = os.path.join(tempfile.mkdtemp(), "p.db")
shutil.copy(src, tmp)
jit = TreeJIT(tmp)
fam = jit.store.q1("SELECT family, COUNT(*) n FROM runs GROUP BY family ORDER BY n DESC")["family"]

acc = collections.defaultdict(float)
cnt = collections.Counter()


def wrap(name):
    f = getattr(B, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            acc[name] += time.perf_counter() - t
            cnt[name] += 1
    setattr(B, name, g)


for name in ("load_runs", "_usage_by_call", "anti_unify", "call_slots", "obs_features", "contexts", "find_rule",
             "learn_decision_list", "guard_of", "postcondition", "excess_negatives", "is_readonly", "is_commit_point"):
    wrap(name)

_tx = jit.store.transaction


def tx():
    t = time.perf_counter()
    ctx = _tx()

    class W:
        def __enter__(s):
            return ctx.__enter__()

        def __exit__(s, *e):
            r = ctx.__exit__(*e)
            acc["sql_write(tx)"] += time.perf_counter() - t
            return r
    return W()
jit.store.transaction = tx

for n in ns:
    jit.cfg.max_runs = n
    times = []
    for rep in range(3):
        acc.clear(); cnt.clear()
        t = time.perf_counter()
        r = B.build_family(jit.store, jit.cfg, fam)
        times.append(time.perf_counter() - t)
    total = min(times)
    stats = dict(acc)
    nsteps = jit.store.q1(f"SELECT SUM(n_steps) s FROM (SELECT n_steps FROM runs WHERE family=? ORDER BY created DESC LIMIT ?)", (fam, n))["s"]
    print(f"\n=== runs={r['runs']} steps={nsteps} edges={r['edges']} nodes={r['nodes']} hot={r['hot']}  build={total*1000:.0f} ms (best of 3)")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"   {k:22s} {v*1000:8.1f} ms  ({100*v/times[-1]:4.1f}%)  calls={cnt[k]}")
    if n in (300, 2000):
        pr = cProfile.Profile()
        pr.enable(); B.build_family(jit.store, jit.cfg, fam); pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
        print("\n".join(l for l in s.getvalue().splitlines() if l.strip())[:6000])
