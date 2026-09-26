"""Estimate the win from caching per-step derived data across rebuilds (monkeypatch; repo untouched).
Memoizes shell tokenization, shape_of and node_id (all pure functions of immutable step data) and times
build_family cold vs warm (warm = what a persistent per-step cache / incremental builder would see).
usage: P1_memo.py <db> [n ...]"""
import functools, os, shutil, sys, tempfile, time, json
import treejit.shellwords as SW, treejit.templates as T, treejit.builder as B, treejit.tree as TR
from treejit import TreeJIT

src = sys.argv[1]; ns = [int(x) for x in sys.argv[2:]] or [300, 1000, 2000]
tmp = os.path.join(tempfile.mkdtemp(), "p.db"); shutil.copy(src, tmp)
jit = TreeJIT(tmp)
fam = jit.store.q1("SELECT family, COUNT(*) n FROM runs GROUP BY family ORDER BY n DESC")["family"]


def best(n, k=3):
    jit.cfg.max_runs = n
    ts = []
    for _ in range(k):
        t = time.perf_counter(); B.build_family(jit.store, jit.cfg, fam); ts.append(time.perf_counter() - t)
    return min(ts) * 1000

base = {n: best(n) for n in ns}
tok = functools.lru_cache(maxsize=None)(SW.tokenize)
SW.tokenize = tok; T.tokenize = tok
_shape = T.shape_of
_sc = {}
def shape_of(call):
    k = (call.name, json.dumps(call.args, sort_keys=True))
    if k not in _sc: _sc[k] = _shape(call)
    return _sc[k]
T.shape_of = shape_of; B.shape_of = shape_of
nid = functools.lru_cache(maxsize=None)(TR.node_id)
B.node_id = nid
for n in ns:
    print(f"runs={n:5d}: stock {base[n]:6.0f} ms | memoized tokenize/shape_of/node_id (warm) {best(n):6.0f} ms")
