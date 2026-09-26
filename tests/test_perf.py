"""Rebuild cost (P1): memoization and caches that must not change what is learned, rebuilds off the
proxy's event loop, background rebuild mode, and stale cached views."""

from __future__ import annotations

import asyncio
import json
import random
import re
import shutil
import sqlite3
import threading
import time

import pytest
from conftest import Model, run_agent

import treejit.builder as B
import treejit.engine as E
from treejit import TreeJIT
from treejit.bindings import _json_index, _json_paths, _source_candidates
from treejit.features import learn_decision_list, parse_json
from treejit.model import ToolCall
from treejit.shellwords import tokenize
from treejit.store import dumps
from treejit.templates import TemplateFold, anti_unify
from treejit.util import h

# ------------------------------------------------------------------ synthetic trace logs


def _coding_steps(i: int) -> tuple[str, list[tuple[str, dict, str]]]:
    mod = f"src/mod{i % 97}.py"
    task = f"Remove the unused module {mod}"
    steps = [
        ("Bash", {"command": "git status --short"}, f" M {mod}\n?? notes{i % 7}.txt"),
        ("Read", {"file_path": mod}, f"def handler_{i % 97}(value):\n    return value * {i % 13}\n"),
        ("Bash", {"command": f"grep -rn 'mod{i % 97}' src tests"}, f"src/app.py:{i % 40}:import mod{i % 97}"),
        ("Bash", {"command": f"git rm -q {mod}"}, ""),
        ("Bash", {"command": "python -m pytest -q"}, f"{10 + i % 5} passed in 0.{i % 9}s"),
        ("Bash", {"command": f"git commit -m 'Remove {mod}'"}, f"[main {h(i, n=7)}] Remove {mod}"),
    ]
    return task, steps


def synth_db(path: str, n_runs: int) -> str:
    """A family with n_runs passing runs: one run through the engine (creates the family), the rest
    written straight into the trace log. Returns the family id."""
    jit = TreeJIT(path)
    task0, steps0 = _coding_steps(0)
    model = Model(lambda task, hist, body: steps0[len(hist)][:2] if len(hist) < len(steps0) else None)
    run_agent(lambda b: jit.wrap(model, dialect="anthropic")(b, extra_headers={"X-TreeJIT-Run": "run0"}), task0,
              lambda name, args: (next(o for n, a, o in steps0 if n == name and a == args), False))
    fam = jit.store.q1("SELECT id FROM families")["id"]
    for i in range(1, n_runs):
        task, steps = _coding_steps(i)
        rid = f"run{i}"
        jit.store.upsert_run(rid, fam, task, h(task))
        jit.store.write_steps(rid, 0, [(k, f"toolu_{i}_{k}", n, dumps(a), o, 0, 0) for k, (n, a, o) in enumerate(steps)])
    jit.store.x("UPDATE runs SET outcome='pass', outcome_at=updated")
    jit.close()
    return fam


def tree_rows(store, family: str) -> dict:
    return {t: sorted(json.dumps(tuple(r), default=str) for r in store.q(f"SELECT * FROM {t} WHERE family=?", (family,)))
            for t in ("edges", "nodes", "node_edges")}


# ------------------------------------------------------------------ (a) memoization, load_runs, template fold


def test_tokenize_memo_returns_fresh_lists():
    a = tokenize("git commit -m 'x'")
    a.append("junk")
    assert tokenize("git commit -m 'x'") == a[:-1]


def test_load_runs_reads_only_the_selected_runs(tmp_path):
    db = str(tmp_path / "t.db")
    fam = synth_db(db, 30)
    jit = TreeJIT(db)
    fetched = []
    q = jit.store.q

    def counting(sql, args=()):
        rows = q(sql, args)
        if "FROM steps" in sql:
            fetched.extend(rows)
        return rows
    jit.store.q = counting
    runs = B.load_runs(jit.store, fam, 5)
    assert len(runs) == 5 and [r.id for r in runs] == [f"run{i}" for i in range(25, 30)]
    assert len(fetched) == sum(len(r.steps) for r in runs) == 30  # not the family's 180 steps


def _random_command(rng: random.Random) -> str:
    prog = rng.choice(["git commit -m", "grep -rn", "git rm -q", "sed -i s/a/b/", "ls -la"])
    words = [rng.choice(["x", "'a b'", "src/m.py", "-v", "--all", "\"q\"", "tests", "$HOME", "1"]) for _ in range(rng.randint(0, 4))]
    tail = rng.choice(["", " | head -5", " && echo done", " > out.txt"])
    return f"{prog} {' '.join(words)}{tail}".strip()


def test_template_fold_resumes_exactly_like_anti_unify():
    rng = random.Random(3)
    for trial in range(150):
        n = rng.randint(1, 25)
        calls = [ToolCall(f"c{i}", "Bash", {"command": _random_command(rng), "timeout": rng.choice([10, 10, 10, 30])})
                 for i in range(n)]
        keys = [dumps(c.args) for c in calls]
        fold = TemplateFold()
        cut = 0
        while cut < n:
            nxt = min(n, cut + rng.randint(1, 5))
            fold = fold.extend(calls[cut:nxt], keys[cut:nxt])
            cut = nxt
            assert fold.template() == anti_unify(calls[:cut]), (trial, cut)
        assert fold.keys == keys


def test_template_fold_bails_out_where_anti_unify_changes_kind():
    a, b = ToolCall("1", "Bash", {"command": "ls a"}), ToolCall("2", "Bash", {"command": ["ls", "b"]})
    assert TemplateFold().extend([a], ["k1"]).extend([b], ["k2"]) is None
    other = [ToolCall("1", "t", {"x": 1}), ToolCall("2", "t", {"x": True}), ToolCall("3", "t", {"x": 1.0})]
    fold = TemplateFold().extend(other[:1], ["a"]).extend(other[1:], ["b", "c"])
    assert fold.template() == anti_unify(other)


def test_incremental_rebuilds_match_from_scratch(tmp_path, monkeypatch):
    """Rebuild after every few new runs (caches warm: folds resume, memos hit) and compare every row of
    edges/nodes/node_edges with a from-scratch build (all caches cleared)."""
    from treejit_bench.runner import run_suite

    src = str(tmp_path / "bench.db")
    run_suite(40, seed=5, family="mixed", mode="treejit+ok", db=src)
    monkeypatch.setattr(B, "now", lambda: 2e9)
    con = sqlite3.connect(src)
    all_runs = con.execute("SELECT * FROM runs ORDER BY created").fetchall()
    con.close()
    db = str(tmp_path / "grow.db")
    shutil.copy(src, db)
    jit = TreeJIT(db)
    jit.store.x("DELETE FROM runs")
    fams = [r["id"] for r in jit.store.q("SELECT id FROM families")]
    resumed = []
    extend = TemplateFold.extend
    monkeypatch.setattr(TemplateFold, "extend", lambda self, calls, keys: resumed.append(len(self.keys)) or extend(self, calls, keys))
    checked = 0
    for k in range(4, len(all_runs) + 1, 6):
        for r in all_runs[:k]:
            jit.store.x(f"INSERT OR IGNORE INTO runs VALUES({','.join('?' * len(r))})", r)
        for f in fams:
            B.build_family(jit.store, jit.cfg, f)
            warm = tree_rows(jit.store, f)
            B.clear_caches()
            B.build_family(jit.store, jit.cfg, f)
            assert tree_rows(jit.store, f) == warm, (k, f)
            checked += sum(len(v) for v in warm.values())
            B.build_family(jit.store, jit.cfg, f)  # re-warm for the next, larger k
    assert checked > 500 and sum(1 for n in resumed if n) > 20  # folds did resume


def _old_json_paths(js, target, prefix=None, depth=4):
    return _json_paths(js, target, prefix, depth)


def test_json_index_is_json_paths_for_every_value():
    rng = random.Random(0)

    def gen(d):
        if d == 0 or rng.random() < 0.3:
            return rng.choice([1, 1.0, True, None, "a", "b", "#W1", 17, "17", False])
        if rng.random() < 0.5:
            return {rng.choice("abcdef"): gen(d - 1) for _ in range(rng.randint(0, 4))}
        return [gen(d - 1) for _ in range(rng.choice([0, 1, 2, 3, 22]) if d <= 2 else rng.randint(0, 3))]
    for _ in range(200):
        js = gen(6)  # deeper than the 4 levels indexed, lists longer than the 20 items indexed
        text = json.dumps(js)
        index = _json_index(text)
        if not isinstance(js, (dict, list)):
            continue
        targets = set(index) | {"a", "1", "true", "zzz"}
        for t in targets:
            assert [list(p) for p in index.get(t, [])] == _old_json_paths(parse_json(text), t)


def _reference_source_candidates(src, t, target):
    """bindings._source_candidates before the per-text index (P1), kept to check candidate order."""
    from treejit.bindings import PATTERNS, _STRIP

    def lines(t):
        return [line.strip() for line in t.splitlines() if line.strip()]

    def toks(t):
        return [w for w in (x.strip(_STRIP) for x in t.split()) if w]

    def re_all(name, t):
        return [m.group(m.lastindex) if m.lastindex else m.group(0) for m in PATTERNS[name].finditer(t)]
    c = []
    js = parse_json(t)
    if t.strip() == target:
        c.append(["x", src, ["whole"]])
    if js is not None:
        for path in _json_paths(js, target)[:3]:
            c.append(["x", src, ["json", path]])
    if target not in t:
        return c
    for m in re.finditer(r"(?im)^[ \t>*-]*([A-Za-z][\w .-]{0,30}?)[ \t]*[:=][ \t]*(.+?)[ \t]*\.?[ \t]*$", t):
        if m.group(2).strip() == target:
            c.append(["x", src, ["kv", m.group(1).strip()]])
    for name in PATTERNS:
        ms = re_all(name, t)
        if target in ms:
            i = ms.index(target)
            c.append(["x", src, ["re", name, i]])
            if i == len(ms) - 1 and i > 0:
                c.append(["x", src, ["re", name, -1]])
    ts = toks(t)
    for j, w in enumerate(ts):
        if w == target and j > 0:
            anchor = ts[j - 1].lower().rstrip(":#")
            if re.fullmatch(r"[a-z][\w-]*", anchor):
                c.append(["x", src, ["after", anchor]])
    ls = lines(t)
    if target in ls:
        i = ls.index(target)
        c.append(["x", src, ["line", i]])
        c.append(["x", src, ["line", i - len(ls)]])
    if target in ts:
        i = ts.index(target)
        c.append(["x", src, ["tok", i]])
        c.append(["x", src, ["tok", i - len(ts)]])
    return c


def test_source_candidates_unchanged_by_the_index():
    texts = [
        json.dumps({"order_id": "#W123", "user_id": "u_1", "items": [{"item_id": "42", "price": 9.5}, {"item_id": "43"}],
                    "status": "pending", "nested": {"a": {"b": {"c": {"d": "#W123"}}}}}),
        json.dumps([{"id": "#W123"}, "#W123", 42, True]),
        "Order: #W123\nstatus = pending\nemail: a.b@example.com\n  version 1.2.3 of src/app.py at abc1234\n",
        "user id u_1 found. Next: #W123, then 42 and 42.",
        "#W123",
    ]
    targets = ["#W123", "42", "pending", "u_1", "a.b@example.com", "1.2.3", "src/app.py", "abc1234", "true", "9.5", "nope"]
    for t in texts:
        for target in targets:
            for src in (["obs", 1], "task"):
                assert _source_candidates(src, t, target) == _reference_source_candidates(src, t, target), (t, target)


def test_shared_only_keeps_the_decision_list():
    rng = random.Random(1)
    vals = [1, True, 1.0, 0, False, "x", "y", None, 2, "1"]
    for _ in range(300):
        n = rng.randint(3, 40)
        examples = []
        for i in range(n):
            feats = {f"json.k{j}": rng.choice(vals) for j in range(rng.randint(0, 5))}
            feats.update({f"json.id{rng.randint(0, 50)}": rng.randint(0, 10 ** 6) for _ in range(rng.randint(0, 6))})
            feats["err"] = rng.random() < 0.3
            examples.append((rng.choice("AB"), feats, rng.choice(["", "ok\nfine", "ok"]), {rng.choice(["x", "y", "z"])}))
        negs = [(rng.choice("AB"), {"json.k0": rng.choice(vals)}, "", {"x"}) for _ in range(rng.randint(0, 3))]
        want = learn_decision_list(examples, 0.8, negatives=negs, confirmed=examples[:2], class_sets=True)
        got = learn_decision_list(B._shared_only(examples), 0.8, negatives=negs, confirmed=examples[:2], class_sets=True)
        assert got == want


# ------------------------------------------------------------------ (c) /outcome off the event loop


def test_health_answers_during_outcome_on_a_1000_run_db(tmp_path):
    httpx = pytest.importorskip("httpx")
    from treejit.proxy import ProxyApp

    db = str(tmp_path / "big.db")
    synth_db(db, 1000)
    jit = TreeJIT(db)
    jit.rebuild()
    jit.store.x("UPDATE runs SET outcome=NULL WHERE id='run999'")
    app = ProxyApp(jit)
    assert jit.rebuild_mode == "background"

    async def main():
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tj")
        lat: list[float] = []
        done = False

        async def health():
            while not done:
                t = time.perf_counter()
                r = await client.get("/health")
                lat.append(time.perf_counter() - t)
                assert r.status_code == 200
                await asyncio.sleep(0.005)
        h_task = asyncio.create_task(health())
        await asyncio.sleep(0.02)
        t0 = time.perf_counter()
        r = await client.post("/outcome", json={"run_id": "run999", "outcome": "pass"})
        took = time.perf_counter() - t0
        done = True
        await h_task
        return r, took, lat
    r, took, lat = asyncio.run(main())
    assert r.status_code == 200 and r.json()["updated"] == ["run999"]
    # the rebuild finished before /outcome answered (so the next request sees the new tree) ...
    assert jit.store.q1("SELECT dirty FROM families")["dirty"] == 0
    # ... and /health kept answering meanwhile
    during = [x for x in lat[1:]]
    assert took > 0.05 and len(during) >= 3
    assert max(during) < 0.05, f"/health took {max(during) * 1000:.0f} ms during a {took * 1000:.0f} ms outcome"
    jit.close()


def test_outcome_wait_false_answers_before_the_rebuild(tmp_path, monkeypatch):
    httpx = pytest.importorskip("httpx")
    from treejit.proxy import ProxyApp

    db = str(tmp_path / "t.db")
    synth_db(db, 10)
    jit = TreeJIT(db)
    gate = threading.Event()
    real = E.build_family

    def slow(store, cfg, family):
        gate.wait(5)
        return real(store, cfg, family)
    monkeypatch.setattr(E, "build_family", slow)
    app = ProxyApp(jit)

    async def main():
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tj")
        return await client.post("/outcome", json={"run_id": "run3", "outcome": "fail", "wait": False})
    r = asyncio.run(main())
    assert r.status_code == 200 and r.json()["updated"] == ["run3"]
    assert jit.store.q1("SELECT dirty FROM families")["dirty"] == 1
    gate.set()
    assert jit.wait_rebuilds(10)
    assert jit.store.q1("SELECT dirty FROM families")["dirty"] == 0
    jit.close()


# ------------------------------------------------------------------ background mode


def test_background_rebuilds_coalesce_and_never_run_on_the_request_thread(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    fam = synth_db(db, 12)
    jit = TreeJIT(db, rebuild="background")
    jit.rebuild()
    monkeypatch.setattr(B, "now", lambda: 2e9)
    builds: list[int] = []
    gate = threading.Event()
    real = E.build_family

    def slow(store, cfg, family):
        builds.append(threading.get_ident())
        gate.wait(5)
        return real(store, cfg, family)
    monkeypatch.setattr(E, "build_family", slow)
    old = jit.view(fam)
    for i in range(1, 8):
        assert jit.outcome(f"run{i}", "fail") == [f"run{i}"]   # returns at once
    # while the first build is blocked, requests get the previous view without building
    assert jit.view(fam) is old
    assert all(t != threading.get_ident() for t in builds)
    gate.set()
    assert jit.wait_rebuilds(10)
    assert 1 <= len(builds) <= 2, builds       # 7 outcomes: the running build plus one coalesced build
    new = jit.view(fam)
    assert new is not old and jit.store.q1("SELECT dirty FROM families")["dirty"] == 0
    swapped = tree_rows(jit.store, fam)
    B.clear_caches()
    jit.rebuild(fam)
    assert tree_rows(jit.store, fam) == swapped  # the coalesced build saw every outcome
    jit.close()


def test_bad_rebuild_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        TreeJIT(str(tmp_path / "t.db"), rebuild="later")


# ------------------------------------------------------------------ (d) stale views


@pytest.mark.parametrize("mode", ["sync", "background"])
def test_outcome_in_another_instance_makes_this_one_replay(tmp_path, mode):
    db = str(tmp_path / "t.db")
    proxy, cli = TreeJIT(db, rebuild=mode), TreeJIT(db)
    model = Model(lambda task, hist, body: ("Read", {"file_path": "a.txt"}) if not hist else None)
    client = proxy.wrap(model, dialect="anthropic")
    ex = lambda n, a: ("hello", False)  # noqa: E731
    for i in range(3):
        rid = f"r{i}"
        run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), f"read a.txt {i}", ex)
        (proxy if i == 0 else cli).outcome(rid, "pass")  # the rest reported by another process (the CLI)
        proxy.wait_rebuilds(10)
    calls0 = model.calls
    run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "r9"}), "read a.txt 9", ex)
    assert model.calls - calls0 == 1  # the Read replays; only the final answer comes from the model
    proxy.close()
    cli.close()


def test_view_reloads_only_when_built_at_changes(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    fam = synth_db(db, 5)
    a, b = TreeJIT(db), TreeJIT(db)
    v1 = a.view(fam)
    assert a.view(fam) is v1                   # nothing changed: the cached view
    b.rebuild(fam)
    v2 = a.view(fam)
    assert v2 is not v1 and a.view(fam) is v2  # rebuilt elsewhere: reloaded once
    a.close()
    b.close()
