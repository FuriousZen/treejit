"""CLI, exports, and a benchmark smoke test."""

from __future__ import annotations

import json
import os

from conftest import Model, run_agent

from treejit import TreeJIT
from treejit.cli import main as cli
from treejit_bench.runner import run_suite


def _populate(db):
    jit = TreeJIT(db)

    def policy(task, hist, body):
        p = task.split()[-1]
        plan = [("Bash", {"command": "git status"}), ("Read", {"file_path": p}), ("Bash", {"command": f"cat {p}"})]
        return plan[len(hist)] if len(hist) < 3 else None

    client = jit.wrap(Model(policy), dialect="anthropic")
    for i in range(4):
        run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": f"r{i}"}), f"show src/x{i}.py", lambda n, a: ("out", False))
        jit.outcome(f"r{i}", "pass")
    jit.close()


def test_cli_show_runs_stats_export(tmp_path, capsys):
    db = str(tmp_path / "c.db")
    _populate(db)
    cli(["--db", db, "show", "--ids"])
    out = capsys.readouterr().out
    assert "[HOT ] Bash(git status)" in out and "file_path←$task" in out and "node=" in out
    cli(["--db", db, "runs"])
    assert "r3" in capsys.readouterr().out
    cli(["--db", db, "stats"])
    assert "served by replay" in capsys.readouterr().out
    html = str(tmp_path / "t.html")
    cli(["--db", db, "export", "--format", "html", "--out", html])
    assert "<title>treejit tree</title>" in open(html).read()
    capsys.readouterr()
    cli(["--db", db, "export", "--format", "mermaid"])
    assert capsys.readouterr().out.startswith("flowchart TD")
    skills = str(tmp_path / "skills")
    cli(["--db", db, "export", "--format", "skills", "--out", skills])
    files = [os.path.join(d, f) for d, _, fs in os.walk(skills) for f in fs]
    assert files and open(files[0]).read().startswith("---\nname: treejit-")


def test_cli_pin_approve_prune_outcome(tmp_path, capsys):
    db = str(tmp_path / "c.db")
    _populate(db)
    jit = TreeJIT(db)
    node = jit.store.q1("SELECT id FROM nodes WHERE depth=2")["id"]
    edge = jit.store.q1("SELECT id FROM edges WHERE tool='Read'")["id"]
    jit.close()
    cli(["--db", db, "pin", node])
    cli(["--db", db, "approve", edge])
    cli(["--db", db, "prune", "--days", "0", "--min-hits", "1000", "--dry-run"])
    assert "would evict" in capsys.readouterr().out
    cli(["--db", db, "prune", "--days", "0", "--min-hits", "1000"])
    jit = TreeJIT(db)
    assert jit.store.q1("SELECT COUNT(*) n FROM evictions")["n"] > 0
    assert jit.store.q1("SELECT 1 FROM nodes WHERE id=?", (node,)), "pinned node must survive pruning"
    jit.close()
    cli(["--db", db, "outcome", "r0", "fail", "--reason", "late regression"])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {"updated": ["r0"]}


def test_bench_learning_curve_smoke():
    base = run_suite(40, seed=5, mode="baseline")
    jit = run_suite(40, seed=5, mode="treejit+ok")
    late = lambda rs: sum(r.model_calls for r in rs[-15:]) / 15  # noqa: E731
    assert late(jit) < late(base) / 2
    assert sum(r.success for r in jit) >= sum(r.success for r in base) - 1
    assert sum(r.replayed_calls for r in jit[-15:]) / sum(r.tool_calls for r in jit[-15:]) > 0.6


def test_bench_default_rows_have_no_cost_columns():
    from treejit_bench.runner import to_rows

    rows = to_rows(run_suite(3, seed=0, mode="baseline"))
    assert not {"cache_read", "cache_write", "cost_tokens", "small_cost", "reward"} & set(rows[0])
    assert list(rows[0])[:6] == ["mode", "index", "family", "kind", "success", "reason"]


def test_payload_and_cache_model():
    from treejit_bench.sim import PAYLOAD_TOKENS

    plain = run_suite(20, seed=2, mode="baseline")
    cc = run_suite(20, seed=2, mode="baseline", payload="claude-code")
    cached = run_suite(20, seed=2, mode="baseline", payload="claude-code", cache=True)
    # the payload is added to every full call's input and nothing else changes
    assert [r.model_calls for r in cc] == [r.model_calls for r in plain]
    assert sum(r.input_tokens for r in cc) - sum(r.input_tokens for r in plain) == \
        PAYLOAD_TOKENS["claude-code"] * sum(r.model_calls for r in plain)
    # the cache model splits the same prompt tokens into reads and writes, and makes them cheaper
    assert [r.input_tokens for r in cached] == [r.input_tokens for r in cc]
    assert all(r.cache_read + r.cache_write == r.input_tokens for r in cached)
    assert sum(r.cache_read for r in cached) > 0.8 * sum(r.input_tokens for r in cached)
    assert sum(r.cost_tokens for r in cached) < 0.3 * sum(r.cost_tokens for r in cc)
    # with a Claude-Code-sized payload a small call costs a few percent of a full one (E2)
    ok = run_suite(60, seed=0, mode="treejit+ok", payload="claude-code")
    small = sum(r.small_cost for r in ok) / max(1, sum(r.small_calls for r in ok))
    full = (sum(r.cost_tokens for r in ok) - sum(r.small_cost for r in ok)) / sum(r.model_calls for r in ok)
    assert sum(r.small_calls for r in ok) > 0 and small / full <= 0.05


def test_rebuild_every_k(tmp_path):
    rs = run_suite(30, seed=5, mode="treejit+ok", rebuild_every=5, db=str(tmp_path / "k.db"))
    assert sum(r.replayed_calls for r in rs[-10:]) > 0


def test_bench_cli_cost_columns_and_chart(tmp_path, capsys):
    import csv

    from treejit_bench.__main__ import main

    out = str(tmp_path / "o")
    main(["--tasks", "12", "--modes", "baseline,treejit,treejit+ok,treejit+compact,treejit+ok+compact",
          "--payload", "tau", "--cache", "--out", out])
    text = capsys.readouterr().out
    assert "cost/task" in text and "cache r/w" in text
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert {"cache_read", "cache_write", "cost_tokens"} <= set(rows[0]) and len(rows) == 60
    page = open(os.path.join(out, "learning_curve.html")).read()
    assert "--s5:" in page and "Small calls per task (T2/T3)" in page and "Cost per task" in page
    assert '"cache": true' in page
