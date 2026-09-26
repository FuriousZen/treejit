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
