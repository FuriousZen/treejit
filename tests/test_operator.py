"""Operator workflow: pending approvals, interactive review, short ids, explain, schema migration."""

from __future__ import annotations

import io
import itertools
import json
import sqlite3

import pytest
from conftest import Model, calls_of, replayed_ids
from test_engine import fs_exec, train

from treejit import TreeJIT
from treejit.cli import main as cli
from treejit.operate import IdError, pending, resolve_id

PEOPLE = {f"people/p{i}": f"person {i}" for i in range(30)}
_notes = itertools.count()


def email_policy(task, hist, body):
    """Read -> send_email (commit point) -> touch (write) -> git commit with a free-form message (hole)."""
    who = task.split()[-1]
    plan = [("Read", {"file_path": f"people/{who}"}), ("send_email", {"to": f"{who}@example.com"}),
            ("Bash", {"command": f"touch sent/{who}"}), ("Bash", {"command": f"git commit -m 'note {next(_notes) * 7919}'"})]
    return plan[len(hist)] if len(hist) < len(plan) else None


def _trained(tmp_path, n=4):
    db = str(tmp_path / "op.db")
    jit = TreeJIT(db)
    model = Model(email_policy)
    train(jit, model, [f"email p{i}" for i in range(n)], fs_exec(PEOPLE))
    return db, jit, model


def _tool_items(items):
    return {it["tool"] + ("" if it["tool"] != "Bash" else ":" + it["example"]): it for it in items}


def test_pending_lists_policy_blocked_edges_only(tmp_path, capsys):
    db, jit, _ = _trained(tmp_path)
    blocked = {(r["tool"], r["blocked"]) for r in jit.store.q(
        "SELECT e.tool, ne.blocked FROM node_edges ne JOIN edges e ON e.id=ne.edge WHERE ne.tier='live'")}
    assert ("send_email", "needs_approval") in blocked and ("Bash", "holes") in blocked
    items = pending(jit.store, jit.cfg)
    tools = _tool_items(items)
    assert set(tools) == {"send_email", "Bash:Bash(touch sent/p3)"}, tools
    mail = tools["send_email"]
    assert mail["commit_point"] and mail["pass_runs"] == 4 and mail["fail_runs"] == 0
    assert mail["example"] == "send_email(p3@example.com)" and mail["needs_more_passing_runs"] == 0
    assert not tools["Bash:Bash(touch sent/p3)"]["commit_point"]
    jit.close()

    cli(["--db", db, "pending"])
    out = capsys.readouterr().out
    assert "send_email" in out and "COMMIT POINT" in out and "git commit" not in out
    assert f"treejit approve {mail['edge'][:8]} --node " in out and f"treejit approve {mail['edge'][:8]}\n" in out
    cli(["--db", db, "pending", "--json"])
    assert {it["tool"] for it in json.loads(capsys.readouterr().out)} == {"send_email", "Bash"}
    cli(["--db", db, "show", "--ids"])
    out = capsys.readouterr().out
    assert "[LIVE:needs_approval] send_email" in out and "[LIVE:holes] Bash(git commit" in out
    assert f"edge={mail['edge'][:8]}\n" in out
    assert mail["edge"] not in out, "--ids prints short ids"


def test_pending_skips_tombstones(jit):
    def policy(task, hist, body):
        who = task.split()[-1]
        if not hist:
            return "Read", {"file_path": f"people/{who}"}
        if len(hist) == 1:
            return ("Bash", {"command": "rm -rf build"}) if "BAD" in task else ("send_email", {"to": f"{who}@example.com"})
        return None

    model = Model(policy)
    jit.cfg.hints = "off"
    ex = fs_exec(PEOPLE)
    train(jit, model, ["email p1", "email p2", "email p3"], ex)
    train(jit, model, ["BAD email p4", "BAD email p5", "BAD email p6"], ex, outcome=False, run_prefix="bad")
    rows = jit.store.q("SELECT ne.tier, ne.blocked FROM node_edges ne JOIN edges e ON e.id=ne.edge WHERE e.tool='Bash'")
    assert rows and any(r["tier"] == "tomb" and r["blocked"] == "tomb" for r in rows)
    assert [it["tool"] for it in pending(jit.store, jit.cfg)] == ["send_email"]


def test_commit_point_needs_more_evidence(tmp_path, capsys):
    db, jit, _ = _trained(tmp_path, n=2)  # promoted (2 runs) but a commit point needs promote_runs + 1
    [mail] = [it for it in pending(jit.store, jit.cfg) if it["tool"] == "send_email"]
    assert mail["needs_approval"] and mail["needs_more_passing_runs"] == 1
    jit.close()
    cli(["--db", db, "pending"])
    assert "needs 1 more passing run before replay (even after approval)" in capsys.readouterr().out
    cli(["--db", db, "approve", mail["edge"][:6]])
    capsys.readouterr()
    cli(["--db", db, "pending"])
    [block] = [b for b in capsys.readouterr().out.split("\n\n") if "send_email" in b]
    assert "approved" in block and "commit_point_needs_evidence" in block and "approve here" not in block
    assert "needs 1 more passing run before replay\n" in block
    jit = TreeJIT(db)
    blocked = jit.store.q1("SELECT blocked FROM node_edges WHERE edge=? ORDER BY pass_runs DESC", (mail["edge"],))["blocked"]
    assert blocked == "commit_point_needs_evidence"
    jit.close()


def test_review_approves_one_skips_another_then_replays(tmp_path, capsys, monkeypatch):
    db, jit, model = _trained(tmp_path)
    order = [it["tool"] for it in pending(jit.store, jit.cfg)]
    assert order == ["send_email", "Bash"], order  # shallowest first
    jit.close()
    monkeypatch.setattr("sys.stdin", io.StringIO("maybe\ny\ns\n"))
    cli(["--db", db, "approve", "--review"])
    out = capsys.readouterr().out
    assert "[1/2]" in out and "[2/2]" in out and "please answer" in out and "1 edge(s) approved" in out and "-> left pending" in out
    jit = TreeJIT(db)
    approved = {(r["edge"], r["node"]) for r in jit.store.q("SELECT edge, node FROM approvals")}
    mail_edge = jit.store.q1("SELECT id FROM edges WHERE tool='send_email'")["id"]
    assert approved and all(e == mail_edge and n for e, n in approved)
    assert [it["tool"] for it in pending(jit.store, jit.cfg) if it["needs_approval"]] == ["Bash"]
    [m] = train(jit, model, ["email p20"], fs_exec(PEOPLE), run_prefix="after")
    calls = calls_of(m)
    assert calls[1] == ("send_email", {"to": "p20@example.com"})
    ids = replayed_ids(m)
    assert len(ids) == 2, calls  # Read + send_email replayed; the skipped touch goes to the model
    touch_id = [b["id"] for msg in m if msg["role"] == "assistant" for b in msg["content"] if b["type"] == "tool_use"][2]
    assert "_tj_" not in touch_id
    jit.close()

    # 'e' approves everywhere; q (or EOF) stops
    monkeypatch.setattr("sys.stdin", io.StringIO("e\n"))
    cli(["--db", db, "approve", "--review"])
    capsys.readouterr()
    jit = TreeJIT(db)
    assert jit.store.q1("SELECT 1 FROM approvals WHERE node=''")
    assert not [it for it in pending(jit.store, jit.cfg) if it["needs_approval"]]
    jit.close()


def test_short_id_prefixes(tmp_path, capsys):
    db, jit, _ = _trained(tmp_path, n=3)
    edge = jit.store.q1("SELECT id FROM edges WHERE tool='send_email'")["id"]
    node = jit.store.q1("SELECT node FROM node_edges WHERE edge=?", (edge,))["node"]
    fam = jit.store.q1("SELECT id FROM families")["id"]
    assert resolve_id(jit.store, "edge", edge[:5]) == edge
    assert resolve_id(jit.store, "family", fam[:4]) == fam
    with pytest.raises(IdError, match="at least 4"):
        resolve_id(jit.store, "edge", edge[:3])
    with pytest.raises(IdError, match="unknown"):
        resolve_id(jit.store, "node", "zzzzzz")
    jit.store.x("INSERT INTO edges(id, family, tool) VALUES('beefcafe0001', ?, 'x'), ('beefcafe0002', ?, 'x')", (fam, fam))
    with pytest.raises(IdError, match="ambiguous edge prefix 'beefcafe': matches 2"):
        resolve_id(jit.store, "edge", "beefcafe")
    assert resolve_id(jit.store, "edge", "beefcafe0002") == "beefcafe0002"
    jit.store.x("INSERT INTO runs(id, family) VALUES('email-a1', ?), ('email-a2', ?)", (fam, fam))
    jit.close()

    with pytest.raises(SystemExit, match="ambiguous edge prefix"):
        cli(["--db", db, "approve", "beefcafe"])
    with pytest.raises(SystemExit, match="ambiguous run prefix"):
        cli(["--db", db, "explain", "email-a"])
    cli(["--db", db, "approve", edge[:6], "--node", node[:6]])
    cli(["--db", db, "pin", node[:6], edge[:6]])
    cli(["--db", db, "show", "--family", fam[:6]])
    assert f"approved {edge} at {node}" in capsys.readouterr().out
    jit = TreeJIT(db)
    assert jit.store.approvals() == {(edge, node)} and (node, edge) in jit.store.pins()
    jit.close()
    cli(["--db", db, "revoke", edge[:6], "--node", node[:6]])
    jit = TreeJIT(db)
    assert jit.store.approvals() == set()
    jit.close()


def test_explain_timeline(tmp_path, capsys):
    db, jit, model = _trained(tmp_path)
    mail_edge = jit.store.q1("SELECT id FROM edges WHERE tool='send_email'")["id"]
    jit.store.x("INSERT INTO approvals(edge, node, ts) VALUES(?, '', 0)", (mail_edge,))
    jit.rebuild()
    train(jit, model, ["email p21"], fs_exec(PEOPLE), run_prefix="exp")
    jit.close()
    cli(["--db", db, "explain", "latest", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["run"] == "exp-0" and d["outcome"] == "pass" and d["replayed_calls"] == 2 and d["steps"] == 4
    assert d["model_calls"] == 3 and d["tokens"] > 0
    steps = [r for r in d["timeline"] if r["step"] is not None]
    assert [r["step"] for r in steps] == [0, 1, 2, 3]
    assert steps[0]["who"] in ("T0", "T1") and steps[0]["call"] == "Read(people/p21)" and steps[0]["obs"] == "person 21"
    assert steps[1]["who"] in ("T0", "T1") and "send_email" in steps[1]["note"] and "conf=" in steps[1]["note"]
    assert steps[2]["who"] == "model" and steps[2]["tokens"] == 110 and "not_replayable" in steps[2]["note"]
    assert d["timeline"][-1]["call"].startswith("(no tool call")
    cli(["--db", db, "explain", "exp-"])
    out = capsys.readouterr().out
    assert "run exp-0" in out and "outcome=pass" in out and "2 replayed" in out
    assert "#0  T0" in out or "#0  T1" in out
    assert "why: not_replayable:live" in out and "obs: sent" in out


def test_migrates_old_node_edges_schema(tmp_path):
    db = str(tmp_path / "old.db")
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE families(id TEXT PRIMARY KEY, tools_hash TEXT, prefix TEXT, dialect TEXT,
      created REAL, updated REAL, built_at REAL DEFAULT 0, dirty INTEGER DEFAULT 1);
    CREATE TABLE node_edges(
      node TEXT, edge TEXT, family TEXT, n INTEGER, pass_runs INTEGER, fail_runs INTEGER, pass_n INTEGER,
      blamed REAL, tomb INTEGER, live INTEGER, replayable INTEGER, tier TEXT, purity REAL, success REAL, conf REAL,
      bindings TEXT, holes TEXT, guard TEXT, post TEXT, ref TEXT, reasons TEXT, commit_point INTEGER,
      savings REAL, latency_ms REAL, score REAL, PRIMARY KEY(node, edge));
    INSERT INTO families(id, dirty) VALUES('fam1', 0);
    INSERT INTO node_edges(node, edge, family, tier) VALUES('n1', 'e1', 'fam1', 'hot');
    """)
    con.commit()
    con.close()
    jit = TreeJIT(db)
    cols = [r["name"] for r in jit.store.q("PRAGMA table_info(node_edges)")]
    assert cols[-1] == "blocked"
    assert jit.store.q1("SELECT dirty FROM families WHERE id='fam1'")["dirty"] == 1
    assert jit.store.q1("SELECT tier, blocked FROM node_edges")["tier"] == "hot"
    jit.close()
    # a migrated file keeps working end to end (builder inserts by column name)
    jit = TreeJIT(db)
    model = Model(email_policy)
    train(jit, model, [f"email p{i}" for i in range(3)], fs_exec(PEOPLE))
    assert [it["tool"] for it in pending(jit.store, jit.cfg)] == ["send_email", "Bash"]
    jit.close()
