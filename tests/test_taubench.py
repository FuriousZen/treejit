"""tau-bench runner (E1). Skipped unless tau-bench is importable (TAUBENCH_PATH=<clone> plus pydantic)."""

from __future__ import annotations

import csv
import json
import os

import pytest

from treejit_bench.taubench import (ClaudeAgent, OracleAgent, ScriptedUser, anthropic_tools, make_suite, oracle_plan,
                                    run_taubench, slip_args, taubench_available)

pytestmark = pytest.mark.skipif(not taubench_available(), reason="tau-bench not available (set TAUBENCH_PATH to a clone)")


def test_suite_tools_system_and_data_restore():
    s = make_suite("retail", "test")
    names = {t["name"] for t in s.tools}
    assert {"find_user_id_by_email", "get_order_details", "cancel_pending_order"} <= names
    assert all(set(t) == {"name", "description", "input_schema"} for t in s.tools)
    assert s.system == s.env.wiki and len(s.system) > 1000
    s.reset(0)
    uid = next(iter(s.env.data["users"]))
    s.env.data["users"][uid]["name"]["first_name"] = "CHANGED"
    s.reset(1)
    assert s.env.data["users"][uid]["name"]["first_name"] != "CHANGED"
    assert anthropic_tools([{"function": {"name": "f"}}])[0]["input_schema"] == {"type": "object", "properties": {}}


def test_oracle_without_noise_scores_full_reward():
    rs = run_taubench(5, "retail", "test", "baseline", noise=0.0)
    assert [r.reward for r in rs] == [1.0] * 5
    assert all(r.model_calls >= 3 and r.small_calls == 0 for r in rs)


def test_slips_are_per_task_and_identical_across_modes():
    s = make_suite("retail", "test")
    a, b = OracleAgent(s, seed=3, noise=0.5), OracleAgent(s, seed=3, noise=0.5)
    for i in range(10):
        t = s.reset(i)
        a.begin(i, t)
        b.begin(i, t)
        assert a.slips == b.slips
    assert any(a.slipped.values())
    plan = oracle_plan(s.reset(0), s.env.data, "retail")
    assert plan[0][0].startswith("find_user_id") and plan[1][0] == "get_user_details"
    name, args = next((n, x) for n, x in plan if n in ("cancel_pending_order", "return_delivered_order_items",
                                                          "exchange_delivered_order_items", "modify_pending_order_items"))
    import random
    assert slip_args(name, args, random.Random(0)) != args


def test_treejit_modes_learn_and_keep_reward(tmp_path):
    base = run_taubench(5, "retail", "test", "baseline", noise=0.2, seed=1)
    ok = run_taubench(5, "retail", "test", "treejit+ok", noise=0.2, seed=1, db=str(tmp_path / "t.db"))
    assert sum(r.reward for r in ok) >= sum(r.reward for r in base)
    assert [("slip" in r.reason) for r in ok] == [("slip" in r.reason) for r in base]
    assert sum(r.model_calls for r in ok) < sum(r.model_calls for r in base)
    assert sum(r.replayed_calls for r in ok) > 0


def test_rebuild_every_runs(tmp_path):
    rs = run_taubench(4, "retail", "test", "treejit", noise=0.0, db=str(tmp_path / "k.db"), rebuild_every=2)
    assert [r.reward for r in rs] == [1.0] * 4


class _FakeSDK:
    """Anthropic-SDK-shaped client backed by the oracle, reporting cache usage like the API."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.bodies = []
        self.messages = self

    def create(self, extra_headers=None, **kw):
        assert not any(k.lower() == "x-treejit-run" for k in (extra_headers or {})), "run header must not go upstream"
        self.bodies.append(kw)
        system = kw["system"][0]["text"] if isinstance(kw.get("system"), list) else kw.get("system", "")
        body = dict(kw, system=system)
        resp = self.oracle(body)
        if not (kw.get("tool_choice") or {}).get("name", "").startswith("treejit_"):
            u = resp["usage"]
            resp["usage"] = {"input_tokens": 10, "cache_read_input_tokens": u["input_tokens"] - 10,
                             "cache_creation_input_tokens": 0, "output_tokens": u["output_tokens"]}
        return resp


def test_claude_agent_plumbing_with_a_fake_client(tmp_path, monkeypatch):
    s = make_suite("retail", "test")
    fake = _FakeSDK(OracleAgent(s, seed=0, noise=0.0))
    orig_begin = ClaudeAgent.begin

    def begin(self, index, task):  # keep the fake oracle on the same task
        orig_begin(self, index, task)
        fake.oracle.begin(index, task)

    monkeypatch.setattr(ClaudeAgent, "begin", begin)
    base = run_taubench(3, "retail", "test", "baseline", agent="claude", client=fake)
    jit = run_taubench(3, "retail", "test", "treejit+ok", agent="claude", client=fake, db=str(tmp_path / "c.db"))
    assert [r.reward for r in base] == [1.0] * 3 and [r.reward for r in jit] == [1.0] * 3
    assert all(r.cache_read > 0 and r.cost_tokens < r.tokens for r in base)
    b = fake.bodies[0]
    assert b["model"] == "claude-opus-5" and b["cache_control"] == {"type": "ephemeral"}
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_scripted_user():
    class T:
        instruction = "hi"
    u = ScriptedUser()
    assert u.first(T) == "hi" and u.reply("Shall I cancel it?") is None
    c = ScriptedUser("confirm", max_turns=1)
    c.first(T)
    assert c.reply("Done.") is None
    assert c.reply("Do you confirm?") == ScriptedUser.CONFIRM
    assert c.reply("Again?") is None


def test_cli_taubench(tmp_path, capsys):
    from treejit_bench.__main__ import main

    out = str(tmp_path / "o")
    main(["--suite", "taubench", "--tau-split", "test", "--tasks", "3", "--modes", "baseline,treejit+ok", "--out", out])
    text = capsys.readouterr().out
    assert "reward" in text and "cost/task" in text
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert len(rows) == 6 and {"reward", "cost_tokens", "cache_read"} <= set(rows[0])
    summ = json.load(open(os.path.join(out, "summary.json")))
    assert 0.0 <= summ["baseline"]["all"]["reward"] <= 1.0 and summ["treejit+ok"]["all"]["n"] == 3
    assert "tau-bench" in open(os.path.join(out, "learning_curve.html")).read()
