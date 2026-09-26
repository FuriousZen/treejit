"""E4: explain accounts for every request. Failed small calls join their run and show as such,
steps carry the tier that produced them (T2 / ck / T3), user turns sit between the steps."""

from __future__ import annotations

import asyncio
import json
import random

import httpx

from conftest import SYSTEM, TOOLS, text_msg, tool_msg

from treejit import TreeJIT, operate
from treejit.cli import main as cli
from treejit.proxy import ProxyApp
from treejit.subcalls import answer_content, subcall_tool
from treejit_bench.runner import _execute, _fresh_jit, _upstream_app, parse_anthropic_sse
from treejit_bench.sim import SimModel, make_task

BASH = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]


def _all_tokens(store):
    return store.q1("SELECT COALESCE(SUM(input_tokens+output_tokens+cache_read+cache_write), 0) n FROM requests "
                    "WHERE family IS NOT NULL")["n"]


def _explained_tokens(store):
    return sum(operate.explain(store, r["id"])["tokens"] for r in store.q("SELECT id FROM runs"))


def test_failed_first_step_subcall_joins_its_run():
    """A T2 subcall on the first request of a header-less run that fails (the model says 'new'):
    its row used to keep run_id NULL, so explain dropped its tokens and showed nothing."""
    small = {"n": 0}

    def model(body):
        forced = subcall_tool(body)
        if forced == "treejit_choose":
            small["n"] += 1
            return {"id": "m", "type": "message", "role": "assistant", "model": "x", "stop_reason": "end_turn",
                    "content": answer_content(body, {"choice": 0}, f"toolu_s{small['n']:012d}"),
                    "usage": {"input_tokens": 300, "output_tokens": 20}}
        msgs = body["messages"]
        task = msgs[0]["content"]
        n = sum(1 for m in msgs if m["role"] == "assistant")
        plan = {"alpha": ["ls", "pwd"], "beta": ["date", "pwd"], "gamma": ["whoami", "pwd"]}[task.split()[0]]
        if n < len(plan):
            content, stop = [{"type": "tool_use", "id": f"toolu_{len(msgs)}_{task.split()[1]}_xxxxxxxx", "name": "Bash",
                              "input": {"command": plan[n]}}], "tool_use"
        else:
            content, stop = [{"type": "text", "text": "done"}], "end_turn"
        return {"id": "m", "type": "message", "role": "assistant", "model": "x", "stop_reason": stop, "content": content,
                "usage": {"input_tokens": 1000, "output_tokens": 50}}

    jit = TreeJIT(":memory:")
    w = jit.wrap(model, dialect="anthropic")
    for t in [f"{k} task{i}" for i in range(6) for k in ("alpha", "beta")] + ["gamma task99"]:
        msgs = [{"role": "user", "content": t}]
        for _ in range(5):
            r = w({"model": "x", "max_tokens": 100, "system": SYSTEM, "tools": BASH, "messages": msgs})
            msgs.append({"role": "assistant", "content": r["content"]})
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
        jit.outcome("latest", "pass")
    assert small["n"] > 0
    assert not jit.store.q("SELECT 1 FROM requests WHERE family IS NOT NULL AND run_id IS NULL")
    assert _explained_tokens(jit.store) == _all_tokens(jit.store)
    rid = jit.store.q1("SELECT id FROM runs WHERE task='gamma task99'")["id"]
    d = operate.explain(jit.store, rid)
    assert d["small_calls"] == 1 and d["small_failed"] == 1 and d["small_tokens"] == 320
    first = d["timeline"][0]
    assert first["who"] == "T2" and first["call"] == "(T2 small call failed: chose_new → model)" and first["tokens"] == 320
    assert [r["who"] for r in d["timeline"]] == ["T2", "model", "model", "model"]
    assert d["timeline"][-1]["call"] == "(no tool call: final answer)"
    text = operate.explain_text(d)
    assert "small calls: 1 (1 failed), 320 tokens" in text and "T2 small call failed: chose_new" in text
    jit.close()


def test_explain_labels_subcall_steps_from_their_call_ids(jit):
    """A step a T2 pick, a budget checkpoint or a T3 fill produced is labelled by its call-id suffix."""
    jit.store.upsert_run("lab", "fam", "t", "th")
    rows = [(0, "toolu_tj_aaaaaaaaaaaa_ffAAAAAAAAAA_t2", "Bash", '{"command":"a"}', "x", 0, 0),
            (1, "toolu_tj_aaaaaaaaaaaa_ffBBBBBBBBBB_ck", "Bash", '{"command":"b"}', "x", 0, 1),
            (2, "toolu_tj_aaaaaaaaaaaa_ffCCCCCCCCCC_t3", "Bash", '{"command":"c"}', "x", 0, 1),
            (3, "user_9_abcdef01", "$user:yes", "{}", "yes please", 0, 0)]
    jit.store.write_steps("lab", 0, rows)
    for i, (cid, tier, note) in enumerate([(rows[0][1], "T2", "ambiguous@r0: Bash(a) | Bash(z); ok: Bash(a)"),
                                           (rows[1][1], "T2", "budget@r1: Bash(b); ok: Bash(b)"),
                                           (rows[2][1], "T3", "holes@r2: Bash(c $0); ok: Bash(c $0)")]):
        jit.store.log_request(family="fam", run_id="lab", tier=tier, n_calls=1, call_ids=json.dumps([cid]), note=note,
                              status=200, input_tokens=100, at_step=i)
    jit.store.log_request(family="fam", run_id="lab", tier="T4", n_calls=0, call_ids="[]", note="end@r3", status=200,
                          input_tokens=1000, at_step=3)
    jit.store.log_request(family="fam", run_id="lab", tier="T4", n_calls=0, call_ids="[]", note="end@r4", status=200,
                          input_tokens=1000, at_step=4)
    d = operate.explain(jit.store, "lab")
    assert [(r["who"], r["step"]) for r in d["timeline"]] == [("T2", 0), ("ck", 1), ("T3", 2), ("model", None), ("user", 3),
                                                               ("model", None)]
    assert d["timeline"][3]["call"] == "(no tool call: replied to the user)" and d["timeline"][4]["call"] == "(yes) yes please"
    assert d["timeline"][-1]["call"] == "(no tool call: final answer)"
    assert d["steps"] == 3 and d["user_turns"] == 1 and d["small_calls"] == 3 and d["small_failed"] == 0
    assert "small calls: 3 (0 failed), 300 tokens" in operate.explain_text(d)
    # no small-call row is ever shown as a final answer
    assert not any(r["who"] in ("T2", "T3", "ck") and "final answer" in r["call"] for r in d["timeline"])


def test_explain_shows_user_turns_in_order(jit):
    def model(body):
        msgs = body["messages"]
        if len(msgs) == 1:
            return tool_msg("Read", {"file_path": "a.py"})
        if isinstance(msgs[-1]["content"], list):
            return text_msg("Should I fix it?") if len(msgs) == 3 else text_msg("Fixed.")
        return tool_msg("Bash", {"command": "sed -i s/a/b/ a.py"})

    client = jit.wrap(model, dialect="anthropic")
    msgs = [{"role": "user", "content": "fix a.py"}]
    for reply in ("yes", None):
        for _ in range(4):
            r = client({"model": "m", "max_tokens": 50, "system": SYSTEM, "tools": TOOLS, "messages": msgs},
                       extra_headers={"X-TreeJIT-Run": "u1"})
            msgs.append({"role": "assistant", "content": r["content"]})
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
        if reply:
            msgs.append({"role": "user", "content": reply})
    d = operate.explain(jit.store, "u1")
    assert [(r["who"], r["step"]) for r in d["timeline"]] == [("model", 0), ("model", None), ("user", 1), ("model", 2),
                                                               ("model", None)]
    assert "(yes) yes" == d["timeline"][2]["call"] and "later user turn" in operate.explain_text(d)


async def _proxy_run(n: int, seed: int, db: str):
    rng = random.Random(seed)
    tasks = [make_task(rng, i, "mixed") for i in range(n)]
    jit = _fresh_jit(db, "treejit+ok")
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=_upstream_app(SimModel(seed + 1))), base_url="http://upstream")
    jit.cfg.anthropic_upstream = "http://upstream"
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://treejit")
    for fam, text, env, tools, system in tasks:
        msgs = [{"role": "user", "content": text}]
        for _ in range(30):
            body = {"model": "sim-1", "max_tokens": 1024, "system": system, "tools": tools, "messages": msgs, "stream": True}
            r = await client.post("/v1/messages", json=body, headers={"x-api-key": "k", "anthropic-version": "2023-06-01"})
            resp = parse_anthropic_sse(r.content)
            msgs.append({"role": "assistant", "content": resp["content"]})
            blocks = _execute(env, resp["content"])
            if not blocks:
                break
            msgs.append({"role": "user", "content": blocks})
        ok, why = env.verify()
        await client.post("/outcome", json={"run_id": "latest", "outcome": "pass" if ok else "fail"})
    await client.aclose()
    await up.aclose()
    return jit


def test_header_less_proxy_run_accounts_for_every_request(tmp_path):
    jit = asyncio.run(_proxy_run(40, 0, str(tmp_path / "e4.db")))
    st = jit.store
    assert not st.q("SELECT 1 FROM requests WHERE family IS NOT NULL AND run_id IS NULL")
    assert not st.q("SELECT 1 FROM runs WHERE outcome IS NULL")
    assert _explained_tokens(st) == _all_tokens(st)
    small = st.q1("SELECT COUNT(*) n FROM requests WHERE tier IN ('T2','T3')")["n"]
    assert small > 0 and sum(operate.explain(st, r["id"])["small_calls"] for r in st.q("SELECT id FROM runs")) == small
    for r in st.q("SELECT id FROM runs"):
        for row in operate.explain(st, r["id"])["timeline"]:
            assert not (row["who"] in ("T2", "T3", "ck") and "final answer" in row["call"])
    jit.close()


def test_cli_db_after_subcommand(tmp_path, capsys):
    db = str(tmp_path / "x.db")
    jit = TreeJIT(db)
    jit.store.upsert_run("cli-run", "fam", "a task", "th")
    jit.close()
    cli(["explain", "cli-run", "--db", db])
    assert "run cli-run" in capsys.readouterr().out
    cli(["--db", db, "explain", "cli-run"])
    assert "run cli-run" in capsys.readouterr().out
