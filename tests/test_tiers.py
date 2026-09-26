"""T2 (choose / checkpoint) and T3 (fill holes): one small forced-tool call instead of a full one."""

from __future__ import annotations

import asyncio
import json
import re
import shlex

import pytest
from conftest import Model, calls_of, replayed_ids, run_agent

from treejit import TreeJIT
from treejit.config import Config
from treejit.dialects import get as dialect
from treejit.model import ToolCall
from treejit.replay import Option, Subcall
from treejit.subcalls import CHOOSE_TOOL, FILL_TOOL, build, resolve, tool_input
from treejit.templates import anti_unify, var_slots
from treejit.tree import EdgeInfo, NodeEdge
from treejit.util import now


class SubModel(Model):
    """A scripted model that also answers treejit subcalls via `answer(body) -> tool input | response`."""

    def __init__(self, policy, answer=None) -> None:
        super().__init__(policy)
        self.answer = answer
        self.small = 0
        self.sub_bodies: list[dict] = []

    def __call__(self, body: dict) -> dict:
        tc = body.get("tool_choice") or {}
        name = tc.get("name", "")
        if not name.startswith("treejit_"):
            return super().__call__(body)
        self.small += 1
        self.sub_bodies.append(body)
        out = self.answer(body)
        if out.get("type") == "message":
            return out
        return {"id": "msg_s", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "toolu_sub", "name": name, "input": out}],
                "usage": {"input_tokens": 30, "output_tokens": 8}}


def fs_exec(files: dict[str, str]):
    def ex(name, args):
        if name == "Read":
            p = args["file_path"]
            return (files[p], False) if p in files else (f"no such file {p}", True)
        return f"ran: {args['command']}", False
    return ex


def train(jit, model, tasks, exec_, prefix="t"):
    client = jit.wrap(model, dialect="anthropic")
    out = []
    for i, t in enumerate(tasks):
        rid = f"{prefix}-{i}"
        out.append(run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), t, exec_))
        jit.outcome(rid, "pass")
    return out


def approve_all(jit):
    jit.store.x("INSERT OR REPLACE INTO approvals(edge, node, ts) VALUES('*', '', ?)", (now(),))
    jit.rebuild()


# ------------------------------------------------------------------ T3

MSGS = ["Refresh docs", "Small cleanup", "Polish wording", "Minor fix", "Tidy up"]
FILES = {f"src/m{i}.py": f"code {i}" for i in range(10)}


def commit_policy(task, hist, body):
    path = task.split()[-1]
    k = int(re.search(r"\d+", path).group())
    plan = [("Read", {"file_path": path}), ("Bash", {"command": f"git commit -m {shlex.quote(MSGS[k % 5])}"})]
    return plan[len(hist)] if len(hist) < 2 else None


def fill_all(value):
    def answer(body):
        props = body["tools"][0]["input_schema"]["properties"]
        return {p: value for p in props if p != "not_this_step"}
    return answer


def _commit_setup(jit, answer):
    model = SubModel(commit_policy, answer)
    train(jit, model, [f"commit src/m{i}.py" for i in range(3)], fs_exec(FILES))
    approve_all(jit)
    ne = jit.store.q1("SELECT holes, tier FROM node_edges ne JOIN edges e ON e.id=ne.edge WHERE e.label LIKE '%commit%' "
                      "AND ne.pass_runs >= 2")
    assert json.loads(ne["holes"]) == ["command#0"] and ne["tier"] == "live"
    return model


def test_t3_fills_free_form_commit_message(jit):
    model = _commit_setup(jit, fill_all("Update the docs"))
    before = model.calls
    [m] = train(jit, model, ["commit src/m7.py"], fs_exec(FILES), prefix="x")
    assert calls_of(m) == [("Read", {"file_path": "src/m7.py"}), ("Bash", {"command": "git commit -m 'Update the docs'"})]
    ids = replayed_ids(m)
    assert len(ids) == 2 and ids[1].endswith("_t3")
    assert model.small == 1 and model.calls - before == 1  # the final answer only
    sub = model.sub_bodies[0]
    assert sub["tool_choice"] == {"type": "tool", "name": FILL_TOOL} and "stream" not in sub
    assert len(sub["messages"]) == 1 and "<command_0>" in sub["messages"][0]["content"]
    assert "commit src/m7.py" in sub["messages"][0]["content"] and "code 7" in sub["messages"][0]["content"]
    row = jit.store.q1("SELECT * FROM requests WHERE tier='T3'")
    assert row["input_tokens"] == 30 and row["output_tokens"] == 8 and row["n_calls"] == 1 and json.loads(row["call_ids"]) == [ids[1]]
    assert jit.stats()["T3"]["requests"] == 1


@pytest.mark.parametrize("answer", [
    {"wrong_key": "x"},                                   # missing the hole
    {"command_0": ""},                                    # empty value
    {"command_0": 42},                                    # wrong type
    {"command_0": "x" * 9000},                            # absurdly long
    {"command_0": "x", "not_this_step": True},            # the model declines
    {"type": "message", "content": [{"type": "text", "text": "garbage"}], "usage": {}},  # no tool call at all
])
def test_t3_bad_output_falls_back_to_t4(jit, answer):
    model = _commit_setup(jit, lambda body: answer)
    before = model.calls
    [m] = train(jit, model, ["commit src/m8.py"], fs_exec(FILES), prefix="x")
    assert calls_of(m)[1] == ("Bash", {"command": "git commit -m 'Minor fix'"})  # the model's own call
    assert model.small == 1 and model.calls - before == 2
    row = jit.store.q1("SELECT note FROM requests WHERE tier='T3'")
    assert "failed:" in row["note"]
    fwd = jit.store.q1("SELECT note FROM requests WHERE tier='T4' AND run_id='x-0' ORDER BY id LIMIT 1")
    assert fwd["note"].startswith("T3_failed:")


def test_t3_values_cannot_change_the_command(jit):
    # a filled value is spliced in as one shell word: it can't add a command or an operator
    model = _commit_setup(jit, fill_all("msg && git push origin main"))
    [m] = train(jit, model, ["commit src/m9.py"], fs_exec(FILES), prefix="x")
    cmd = calls_of(m)[1][1]["command"]
    assert shlex.split(cmd) == ["git", "commit", "-m", "msg && git push origin main"]


def test_t2_t3_disabled_restores_old_behaviour(tmp_path):
    jit = TreeJIT(str(tmp_path / "t.db"), t2=False, t3=False)
    model = _commit_setup(jit, fill_all("never used"))
    before = model.calls
    [m] = train(jit, model, ["commit src/m7.py"], fs_exec(FILES), prefix="x")
    assert model.small == 0 and model.calls - before == 2 and len(replayed_ids(m)) == 1
    assert "T3" not in jit.stats()
    jit.close()


# ------------------------------------------------------------------ T2


def branch_policy(task, hist, body):
    n = int(task.split()[-1][1:])
    if not hist:
        return "Read", {"file_path": f"status/s{n}"}
    if len(hist) == 1:  # depends on nothing the tree can see: a coin only the model knows
        return ("Bash", {"command": "ls healthy"}) if n % 2 else ("Read", {"file_path": "status/log"})
    return None


def pick_like_policy(body):
    prompt = body["messages"][0]["content"]
    n = int(re.search(r"<task>\ncheck s(\d+)\n</task>", prompt).group(1))
    if '<step n="2">' in prompt:
        return {"choice": 0}  # done: none of the known next steps
    want = "ls healthy" if n % 2 else "status/log"
    for line in prompt.splitlines():
        m = re.match(r"(\d+)\. (.*)", line)
        if m and want in m.group(2):
            return {"choice": int(m.group(1))}
    return {"choice": 0}


def test_t2_resolves_ambiguous_node(jit):
    files = {f"status/s{i}": "status report" for i in range(30)}
    files["status/log"] = "log"
    model = SubModel(branch_policy, pick_like_policy)
    jit.cfg.t2 = False
    train(jit, model, [f"check s{i}" for i in range(6)], fs_exec(files))
    jit.cfg.t2 = True
    assert model.small == 0
    before = model.calls
    ok, down = train(jit, model, ["check s11", "check s12"], fs_exec(files), prefix="x")
    assert calls_of(ok)[1] == ("Bash", {"command": "ls healthy"})
    assert calls_of(down)[1] == ("Read", {"file_path": "status/log"})
    assert replayed_ids(ok)[1].endswith("_t2") and replayed_ids(down)[1].endswith("_t2")
    assert model.calls - before == 2  # final answers only
    # 2 small calls, one per task. After Read(status/log) the one-edge n-gram context [Read] is
    # ambiguous too, but the root path there has only seen the model end the episode (END), so the
    # final answer goes straight to T4 without a wasted "something else" subcall.
    assert model.small == 2
    end_notes = [r["note"] for r in jit.store.q("SELECT note FROM requests WHERE run_id LIKE 'x-%' AND tier='T4'")]
    assert end_notes and all(n.startswith("end@") for n in end_notes)
    sub = model.sub_bodies[0]
    assert sub["tool_choice"]["name"] == CHOOSE_TOOL and "(used in 3 earlier successful runs)" in sub["messages"][0]["content"]
    # a T2 pick is the model's choice: it is logged as such, so it feeds the node's evidence
    assert jit.store.q1("SELECT replayed FROM steps WHERE run_id='x-0' AND idx=1")["replayed"] == 0
    # "something else" goes to the model
    model.answer = lambda body: {"choice": 0}
    before = model.calls
    [m] = train(jit, model, ["check s13"], fs_exec(files), prefix="y")
    assert calls_of(m)[1] == ("Bash", {"command": "ls healthy"}) and model.calls - before == 2


def test_t2_checkpoint_resets_budget(tmp_path):
    def policy(task, hist, body):
        return ("Bash", {"command": f"ls dir{len(hist)}"}) if len(hist) < 9 else None

    def run(**kw):
        jit = TreeJIT(str(tmp_path / f"b{len(kw)}.db"), hard_cap=20, t3=False, **kw)
        model = SubModel(policy, lambda body: {"choice": 1})
        train(jit, model, [f"walk {i}" for i in range(4)], fs_exec({}))
        before, small = model.calls, model.small
        [m] = train(jit, model, ["walk 9"], fs_exec({}), prefix="w")
        notes = [r["note"] for r in jit.store.q("SELECT note FROM requests WHERE run_id='w-0'")]
        jit.close()
        return m, model.calls - before, model.small - small, notes

    m, full, small, notes = run(t2=False)
    assert full > 1 and small == 0 and any(n.startswith("budget") for n in notes)
    m, full, small, notes = run()
    assert full == 1 and small >= 2  # checkpoints instead of full calls; only the final answer is generated
    ids = replayed_ids(m)
    assert len(ids) == 9 and sum(i.endswith("_ck") for i in ids) == small
    assert [c[1]["command"] for c in calls_of(m)] == [f"ls dir{k}" for k in range(9)]


# ------------------------------------------------------------------ proxy


def test_proxy_subcall_then_sse_replay(tmp_path):
    httpx = pytest.importorskip("httpx")
    from test_proxy import upstream_app

    from treejit.proxy import ProxyApp
    from treejit_bench.runner import parse_anthropic_sse

    async def task(client, text, rid):
        msgs, tiers = [{"role": "user", "content": text}], []
        ex = fs_exec(FILES)
        for _ in range(6):
            body = {"model": "m", "max_tokens": 64, "system": "You are a careful agent working in a repository. " * 8,
                    "tools": [{"name": "Bash", "input_schema": {"type": "object"}}, {"name": "Read", "input_schema": {"type": "object"}}],
                    "messages": msgs, "stream": True}
            r = await client.post("/v1/messages", json=body, headers={"x-api-key": "k", "X-TreeJIT-Run": rid})
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            tiers.append(r.headers["x-treejit-tier"])
            resp = parse_anthropic_sse(r.content)
            msgs.append({"role": "assistant", "content": resp["content"]})
            uses = [b for b in resp["content"] if b["type"] == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": u["id"], "content": ex(u["name"], u["input"])[0]} for u in uses]})
        await client.post("/outcome", json={"run_id": rid, "outcome": "pass"})
        return msgs, tiers

    async def main():
        jit = TreeJIT(str(tmp_path / "p.db"))
        jit.cfg.anthropic_upstream = "http://up"
        model, seen = SubModel(commit_policy, fill_all("Proxy message")), []
        up = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream_app(model, seen)), base_url="http://up")
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://tj")
        for i in range(3):
            await task(client, f"commit src/m{i}.py", f"p{i}")
        approve_all(jit)
        before = model.calls
        msgs, tiers = await task(client, "commit src/m6.py", "p9")
        assert tiers == ["T0", "T3", "T4"]
        assert calls_of(msgs)[1] == ("Bash", {"command": "git commit -m 'Proxy message'"})
        assert model.small == 1 and model.calls - before == 1
        path, headers, raw = seen[-2]
        sub = json.loads(raw)
        assert path == "/v1/messages" and headers.get("x-api-key") == "k" and "x-treejit-run" not in headers
        assert sub["tool_choice"]["name"] == FILL_TOOL and not sub.get("stream")
        await client.aclose()
        await up.aclose()
        jit.close()

    asyncio.run(main())


# ------------------------------------------------------------------ OpenAI wire format


def test_openai_subcall_body_and_parse():
    ne = NodeEdge("n1", "e1", 3, 3, 0, 3, 0.0, False, True, False, "live", 1.0, 0.8, 0.8, {"command#0": None},
                  ["command#0"], {}, {}, {"command": "git commit -m 'old'"}, [], False, 0.0, 0.0, 0.0, True)
    tpl = anti_unify([ToolCall("a", "Bash", {"command": "git commit -m 'old'"}), ToolCall("b", "Bash", {"command": "git commit -m 'x y'"})])
    assert var_slots(tpl) == ["command#0"]
    opt = Option(ne, EdgeInfo("e1", "Bash", "s", tpl, "Bash(git commit -m $0)"), {}, ["command#0"])
    sub = Subcall("fill", "n1", "r1", [opt], "holes")
    req = dialect("openai").parse_request({"model": "gpt", "max_completion_tokens": 9, "tools": [],
                                           "messages": [{"role": "user", "content": "do it"}]})
    body = build("openai", sub, req, Config(small_model="mini"))
    assert body["model"] == "mini" and body["max_completion_tokens"] == 512
    assert body["tool_choice"] == {"type": "function", "function": {"name": FILL_TOOL}}
    assert "command_0" in body["tools"][0]["function"]["parameters"]["properties"]
    resp = {"choices": [{"message": {"tool_calls": [{"id": "c", "type": "function", "function": {
        "name": FILL_TOOL, "arguments": json.dumps({"command_0": "New msg"})}}]}}]}
    o, vals, why = resolve(sub, tool_input("openai", sub, resp))
    assert why == "" and vals["command#0"].cooked == "New msg"
    assert tool_input("openai", sub, {"choices": [{"message": {"content": "hi"}}]}) is None
