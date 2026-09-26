"""R1: the OpenAI Responses API dialect (Codex-style stateless use), argv shell arguments, proxy route,
inline wrap, compaction and subcalls in that dialect."""

from __future__ import annotations

import asyncio
import itertools
import json

import pytest

from treejit import TreeJIT, compaction
from treejit.config import Config
from treejit.dialects import CUSTOM_INPUT, SSEParser
from treejit.dialects import get as dialect
from treejit.model import ToolCall
from treejit.policy import commit_reason, is_readonly
from treejit.replay import Option, Subcall
from treejit.subcalls import FILL_TOOL, build, resolve, subcall_tool, tool_input
from treejit.templates import anti_unify, call_slots, render, shape_of, shell_text, var_slots
from treejit.tree import EdgeInfo, NodeEdge

R = dialect("responses")
INSTR = "You are Codex, a coding agent working in a repository. " * 6
SHELL = {"type": "function", "name": "shell", "description": "Runs a shell command",
         "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}},
                                                         "workdir": {"type": "string"}}, "required": ["command"]}}
PATCH = {"type": "custom", "name": "apply_patch", "description": "Apply a patch", "format": {"type": "text"}}
ENV = "<environment_context>\n  <cwd>/repo</cwd>\n  <shell>bash</shell>\n</environment_context>"
_n = itertools.count()


def user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def fc(cid: str, cmd: str) -> dict:
    return {"type": "function_call", "id": f"fc_{cid}", "call_id": cid, "name": "shell",
            "arguments": json.dumps({"command": ["bash", "-lc", cmd], "workdir": "/repo"})}


def out(cid: str, text: str) -> dict:
    return {"type": "function_call_output", "call_id": cid, "output": text}


def body(items: list, **kw) -> dict:
    return {"model": "gpt-5-codex", "instructions": INSTR, "input": items, "tools": [SHELL, PATCH],
            "tool_choice": "auto", "parallel_tool_calls": False, "store": False, "stream": False, **kw}


# ------------------------------------------------------------------ a scripted Responses model


def calls_in(items: list) -> list[tuple[str, dict, str]]:
    outs = {i["call_id"]: i["output"] for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"}
    return [(i["name"], json.loads(i["arguments"]), outs.get(i["call_id"], "")) for i in items
            if isinstance(i, dict) and i.get("type") == "function_call"]


def task_of(items: list) -> str:
    texts = [p["text"] for i in items if isinstance(i, dict) and i.get("role") == "user"
             for p in (i["content"] if isinstance(i["content"], list) else [{"text": i["content"]}])]
    return next(t for t in texts if not t.startswith("<"))


class RModel:
    """policy(task, calls) -> (name, args) or None; answers Responses bodies, counts full calls."""

    def __init__(self, policy) -> None:
        self.policy, self.calls, self.bodies = policy, 0, []

    def __call__(self, b: dict) -> dict:
        self.calls += 1
        self.bodies.append(b)
        act = self.policy(task_of(b["input"]), calls_in(b["input"]))
        k = next(_n)
        usage = {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1000}, "output_tokens": 30,
                 "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 1230}
        if act is None:
            output = [{"type": "reasoning", "id": f"rs_{k:08d}", "summary": []},
                      {"type": "message", "id": f"msg_{k:08d}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]
        else:
            output = [{"type": "reasoning", "id": f"rs_{k:08d}", "summary": []},
                      {"type": "function_call", "id": f"fc_{k:08d}", "call_id": f"call_m{k:012d}xyz", "name": act[0],
                       "arguments": json.dumps(act[1]), "status": "completed"}]
        return {"id": f"resp_{k:08d}", "object": "response", "created_at": 1, "status": "completed", "model": b["model"],
                "output": output, "usage": usage}


def read_policy(task, calls):
    path = task.split()[-1]
    plan = [("shell", {"command": ["bash", "-lc", "git status --short"], "workdir": "/repo"}),
            ("shell", {"command": ["bash", "-lc", f"cat {path}"], "workdir": "/repo"})]
    return plan[len(calls)] if len(calls) < len(plan) else None


def run_codex(call, task: str, max_steps: int = 8, stream: bool = False) -> list:
    """A Codex-like loop: whole conversation in `input` every turn, store=false."""
    items = [user(ENV), user(task)]
    for _ in range(max_steps):
        resp = call(body(items, stream=stream, prompt_cache_key=f"conv-{task}"))
        new = [i for i in resp["output"]]
        items += new
        fcs = [i for i in new if i.get("type") == "function_call"]
        if not fcs:
            break
        for i in fcs:
            cmd = json.loads(i["arguments"])["command"][-1]
            items.append(out(i["call_id"], f"output of {cmd}\n" + "a line of tool output text\n" * 60))
    return items


def replayed(items: list) -> list[str]:
    return [i["call_id"] for i in items if i.get("type") == "function_call" and "_tj_" in i["call_id"]]


# ------------------------------------------------------------------ parsing


def test_parse_request_codex_shape():
    items = [{"role": "developer", "content": "Be terse."}, user(ENV), user("fix the typo in README.md"),
             {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "xx"},
             fc("call_a", "git status"), out("call_a", " M README.md"),
             {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_b", "name": "apply_patch", "input": "*** Begin Patch"},
             {"type": "custom_tool_call_output", "call_id": "call_b", "output": "Done!"}]
    req = R.parse_request(body(items, prompt_cache_key="sess-1"))
    assert req.dialect == "responses" and req.system == INSTR + "\nBe terse."
    assert [t["name"] for t in req.tools] == ["shell", "apply_patch"] and req.tools[0]["schema"] == SHELL["parameters"]
    ep = req.episode
    assert ep.task == "fix the typo in README.md" and ep.session == "sess-1" and ep.ready
    assert [(s.call.id, s.call.name) for s in ep.steps] == [("call_a", "shell"), ("call_b", "apply_patch")]
    assert ep.steps[0].call.args == {"command": ["bash", "-lc", "git status"], "workdir": "/repo"}
    assert ep.steps[0].obs.text == " M README.md" and ep.steps[1].call.args == {CUSTOM_INPUT: "*** Begin Patch"}
    assert not R.parse_request(body(items[:-1])).episode.ready     # a call without its output
    assert R.parse_request({"model": "m", "input": "hello", "tools": [SHELL]}).episode.task == "hello"


def test_stateful_requests_pass_through_untouched(jit):
    for extra in ({"previous_response_id": "resp_abc"}, {"conversation": "conv_1"}, {"conversation": {"id": "conv_1"}}):
        b = body([out("call_a", "x")], store=True, **extra)
        snapshot = json.dumps(b, sort_keys=True)
        res = jit.handle("responses", b)
        assert res.kind == "forward" and res.body is b and json.dumps(res.body, sort_keys=True) == snapshot
    rows = jit.store.q("SELECT tier, note, family, run_id FROM requests")
    assert rows and all(r["tier"] == "pass" and r["note"] == "stateful" and r["family"] is None and r["run_id"] is None
                        for r in rows)
    assert jit.store.q1("SELECT COUNT(*) n FROM runs")["n"] == 0


# ------------------------------------------------------------------ replay responses round-trip


CALLS = [ToolCall("call_tj_abcdef012345_ff0123456789", "shell", {"command": ["bash", "-lc", "ls"], "workdir": "/r"}),
         ToolCall("call_tj_abcdef012345_ff9876543210", "apply_patch", {CUSTOM_INPUT: "*** Begin Patch\n*** End Patch"})]


def test_replay_json_round_trip():
    resp = R.build_response("gpt-5-codex", CALLS)
    assert resp["object"] == "response" and resp["status"] == "completed" and resp["id"].startswith("resp_tj")
    fc_item, ctc = resp["output"]
    assert fc_item["type"] == "function_call" and fc_item["id"].startswith("fc_tj") and fc_item["call_id"] == CALLS[0].id
    assert ctc["type"] == "custom_tool_call" and ctc["input"] == CALLS[1].args[CUSTOM_INPUT]
    info = R.parse_response(resp)
    assert info.calls == CALLS and info.stop_reason == "tool_calls" and info.usage.total == 0


def test_replay_sse_round_trip_through_its_own_accumulator():
    chunks = R.build_sse("gpt-5-codex", CALLS, {})
    events = [(e, json.loads(d)) for c in chunks for e, d in SSEParser().feed(c)]
    kinds = [e for e, _ in events]
    assert kinds[:2] == ["response.created", "response.in_progress"] and kinds[-1] == "response.completed"
    assert "response.function_call_arguments.delta" in kinds and "response.function_call_arguments.done" in kinds
    assert all(d["type"] == e for e, d in events)
    assert [d["sequence_number"] for _, d in events] == list(range(len(events)))
    acc = R.stream_accumulator()
    for c in chunks:           # also byte-split: the parser must reassemble events
        for i in range(0, len(c), 7):
            acc.feed(c[i : i + 7])
    info = acc.result()
    assert info.calls == CALLS and info.stop_reason == "tool_calls"


def test_upstream_stream_usage_text_and_truncation():
    acc = R.stream_accumulator()
    evs = [{"type": "response.created", "response": {"status": "in_progress"}},
           {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant", "content": []}},
           {"type": "response.output_text.delta", "output_index": 0, "delta": "All "},
           {"type": "response.output_text.delta", "output_index": 0, "delta": "done."},
           {"type": "response.completed", "response": {"status": "completed", "output": [], "usage": {
               "input_tokens": 500, "input_tokens_details": {"cached_tokens": 400}, "output_tokens": 9}}}]
    for i, e in enumerate(evs):
        acc.feed(f"event: {e['type']}\ndata: {json.dumps(dict(e, sequence_number=i))}\n\n".encode())
    info = acc.result()
    assert info.text == "All done." and info.calls == [] and info.stop_reason == "stop"
    assert (info.usage.input_tokens, info.usage.cache_read, info.usage.output_tokens) == (100, 400, 9)
    cut = R.parse_response({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []})
    assert cut.stop_reason == "length"   # engine.TRUNCATED: never an END


def test_prepare_forward_strips_synthetic_item_ids():
    items = [user("t"), fc("call_tj_abcdef012345_ff0123456789", "ls"), out("call_tj_abcdef012345_ff0123456789", "a"),
             fc("call_model_000000000001", "pwd"), out("call_model_000000000001", "/r")]
    req = R.parse_request(body(items))
    fwd = R.prepare_forward(req)
    assert "id" not in fwd["input"][1] and fwd["input"][1]["call_id"].startswith("call_tj_")
    assert fwd["input"][3]["id"] == "fc_call_model_000000000001"      # the model's own item keeps its id
    assert req.raw["input"][1]["id"].startswith("fc_")                 # the harness body is untouched


# ------------------------------------------------------------------ argv shell arguments


def test_argv_shell_args_policy():
    cfg = Config()
    assert shell_text("command", ["bash", "-lc", "git status"]) == ("git status", ["bash", "-lc"])
    assert shell_text("command", ["ls", "-la", "my dir"]) == ("ls -la 'my dir'", "argv")
    assert shell_text("command", "ls") == ("ls", None) and shell_text("path", ["a"]) is None
    ro = lambda cmd: is_readonly("shell", {"command": cmd, "workdir": "/r"}, cfg)  # noqa: E731
    assert ro(["bash", "-lc", "git status"]) and ro(["/bin/sh", "-c", "cat a.txt | head -5"]) and ro(["ls", "-la"])
    assert not ro(["bash", "-lc", "rm -rf build"]) and not ro(["bash", "-lc", "echo x > f"]) and not ro(["rm", "-rf", "b"])
    assert not ro(["bash", "-lc", "ls", "extra"])   # $0 form: not the plain wrapper, so bash itself is the program
    assert commit_reason("shell", {"command": ["bash", "-lc", "git push origin main"]}, cfg)
    assert commit_reason("shell", {"command": ["git", "push"]}, cfg)
    assert not commit_reason("shell", {"command": ["bash", "-lc", "git status"]}, cfg)
    assert commit_reason("shell", {"command": ["python", "-c", "import os"]}, cfg)       # opaque executor


def test_argv_shell_args_templates():
    calls = [ToolCall("a", "shell", {"command": ["bash", "-lc", f"cat src/{f}.py"], "workdir": "/r"}) for f in ("a", "b")]
    assert shape_of(calls[0]) == shape_of(calls[1])
    # string commands keep their shape (edges learned before this change are unaffected)
    assert shape_of(ToolCall("x", "Bash", {"command": "cat a"})) == '["Bash",[["command",["cat"]]]]'
    assert shape_of(ToolCall("x", "shell", {"command": "cat a"})) != shape_of(calls[0])
    tpl = anti_unify(calls)
    assert tpl["args"]["command"]["wrap"] == ["bash", "-lc"] and var_slots(tpl) == ["command#0"]
    slots = call_slots(tpl, calls[1])
    assert slots["command#0"].cooked == "src/b.py"
    out = render(tpl, calls[0].args, {"command#0": slots["command#0"].__class__("src/new file.py")})
    assert out["command"] == ["bash", "-lc", "cat 'src/new file.py'"] and out["workdir"] == "/r"
    plain = [ToolCall("a", "shell", {"command": ["cat", f"src/{f}.py"]}) for f in ("a", "b")]
    tpl2 = anti_unify(plain)
    assert tpl2["args"]["command"]["wrap"] == "argv"
    new = render(tpl2, plain[0].args, {"command#0": call_slots(tpl2, plain[1])["command#0"]})
    assert new == {"command": ["cat", "src/b.py"]}
    assert call_slots(tpl, ToolCall("c", "shell", {"command": ["cat", "src/a.py"], "workdir": "/r"})) is None


# ------------------------------------------------------------------ learning and replay (inline)


def test_scripted_responses_agent_learns_and_replays(jit):
    model = RModel(read_policy)
    client = jit.wrap(model, dialect="responses")
    for i in range(3):
        run_codex(lambda b: client(b, extra_headers={"X-TreeJIT-Run": f"r{i}"}), f"inspect src/m{i}.py")
        jit.outcome(f"r{i}", "pass")
    before = model.calls
    items = run_codex(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "probe"}), "inspect src/m7.py")
    ids = replayed(items)
    assert len(ids) == 2, [r["note"] for r in jit.store.q("SELECT note FROM requests WHERE run_id='probe'")]
    got = [json.loads(i["arguments"]) for i in items if i.get("type") == "function_call"]
    assert got[1] == {"command": ["bash", "-lc", "cat src/m7.py"], "workdir": "/repo"}  # read-only: no approval needed
    assert model.calls - before == 1                                                  # the final answer only
    # replayed item ids never go upstream; the final request is recorded with its usage
    last = model.bodies[-1]["input"]
    assert all("id" not in i for i in last if i.get("type") == "function_call" and "_tj_" in i["call_id"])
    row = jit.store.q1("SELECT * FROM requests WHERE run_id='probe' AND tier='T4' ORDER BY id DESC LIMIT 1")
    assert (row["input_tokens"], row["cache_read"], row["output_tokens"]) == (200, 1000, 30)
    assert jit.store.run("probe")["ended_after"] == 2


def test_forward_records_usage_and_run(jit):
    model = RModel(read_policy)
    client = jit.wrap(model, dialect="responses")
    run_codex(client, "inspect src/a.py")
    rows = jit.store.q("SELECT * FROM requests WHERE tier='T4' ORDER BY id")
    assert len(rows) == 3 and all(r["run_id"] and r["input_tokens"] == 200 and r["cache_read"] == 1000 for r in rows)
    assert json.loads(rows[0]["call_ids"])[0].startswith("call_m")
    assert jit.store.q1("SELECT n_steps FROM runs")["n_steps"] == 2


def test_compaction_rewrites_function_call_output(tmp_path):
    jit = TreeJIT(str(tmp_path / "c.db"), compact=True, compact_keep_last=0, compact_min_chars=50, theta=0.0,
                  hard_cap=100, batch=False, t2=False, t3=False)
    model = RModel(read_policy)
    client = jit.wrap(model, dialect="responses")
    for i in range(3):
        run_codex(lambda b: client(b, extra_headers={"X-TreeJIT-Run": f"r{i}"}), f"inspect src/m{i}.py")
        jit.outcome(f"r{i}", "pass")
    items = run_codex(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "probe"}), "inspect src/m7.py")
    assert len(replayed(items)) == 2
    sent = model.bodies[-1]["input"]
    outs = [i["output"] for i in sent if i.get("type") == "function_call_output"]
    assert any(o.startswith("[treejit: replayed & verified step") for o in outs), outs
    assert all(not o.startswith("[treejit") for o in (i["output"] for i in items if i.get("type") == "function_call_output"))
    jit.close()
    # unit: string and content-part outputs, by call_id
    b = {"input": [out("c1", "long"), {"type": "function_call_output", "call_id": "c2",
                                       "output": [{"type": "input_text", "text": "long"}]}, out("c3", "keep")]}
    nb, done = compaction._rewrite("responses", b, {"c1": "D1", "c2": "D2"})
    assert done == {"c1", "c2"} and nb["input"][0]["output"] == "D1" and nb["input"][2]["output"] == "keep"
    assert nb["input"][1]["output"] == [{"type": "input_text", "text": "D2"}] and b["input"][0]["output"] == "long"


# ------------------------------------------------------------------ subcalls


def fill_subcall() -> Subcall:
    ne = NodeEdge("n1", "e1", 3, 3, 0, 3, 0.0, False, True, False, "live", 1.0, 0.8, 0.8, {"command#0": None},
                  ["command#0"], {}, {}, {"command": "git commit -m 'old'"}, [], False, 0.0, 0.0, 0.0, True)
    tpl = anti_unify([ToolCall("a", "Bash", {"command": "git commit -m 'old'"}),
                      ToolCall("b", "Bash", {"command": "git commit -m 'x y'"})])
    opt = Option(ne, EdgeInfo("e1", "Bash", "s", tpl, "Bash(git commit -m $0)"), {}, ["command#0"])
    return Subcall("fill", "n1", "r1", [opt], "holes")


def test_responses_subcall_body_and_parse():
    sub = fill_subcall()
    req = R.parse_request(body([user("do it")]))
    b = build("responses", sub, req, Config(small_model="mini"))
    assert b["model"] == "mini" and b["tool_choice"] == {"type": "function", "name": FILL_TOOL}
    assert b["tools"][0]["name"] == FILL_TOOL and "command_0" in b["tools"][0]["parameters"]["properties"]
    assert subcall_tool(b) == FILL_TOOL and b["store"] is False and "stream" not in b
    resp = {"status": "completed", "output": [{"type": "function_call", "call_id": "c", "name": FILL_TOOL,
                                               "arguments": json.dumps({"command_0": "New msg"})}],
            "usage": {"input_tokens": 40, "output_tokens": 5}}
    o, vals, why = resolve(sub, tool_input("responses", sub, resp))
    assert why == "" and vals["command#0"].cooked == "New msg"
    assert tool_input("responses", sub, {"output": [{"type": "message", "content": []}]}) is None


# ------------------------------------------------------------------ proxy route


def test_proxy_route_learns_replays_and_passes_stateful_through(tmp_path):
    httpx = pytest.importorskip("httpx")
    from treejit.proxy import ProxyApp

    seen: list = []
    model = RModel(read_policy)

    async def upstream(scope, receive, send):
        raw = b""
        while True:
            m = await receive()
            raw += m.get("body", b"")
            if not m.get("more_body"):
                break
        req = json.loads(raw)
        seen.append((scope["path"], dict((k.decode(), v.decode()) for k, v in scope["headers"]), req))
        resp = model(req) if not req.get("previous_response_id") else {
            "id": "resp_next", "object": "response", "status": "completed", "output": [], "usage": {}}
        if req.get("stream"):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            evs = [{"type": "response.created", "response": dict(resp, status="in_progress", output=[])}]
            for i, item in enumerate(resp["output"]):
                evs += [{"type": "response.output_item.added", "output_index": i, "item": item},
                        {"type": "response.output_item.done", "output_index": i, "item": item}]
            evs.append({"type": "response.completed", "response": resp})
            for n, e in enumerate(evs):
                data = f"event: {e['type']}\ndata: {json.dumps(dict(e, sequence_number=n))}\n\n".encode()
                await send({"type": "http.response.body", "body": data, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        else:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": json.dumps(resp).encode()})

    def parse_sse(raw: bytes) -> dict:
        acc = R.stream_accumulator()
        acc.feed(raw)
        final = acc._resp
        assert final is not None
        return final

    async def task(client, text, rid):
        items, tiers = [user(ENV), user(text)], []
        for _ in range(6):
            r = await client.post("/v1/responses", json=body(items, stream=True),
                                  headers={"authorization": "Bearer k", "X-TreeJIT-Run": rid})
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            tiers.append(r.headers["x-treejit-tier"])
            resp = parse_sse(r.content)
            items += resp["output"]
            fcs = [i for i in resp["output"] if i.get("type") == "function_call"]
            if not fcs:
                break
            items += [out(i["call_id"], "file contents\n" * 5) for i in fcs]
        await client.post("/outcome", json={"run_id": rid, "outcome": "pass"})
        return items, tiers

    async def main():
        jit = TreeJIT(str(tmp_path / "p.db"))
        jit.cfg.openai_upstream = "http://up"
        up = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream), base_url="http://up")
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://tj")
        for i in range(3):
            await task(client, f"inspect src/m{i}.py", f"p{i}")
        before = model.calls
        items, tiers = await task(client, "inspect src/m5.py", "p9")
        # both steps replay (batched into one response: independent proven edges), then the final answer
        assert tiers == ["T0", "T4"] and model.calls - before == 1 and len(replayed(items)) == 2
        path, headers, sent = seen[-1]
        assert path == "/v1/responses" and headers.get("authorization") == "Bearer k" and "x-treejit-run" not in headers
        # stateful: forwarded as sent, never learned from
        n = len(seen)
        stateful = body([out("call_zz", "x")], previous_response_id="resp_prev", store=True)
        r = await client.post("/v1/responses", json=stateful, headers={"authorization": "Bearer k"})
        assert r.status_code == 200 and "x-treejit-tier" not in r.headers and len(seen) == n + 1
        assert seen[-1][2] == stateful
        assert jit.store.q1("SELECT tier, note FROM requests ORDER BY id DESC LIMIT 1")["note"] == "stateful"
        await client.aclose()
        await up.aclose()
        jit.close()

    asyncio.run(main())


# ------------------------------------------------------------------ the OpenAI SDK, inline


def test_openai_sdk_responses_create_inline(jit):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    model = RModel(read_policy)

    def handler(request):
        req = json.loads(request.content)
        assert request.url.path == "/v1/responses" and "x-treejit-run" not in request.headers
        return httpx.Response(200, json=model(req))

    sdk = openai.OpenAI(api_key="k", base_url="http://up/v1", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client = jit.wrap(sdk)
    assert client.responses.create is not sdk.responses.create and callable(client.chat.completions.create)

    for i in range(3):
        run_codex(lambda b: client.responses.create(**b, extra_headers={"X-TreeJIT-Run": f"s{i}"}).model_dump(mode="json"),
                  f"inspect src/m{i}.py")
        jit.outcome(f"s{i}", "pass")
    b = body([user(ENV), user("inspect src/m8.py")])
    resp = client.responses.create(**b, extra_headers={"X-TreeJIT-Run": "sdk"})
    assert isinstance(resp, openai.types.responses.Response) and resp.output[0].call_id.startswith("call_tj_")
    # streamed replay: SDK event models, no upstream call
    before = model.calls
    stream = client.responses.create(**dict(b, stream=True), extra_headers={"X-TreeJIT-Run": "sdk2"})
    evs = list(stream)
    assert model.calls == before and evs[0].type == "response.created" and evs[-1].type == "response.completed"
    assert evs[-1].response.output[0].name == "shell"


def test_inline_streamed_forward_records_usage_and_calls(jit):
    model = RModel(read_policy)

    def upstream(b):
        resp = model(b)
        assert b["stream"] is True
        evs = [{"type": "response.created", "response": dict(resp, status="in_progress", output=[])}]
        for i, item in enumerate(resp["output"]):
            evs += [{"type": "response.output_item.added", "output_index": i, "item": dict(item, arguments="")}]
            if item["type"] == "function_call":
                evs.append({"type": "response.function_call_arguments.delta", "output_index": i, "delta": item["arguments"]})
            evs.append({"type": "response.output_item.done", "output_index": i, "item": item})
        evs.append({"type": "response.completed", "response": resp})
        return iter([dict(e, sequence_number=n) for n, e in enumerate(evs)])

    client = jit.wrap(upstream, dialect="responses")
    stream = client(body([user(ENV), user("inspect src/q.py")], stream=True))
    evs = list(stream)
    assert evs[-1]["type"] == "response.completed"
    row = jit.store.q1("SELECT * FROM requests WHERE tier='T4'")
    assert row["status"] == 200 and (row["input_tokens"], row["cache_read"], row["output_tokens"]) == (200, 1000, 30)
    assert json.loads(row["call_ids"])[0].startswith("call_m") and row["n_calls"] == 1
