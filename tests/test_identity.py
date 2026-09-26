"""T2: run identity. Runs never merge conversations, never grow after their outcome, and follow
one episode across interrupts; harness session ids give stable ids from the first request."""

from __future__ import annotations

import json

from conftest import SYSTEM, TOOLS, text_msg, tool_msg

from treejit.model import USER, weak_call_id

OTOOLS = [{"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def _oai(content=None, calls=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"id": "c", "object": "chat.completion", "model": "m", "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}]}


def _runs(jit):
    return jit.store.q("SELECT id, task, n_steps, outcome, inherited FROM runs ORDER BY created, id")


def test_weak_call_ids():
    assert weak_call_id("call_0") and weak_call_id("call_12") and weak_call_id("") and weak_call_id("toolu_01")
    assert weak_call_id("functions.Bash:0")
    assert not weak_call_id("toolu_01A09q90qw90lq917835lq9") and not weak_call_id("call_Xy12abCD34efGH56")
    assert not weak_call_id("toolu_m00000001")


def test_colliding_call_ids_give_separate_runs(jit):
    """A local server numbering tool calls (`call_0`, `call_1`) for three conversations with the same task:
    the first passes; the others must neither extend it nor inherit its pass."""
    scripts = [["ls", "cat README.md"], ["ls", "rm -rf build"], ["ls", "rm -rf build"]]
    outs = [["README.md", "# readme"], ["README.md build/", "removed"], ["src/ build/", "removed"]]

    def run(k):
        def model(body):
            n = sum(1 for m in body["messages"] if m["role"] == "tool")
            if n >= 2:
                return _oai("done")
            return _oai(None, [{"id": f"call_{n}", "type": "function",
                                "function": {"name": "Bash", "arguments": json.dumps({"command": scripts[k][n]})}}])

        client = jit.wrap(model, dialect="openai")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "task A: inspect the repo"}]
        for _ in range(4):
            r = client({"model": "m", "messages": msgs, "tools": OTOOLS})
            msg = r["choices"][0]["message"]
            msgs.append(msg)
            if not msg.get("tool_calls"):
                break
            for tc in msg["tool_calls"]:
                n = sum(1 for m in msgs if m["role"] == "tool")
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": outs[k][n]})

    run(0)
    [first] = _runs(jit)
    jit.outcome(first["id"], "pass")
    run(1)
    run(2)
    runs = _runs(jit)
    assert len(runs) == 3 and all(r["n_steps"] == 2 for r in runs)
    steps = {r["id"]: [json.loads(s["args"])["command"] for s in jit.store.steps(r["id"])] for r in runs}
    assert steps[first["id"]] == ["ls", "cat README.md"]                   # the passing run was not extended
    assert [r["outcome"] for r in runs] == ["pass", None, None]            # ...and nobody inherited its pass
    assert sorted(steps.values()) == sorted([["ls", "cat README.md"], ["ls", "rm -rf build"], ["ls", "rm -rf build"]])
    # the first request of each conversation waited for its first observation, then joined its run
    assert not jit.store.q("SELECT 1 FROM requests WHERE family IS NOT NULL AND run_id IS NULL")
    assert [r["n"] for r in jit.store.q("SELECT COUNT(*) n FROM requests GROUP BY run_id ORDER BY MIN(id)")] == [3, 3, 3]


def test_run_with_an_outcome_is_forked_not_extended(jit):
    """A Stop hook reports after every agent turn; the user then confirms and the agent goes on."""
    def model(body):
        msgs = body["messages"]
        names = [b["name"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
        if not names:
            return tool_msg("Read", {"file_path": "src/app.py"})
        if isinstance(msgs[-1]["content"], list):
            return text_msg("Found the bug. Should I fix it?") if len(names) == 1 else text_msg("Fixed.")
        return tool_msg("Bash", {"command": "sed -i s/a/b/ src/app.py"})

    client = jit.wrap(model, dialect="anthropic")
    call = lambda b: client(b, extra_headers={"X-TreeJIT-Run": "sess"})  # noqa: E731
    msgs = [{"role": "user", "content": "fix the bug in src/app.py"}]
    for reply in ("yes", None):
        for _ in range(4):
            r = call({"model": "m", "max_tokens": 50, "system": SYSTEM, "tools": TOOLS, "messages": msgs})
            msgs.append({"role": "assistant", "content": r["content"]})
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
        jit.outcome("sess", "pass")
        if reply:
            msgs.append({"role": "user", "content": reply})
    runs = _runs(jit)
    assert [(r["id"], r["n_steps"], r["outcome"], r["inherited"]) for r in runs] == [
        ("sess", 1, "pass", 0), ("sess.2", 3, "pass", 1)]
    assert [s["tool"] for s in jit.store.steps("sess.2")] == ["Read", f"{USER}:yes", "Bash"]
    # the inherited prefix is context, not a second passing run for the Read edge
    read = jit.store.q1("SELECT ne.pass_runs FROM node_edges ne JOIN edges e ON e.id=ne.edge "
                        "JOIN nodes n ON n.id=ne.node WHERE e.tool='Read' AND n.kind='r' AND n.depth=0")
    assert read["pass_runs"] == 1


def test_interrupted_episode_is_one_run(jit):
    k = {"n": 0}

    def model(body):
        k["n"] += 1
        return tool_msg("Bash", {"command": f"echo {k['n']}"})

    client = jit.wrap(model, dialect="anthropic")
    msgs = [{"role": "user", "content": "prompt 1: add a flag"}]

    def step():
        r = client({"model": "m", "max_tokens": 10, "system": SYSTEM, "tools": TOOLS, "messages": msgs})
        msgs.append({"role": "assistant", "content": r["content"]})
        return r["content"][0]["id"]

    for _ in range(3):
        cid = step()
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": "ok"}]})
    cid = step()
    msgs.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": cid, "content": "The user doesn't want to proceed with this tool use.", "is_error": True},
        {"type": "text", "text": "[Request interrupted by user for tool use]"}]})
    msgs.append({"role": "user", "content": "actually use --verbose"})
    for _ in range(2):
        cid = step()
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": "ok"}]})
    step()
    runs = _runs(jit)
    assert len(runs) == 1 and runs[0]["task"] == "prompt 1: add a flag"
    tools = [s["tool"] for s in jit.store.steps(runs[0]["id"])]
    assert tools == ["Bash"] * 4 + [f"{USER}:steer"] + ["Bash"] * 2


def _session_conv(jit, prompts, meta=None, headers=None, text="Done."):
    """One conversation: each prompt gets one Bash call and a final answer. Returns the per-request run ids."""
    calls = {"n": 0}

    def model(body):
        last = body["messages"][-1]["content"]
        calls["n"] += 1
        return tool_msg("Bash", {"command": "ls"}) if isinstance(last, str) else text_msg(text)

    client = jit.wrap(model, dialect="anthropic")
    msgs: list = []
    for p in prompts:
        msgs.append({"role": "user", "content": p})
        for _ in range(3):
            body = {"model": "m", "max_tokens": 10, "system": SYSTEM, "tools": TOOLS, "messages": msgs}
            if meta:
                body["metadata"] = meta
            r = client(body, extra_headers=headers or {})
            msgs.append({"role": "assistant", "content": r["content"]})
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
    return msgs


def test_session_id_and_episode_index_give_stable_ids(jit):
    meta = {"user_id": "user_ab_account_0000aaaa-0000-0000-0000-000000000000_session_11112222-3333-4444-5555-666677778888"}
    _session_conv(jit, ["check the repo", "check the repo"], meta=meta)
    rows = jit.store.q("SELECT run_id FROM requests ORDER BY id")
    ids = [r["run_id"] for r in rows]
    # same session, same task text twice: two episodes (index 0 and 1), two runs; each known from its first request
    assert None not in ids and ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    # the same conversation replayed from scratch (a new process, say) maps to the same ids
    first = list(ids)
    jit.store.x("DELETE FROM requests")
    jit.store.x("DELETE FROM steps")
    jit.store.x("DELETE FROM runs")
    _session_conv(jit, ["check the repo", "check the repo"], meta=meta)
    assert [r["run_id"] for r in jit.store.q("SELECT run_id FROM requests ORDER BY id")][::2] == first[::2]
    # another session with the same prompts: other runs
    meta2 = {"user_id": "user_ab_account_0000aaaa-0000-0000-0000-000000000000_session_99992222-3333-4444-5555-666677778888"}
    _session_conv(jit, ["check the repo"], meta=meta2)
    assert jit.store.q1("SELECT run_id FROM requests ORDER BY id DESC LIMIT 1")["run_id"] not in first


def test_session_header_and_openai_prompt_cache_key(jit):
    _session_conv(jit, ["look around"], headers={"X-Claude-Code-Session-Id": "abc-session-1"})
    a = {r["run_id"] for r in jit.store.q("SELECT run_id FROM requests")}
    assert len(a) == 1 and None not in a

    def model(body):
        n = sum(1 for m in body["messages"] if m["role"] == "tool")
        return _oai(None, [{"id": "call_0", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}]) \
            if n == 0 else _oai("done")

    client = jit.wrap(model, dialect="openai")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "look around"}]
    r = client({"model": "m", "messages": msgs, "tools": OTOOLS, "prompt_cache_key": "codex-conv-7"})
    msgs += [r["choices"][0]["message"], {"role": "tool", "tool_call_id": "call_0", "content": "x"}]
    client({"model": "m", "messages": msgs, "tools": OTOOLS, "prompt_cache_key": "codex-conv-7"})
    b = [r["run_id"] for r in jit.store.q("SELECT run_id FROM requests WHERE dialect='openai' ORDER BY id")]
    assert b[0] is not None and b[0] == b[1] and b[0] not in a   # a session key needs no call-id salt


def test_header_run_keeps_its_name(jit):
    _session_conv(jit, ["first task"], headers={"X-TreeJIT-Run": "h1"})
    assert [r["id"] for r in _runs(jit)] == ["h1"]
    jit.outcome("h1", "pass")
    _session_conv(jit, ["first task"], headers={"X-TreeJIT-Run": "h1"})        # the same header, a new conversation
    runs = _runs(jit)
    assert [r["id"] for r in runs] == ["h1", "h1.2"] and runs[0]["n_steps"] == 1 and runs[1]["inherited"] == 0
