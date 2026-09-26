"""T1: inline-mode streaming goes through the engine (replay, record, END) and the wrapped
client keeps the real client's surface. Runs without the SDKs; SDK-typed checks skip then."""

from __future__ import annotations

import json

import pytest
from conftest import SYSTEM, TOOLS, Model

from treejit import TreeJIT
from treejit.dialects import get as dialect
from treejit.inline import ReplayStream, TeeStream, _MessageStream, _StreamManager, _to_dict
from treejit.model import REPLAY_MARK

PLAN = [("Bash", {"command": "ls"}), ("Bash", {"command": "cat README.md"})]


def policy(task, hist, body):
    return PLAN[len(hist)] if len(hist) < len(PLAN) else None


def execute(name, args):
    return ("README.md\nsrc" if args["command"] == "ls" else "# readme\n" * 5), False


def anthropic_events(msg: dict):
    """An Anthropic message as raw stream events (dicts, as the SDK's RawMessageStreamEvent.model_dump())."""
    yield {"type": "message_start", "message": dict(msg, content=[], stop_reason=None)}
    for i, b in enumerate(msg["content"]):
        if b["type"] == "tool_use":
            yield {"type": "content_block_start", "index": i, "content_block": dict(b, input={})}
            yield {"type": "content_block_delta", "index": i,
                   "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}}
        else:
            yield {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}
        yield {"type": "content_block_stop", "index": i}
    yield {"type": "message_delta", "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
           "usage": {"output_tokens": msg["usage"]["output_tokens"]}}
    yield {"type": "message_stop"}


def openai_chunks(msg: dict):
    """The Anthropic-shaped scripted message as OpenAI chat.completion.chunk dicts (with a usage chunk)."""
    base = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    yield dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}])
    calls = [b for b in msg["content"] if b["type"] == "tool_use"]
    for i, b in enumerate(calls):
        yield dict(base, choices=[{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": i, "id": b["id"].replace("toolu", "call"), "type": "function",
             "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}]}}])
    for b in msg["content"]:
        if b["type"] == "text":
            yield dict(base, choices=[{"index": 0, "delta": {"content": b["text"]}, "finish_reason": None}])
    yield dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}])
    yield dict(base, choices=[], usage={"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107})


class FakeMessages:
    """SDK-shaped `client.messages`: create() streams dict events when stream=True."""

    batches = "real-batches"

    def __init__(self, model: Model) -> None:
        self.model = model
        self.upstream = 0
        self.headers_seen: list[dict] = []
        self.bodies: list[dict] = []

    def create(self, extra_headers=None, **body):
        self.upstream += 1
        self.headers_seen.append(dict(extra_headers or {}))
        self.bodies.append(body)
        msg = self.model(body)
        return anthropic_events(msg) if body.get("stream") else msg

    def count_tokens(self, **body):
        return {"input_tokens": 42}

    def stream(self, **body):
        return "real-stream-manager"


class FakeAnthropic:
    def __init__(self, model: Model | None = None) -> None:
        self.messages = FakeMessages(model or Model(policy))
        self.api_key = "k"


class FakeCompletions:
    def __init__(self, model: Model) -> None:
        self.model, self.upstream, self.headers_seen = model, 0, []

    def create(self, extra_headers=None, **body):
        self.upstream += 1
        self.headers_seen.append(dict(extra_headers or {}))
        anth = self.model(_openai_to_anthropic(body))
        if body.get("stream"):
            return openai_chunks(anth)
        return {"id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "finish_reason": "tool_calls" if anth["stop_reason"] == "tool_use" else "stop",
                             "message": {"role": "assistant", "content": None, "tool_calls": [
                                 {"id": b["id"].replace("toolu", "call"), "type": "function",
                                  "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                                 for b in anth["content"] if b["type"] == "tool_use"]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}}


class FakeOpenAI:
    def __init__(self, model: Model) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(model)
        self.chat.other = "chat-other"


def _openai_to_anthropic(body: dict) -> dict:
    """Map an OpenAI request into the Anthropic shape conftest.Model reads (task + history)."""
    msgs, uses = [], {}
    for m in body["messages"]:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = [{"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                       "input": json.loads(tc["function"]["arguments"])} for tc in m.get("tool_calls") or []]
            uses.update({b["id"]: b for b in blocks})
            msgs.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                                                      "content": m["content"]}]})
    return dict(body, messages=msgs)


def accumulate(d: str, events) -> tuple[list, str]:
    acc = dialect(d).stream_accumulator()
    for e in events:
        acc.on_event(None, _to_dict(e))
    info = acc.result()
    return [(c.id, c.name, c.args) for c in info.calls], info.text


def run_stream(create, task: str, rid: str | None = None, max_steps: int = 6) -> list[dict]:
    """Anthropic agent loop over streamed responses."""
    msgs = [{"role": "user", "content": task}]
    kw = {"extra_headers": {"X-TreeJIT-Run": rid}} if rid else {}
    for _ in range(max_steps):
        with create(model="m", max_tokens=100, system=SYSTEM, tools=TOOLS, messages=msgs, stream=True, **kw) as s:
            calls, text = accumulate("anthropic", s)
        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]
        msgs.append({"role": "assistant", "content": content})
        if not calls:
            break
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": execute(n, a)[0]}
                                                 for i, n, a in calls]})
    return msgs


def run_json(create, task: str, rid: str) -> None:
    msgs = [{"role": "user", "content": task}]
    for _ in range(6):
        out = _to_dict(create(model="m", max_tokens=100, system=SYSTEM, tools=TOOLS, messages=msgs,
                              extra_headers={"X-TreeJIT-Run": rid}))
        msgs.append({"role": "assistant", "content": out["content"]})
        uses = [b for b in out["content"] if b["type"] == "tool_use"]
        if not uses:
            return
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"],
                                                  "content": execute(u["name"], u["input"])[0]} for u in uses]})


def trained(jit: TreeJIT, n: int = 3) -> tuple[FakeAnthropic, object]:
    fake = FakeAnthropic()
    client = jit.wrap(fake)
    for i in range(n):
        run_json(client.messages.create, f"inspect repo {i}", f"train{i}")
        jit.outcome(f"train{i}", "pass")
    return fake, client


def first_request(task: str) -> dict:
    return dict(model="m", max_tokens=100, system=SYSTEM, tools=TOOLS, messages=[{"role": "user", "content": task}])


# ------------------------------------------------------------------ replay


def test_streamed_replay_makes_no_upstream_call_and_events_equal_the_replay(jit):
    fake, client = trained(jit)
    n0 = fake.messages.upstream
    s = client.messages.create(**first_request("inspect repo 9"), stream=True, extra_headers={"X-TreeJIT-Run": "p"})
    assert isinstance(s, ReplayStream) and s.response is None
    assert fake.messages.upstream == n0
    calls, _ = accumulate("anthropic", s)
    row = jit.store.q1("SELECT * FROM requests WHERE run_id='p' ORDER BY id DESC")
    assert row["tier"] in ("T0", "T1") and [c[0] for c in calls] == json.loads(row["call_ids"])
    assert [(n, a) for _, n, a in calls] == PLAN[: len(calls)] and all(REPLAY_MARK in c[0] for c in calls)
    assert list(s) == []  # iterable once, like the SDK's Stream


def test_streamed_run_matches_json_run(jit):
    fake, client = trained(jit)
    n0 = fake.messages.upstream
    msgs = run_stream(client.messages.create, "inspect repo 10", "probe-sse")
    assert fake.messages.upstream - n0 == 1  # the final answer only
    ids = [b["id"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    assert len(ids) == 2 and all(REPLAY_MARK in i for i in ids)
    r = jit.store.run("probe-sse")
    assert r is not None and jit.store.n_steps("probe-sse") == 2 and r["ended_after"] == 2
    assert all(RUN not in {k.lower() for k in h} for h in fake.messages.headers_seen)


RUN = "x-treejit-run"


def test_replay_stream_context_manager_and_close():
    s = ReplayStream([{"type": "message_start"}, {"type": "message_stop"}])
    with s as it:
        assert next(iter(it)) == {"type": "message_start"}
    assert list(s) == []


# ------------------------------------------------------------------ forward


def test_forwarded_stream_records_usage_steps_and_end(jit):
    fake = FakeAnthropic()
    client = jit.wrap(fake)
    msgs = run_stream(client.messages.create, "inspect repo 1", "fwd")
    assert fake.messages.upstream == 3 and all(b["stream"] for b in fake.messages.bodies)
    rows = jit.store.q("SELECT * FROM requests WHERE run_id='fwd' ORDER BY id")
    assert len(rows) == 3 and all(r["status"] == 200 and r["input_tokens"] == 100 and r["output_tokens"] > 0 for r in rows)
    assert [r["n_calls"] for r in rows] == [1, 1, 0]
    assert jit.store.n_steps("fwd") == 2 and jit.store.run("fwd")["ended_after"] == 2
    assert msgs[-1]["content"] == [{"type": "text", "text": "done"}]
    assert all(RUN not in {k.lower() for k in h} for h in fake.messages.headers_seen)


def test_forwarded_stream_yields_upstream_events_unchanged(jit):
    fake = FakeAnthropic()
    client = jit.wrap(fake)
    s = client.messages.create(**first_request("inspect repo 1"), stream=True)
    assert isinstance(s, TeeStream)
    got = list(s)
    expect = list(anthropic_events(Model(policy)(first_request("inspect repo 1"))))
    assert [e["type"] for e in got] == [e["type"] for e in expect]
    assert got[2]["delta"] == expect[2]["delta"]
    # no run header: the run id is derived from the first tool call once the stream completes
    row = jit.store.q1("SELECT * FROM requests ORDER BY id DESC")
    assert row["status"] == 200 and row["run_id"] and row["run_id"] == s.run_id and jit.store.run(row["run_id"])


def test_early_close_is_499_without_end(jit):
    fake = FakeAnthropic(Model(lambda t, h, b: None))  # a final answer: END if completed
    client = jit.wrap(fake)
    body = first_request("inspect repo 1")
    body["messages"] += [{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {"command": "ls"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "x"}]}]
    s = client.messages.create(**body, stream=True, extra_headers={"X-TreeJIT-Run": "cut"})
    with s:
        next(s)
    row = jit.store.q1("SELECT * FROM requests WHERE run_id='cut' ORDER BY id DESC")
    assert row["status"] == 499 and jit.store.run("cut")["ended_after"] is None
    s.close()  # idempotent: completes once
    assert jit.store.q1("SELECT COUNT(*) n FROM requests WHERE run_id='cut'")["n"] == 1

    s = client.messages.create(**body, stream=True, extra_headers={"X-TreeJIT-Run": "whole"})
    list(s)
    assert jit.store.run("whole")["ended_after"] == 1


def test_close_after_stop_reason_counts_as_complete(jit):
    fake = FakeAnthropic()
    client = jit.wrap(fake)
    with client.messages.create(**first_request("inspect repo 1"), stream=True, extra_headers={"X-TreeJIT-Run": "r"}) as s:
        for e in s:
            if _to_dict(e)["type"] == "message_delta":
                break  # the caller skips message_stop
    row = jit.store.q1("SELECT * FROM requests WHERE run_id='r'")
    assert row["status"] == 200 and row["n_calls"] == 1


def test_upstream_error_mid_stream_is_recorded_and_reraised(jit):
    class Boom(Exception):
        status_code = 529

    def fn(body):
        def gen():
            yield {"type": "message_start", "message": {"usage": {"input_tokens": 3}}}
            raise Boom()
        return gen()

    client = jit.wrap(fn, dialect="anthropic")
    s = client(first_request("x"), stream=True, extra_headers={"X-TreeJIT-Run": "e"})
    with pytest.raises(Boom):
        list(s)
    assert jit.store.q1("SELECT status FROM requests WHERE run_id='e'")["status"] == 529


def test_plain_callable_streaming_raw_sse_bytes(jit):
    """A callable may stream raw SSE bytes: forwarded unchanged and accounted; replays are dicts."""
    model = Model(policy)

    def fn(body):
        msg = model(body)
        if not body.get("stream"):
            return msg
        return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in anthropic_events(msg)]

    client = jit.wrap(fn, dialect="anthropic")
    s = client(first_request("inspect repo 1"), stream=True, extra_headers={"X-TreeJIT-Run": "b"})
    chunks = list(s)
    assert all(isinstance(c, bytes) for c in chunks)
    row = jit.store.q1("SELECT * FROM requests WHERE run_id='b'")
    assert row["status"] == 200 and row["n_calls"] == 1 and row["input_tokens"] == 100


# ------------------------------------------------------------------ OpenAI


def test_openai_stream_replay_and_forward(jit):
    fake = FakeOpenAI(Model(policy))
    client = jit.wrap(fake)
    assert client.chat.other == "chat-other"  # the real chat namespace behind the override

    def run(task, rid, stream):
        msgs = [{"role": "user", "content": task}]
        for _ in range(6):
            out = client.chat.completions.create(model="m", tools=[{"type": "function", "function": {
                "name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}],
                messages=[{"role": "system", "content": SYSTEM}] + msgs, stream=stream,
                stream_options={"include_usage": True} if stream else None, extra_headers={"X-TreeJIT-Run": rid})
            calls = accumulate("openai", out)[0] if stream else [
                (tc["id"], tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                for tc in _to_dict(out)["choices"][0]["message"].get("tool_calls") or []]
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, n, a in calls]})
            if not calls:
                return msgs
            msgs += [{"role": "tool", "tool_call_id": i, "content": execute(n, a)[0]} for i, n, a in calls]
        return msgs

    run("inspect repo 0", "o-fwd", True)
    assert jit.store.n_steps("o-fwd") == 2 and jit.store.run("o-fwd")["ended_after"] == 2
    assert jit.store.q1("SELECT SUM(input_tokens) t FROM requests WHERE run_id='o-fwd'")["t"] == 300
    jit.outcome("o-fwd", "pass")
    for i in (1, 2):
        run(f"inspect repo {i}", f"o{i}", False)
        jit.outcome(f"o{i}", "pass")
    n0 = fake.chat.completions.upstream
    msgs = run("inspect repo 9", "o-probe", True)
    assert fake.chat.completions.upstream - n0 == 1
    ids = [tc["id"] for m in msgs if m["role"] == "assistant" for tc in m["tool_calls"]]
    assert len(ids) == 2 and all(REPLAY_MARK in i for i in ids)
    assert all(RUN not in {k.lower() for k in h} for h in fake.chat.completions.headers_seen)


# ------------------------------------------------------------------ T2/T3 subcalls stay non-streaming


def test_t3_subcall_is_non_streaming_and_replay_streams(jit):
    from test_tiers import FILES, _commit_setup, fill_all, fs_exec

    model = _commit_setup(jit, fill_all("Update the docs"))
    ex = fs_exec(FILES)
    seen: list[bool] = []

    def fn(body):
        seen.append(bool(body.get("stream")))
        msg = model(body)
        return anthropic_events(msg) if body.get("stream") else msg

    client = jit.wrap(fn, dialect="anthropic")
    msgs = [{"role": "user", "content": "commit src/m7.py"}]
    for _ in range(4):
        s = client(model="m", max_tokens=100, system=SYSTEM, tools=TOOLS, messages=msgs, stream=True,
                   extra_headers={"X-TreeJIT-Run": "sub"})
        calls, text = accumulate("anthropic", s)
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]
                     or [{"type": "text", "text": text}]})
        if not calls:
            break
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": ex(n, a)[0]}
                                                 for i, n, a in calls]})
    cmds = [b["input"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    assert cmds[-1] == {"command": "git commit -m 'Update the docs'"}
    assert model.small == 1 and "stream" not in model.sub_bodies[0]
    assert seen.count(False) == 1 and seen.count(True) == 1  # the subcall, then the final answer


# ------------------------------------------------------------------ surface


def test_wrapped_namespace_falls_back_to_the_real_client(jit):
    fake = FakeAnthropic()
    client = jit.wrap(fake)
    assert client.messages.count_tokens(model="m", messages=[]) == {"input_tokens": 42}
    assert client.messages.batches == "real-batches" and client.api_key == "k"
    assert client.messages.stream is not fake.messages.stream  # routed through treejit
    with pytest.raises(AttributeError):
        client.messages.nope


def test_run_id_from_wrap_never_goes_upstream(jit):
    fake = FakeAnthropic()
    client = jit.wrap(fake, run_id="fixed")
    list(client.messages.create(**first_request("inspect repo 1"), stream=True, extra_headers={"X-Other": "1"}))
    client.messages.create(**first_request("inspect repo 1"))
    assert fake.messages.headers_seen == [{"X-Other": "1"}, {}]
    assert {r["run_id"] for r in jit.store.q("SELECT run_id FROM requests")} == {"fixed"}


def test_async_clients_are_rejected(jit):
    class AsyncAnthropic(FakeAnthropic):
        pass

    with pytest.raises(TypeError, match="synchronous"):
        jit.wrap(AsyncAnthropic())


def test_messages_stream_fallback_shim():
    events = list(anthropic_events({"id": "m", "type": "message", "role": "assistant", "model": "m",
                                    "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"},
                                                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
                                    "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}}))
    with _StreamManager(lambda: ReplayStream(list(events))) as s:
        assert isinstance(s, _MessageStream)
        assert "".join(s.text_stream) == "hi there"
        final = s.get_final_message()
    assert final["content"][2] == {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
    assert final["stop_reason"] == "tool_use" and final["usage"]["output_tokens"] == 3
    with _StreamManager(lambda: ReplayStream(list(events))) as s:
        assert len(list(s)) == len(events) and s.get_final_text() == "hi there"


# ------------------------------------------------------------------ real SDK types (skipped without the SDKs)


def test_sdk_anthropic_event_types_and_messages_stream(jit):
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types import RawContentBlockDeltaEvent, RawMessageStartEvent

    fake, client = trained(jit)
    n0 = fake.messages.upstream
    s = client.messages.create(**first_request("inspect repo 9"), stream=True)
    events = list(s)
    assert isinstance(events[0], RawMessageStartEvent) and isinstance(events[2], RawContentBlockDeltaEvent)

    # messages.stream(): the SDK's own MessageStream over the replay, then over a forward
    with client.messages.stream(**first_request("inspect repo 9")) as ms:
        final = ms.get_final_message()
    assert isinstance(final, anthropic.types.Message) and final.content[0].type == "tool_use"
    assert REPLAY_MARK in final.content[0].id and fake.messages.upstream == n0
    body = first_request("inspect repo 9")
    body["messages"] += [{"role": "assistant", "content": [_to_dict(b) for b in final.content]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": b.id, "content": "x"}
                                                      for b in final.content]}]
    with client.messages.stream(**body, extra_headers={"X-TreeJIT-Run": "ms"}) as ms:
        kinds = [e.type for e in ms]
    assert "input_json" in kinds or "text" in kinds
    assert RUN not in {k.lower() for k in fake.messages.headers_seen[-1]}


def test_sdk_openai_chunk_types(jit):
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletionChunk

    from treejit.inline import _event_obj, _sse_events
    from treejit.model import ToolCall

    sse = dialect("openai").build_sse("m", [ToolCall("call_tj_x", "Bash", {"command": "ls"})], {"stream_options": {"include_usage": True}})
    evs = [_event_obj("openai", e) for e in _sse_events(sse)]
    assert all(isinstance(e, ChatCompletionChunk) for e in evs)
    assert accumulate("openai", evs)[0] == [("call_tj_x", "Bash", {"command": "ls"})]


def test_real_anthropic_client_passthrough_attrs(jit):
    anthropic = pytest.importorskip("anthropic")
    real = anthropic.Anthropic(api_key="x", base_url="http://127.0.0.1:9")
    w = jit.wrap(real)
    assert w.messages.count_tokens == real.messages.count_tokens and w.messages.batches is real.messages.batches
    with pytest.raises(TypeError, match="synchronous"):
        jit.wrap(anthropic.AsyncAnthropic(api_key="x"))
