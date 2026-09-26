"""H1 / X2: frontier hints are sticky and append-only; the thinking drop is model-aware."""

from __future__ import annotations

import copy

import pytest
from cache_model import HINT_TAG, _dump, append_only, bill, scenario
from conftest import SYSTEM, TOOLS

from treejit import TreeJIT, compaction
from treejit.dialects import get as dialect
from treejit.dialects import thinking_always_on, thinking_default_on


def _no_cc(x):
    if isinstance(x, dict):
        return {k: _no_cc(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_no_cc(v) for v in x]
    return x


def strict_append_only(bodies: list[dict], key: str = "messages") -> list[int]:
    """Indices i where forward i does not extend forward i-1 *including* its last message (a hint given
    on the last message must still be there, identical, in every later forward)."""
    return [i for i in range(1, len(bodies))
            if _no_cc(bodies[i][key][: len(bodies[i - 1][key])]) != _no_cc(bodies[i - 1][key])]


def hint_tokens(bodies: list[dict]) -> float:
    n = 0
    for b in bodies:
        for m in b["messages"]:
            c = m.get("content")
            n += sum(len(_dump([m.get("role"), x])) for x in (c if isinstance(c, list) else [])
                     if isinstance(x, dict) and str(x.get("text", "")).startswith(HINT_TAG))
    return n / 4   # the hint blocks as the cache model serializes them (tokens = chars / 4)


@pytest.mark.parametrize("pattern", ["interleaved", "bursty"])
@pytest.mark.parametrize("mode", [None, "first_sight"])
def test_consecutive_forwards_with_hints_are_append_only(pattern, mode):
    bodies = [b for _, b in scenario(pattern, mode, hints="always")]
    assert sum(1 for b in bodies if hint_tokens([b])) >= 2      # hints were given, more than once
    assert append_only(bodies) == []                            # the C1 criterion
    assert strict_append_only(bodies) == []                     # and the previous last message too (H1)
    # every hint of a request is still in the next one, at the same position
    for a, b in zip(bodies, bodies[1:]):
        for i, m in enumerate(a["messages"]):
            assert b["messages"][i] == m


@pytest.mark.parametrize("pattern", ["interleaved", "bursty"])
@pytest.mark.parametrize("breakpoints", ["harness", "auto"])
def test_cache_model_hinted_billed_at_most_unhinted_plus_hint_tokens(pattern, breakpoints):
    off = scenario(pattern, "first_sight")
    on = scenario(pattern, "first_sight", hints="always")
    b_off, b_on = bill(off, breakpoints=breakpoints), bill(on, breakpoints=breakpoints)
    assert b_on.requests == b_off.requests
    # before the fix a hint broke the next request's prefix: +156-247% under automatic caching
    assert b_on.billed <= b_off.billed + hint_tokens([b for _, b in on])
    assert b_on.vs(b_off) < 0.10


def _conv(n_results: int) -> list[dict]:
    msgs = [{"role": "user", "content": "inspect src/a.py"}]
    for i in range(n_results):
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_model{i:08d}", "name": "Read",
                                                     "input": {"file_path": f"f{i}"}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_model{i:08d}", "content": "x"}]}]
    return msgs


def _body(msgs, **kw):
    return {"model": "m", "max_tokens": 10, "system": SYSTEM, "tools": TOOLS, "messages": copy.deepcopy(msgs), **kw}


def test_sticky_hints_unit_anthropic(jit):
    d = dialect("anthropic")
    r1 = d.parse_request(_body(_conv(1)))
    b1, h1 = compaction.sticky_hints(jit.store, d, r1, r1.raw, "<treejit-hints>A</treejit-hints>")
    assert h1 == "<treejit-hints>A</treejit-hints>" and b1["messages"][-1]["content"][-1]["text"] == h1
    assert r1.raw["messages"][-1]["content"][-1]["type"] == "tool_result"     # harness body untouched
    # a retry of the same request reuses the stored hint (the first one given at a position wins)
    b1b, h1b = compaction.sticky_hints(jit.store, d, r1, r1.raw, "<treejit-hints>B</treejit-hints>")
    assert h1b == h1 and b1b == b1
    # the next request: the old hint is re-inserted where it was, the new one goes at the end
    r2 = d.parse_request(_body(_conv(2), cache_control={"type": "ephemeral"}))
    r2.raw["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}   # harness breakpoint moves
    b2, h2 = compaction.sticky_hints(jit.store, d, r2, r2.raw, "<treejit-hints>C</treejit-hints>")
    assert b2["messages"][2] == b1["messages"][2] and h2 == "<treejit-hints>C</treejit-hints>"
    # no new hint: the earlier ones still go
    r3 = d.parse_request(_body(_conv(3)))
    b3, h3 = compaction.sticky_hints(jit.store, d, r3, r3.raw, None)
    assert h3 is None and b3["messages"][2] == b1["messages"][2]
    assert b3["messages"][4]["content"][-1]["text"] == "<treejit-hints>C</treejit-hints>"
    assert strict_append_only([b1, b2, b3]) == []
    # an assistant-last body can't take a hint at the end: nothing is stored for it
    r4 = d.parse_request(_body(_conv(1) + [{"role": "assistant", "content": [{"type": "text", "text": "hm"}]}]))
    b4, h4 = compaction.sticky_hints(jit.store, d, r4, r4.raw, "<treejit-hints>D</treejit-hints>")
    assert h4 is None and b4["messages"][-1] == r4.raw["messages"][-1]


def test_sticky_hints_openai_and_responses(jit):
    d = dialect("openai")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_abcdefghijkl", "type": "function",
                                                                   "function": {"name": "Read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_abcdefghijkl", "content": "out"}]
    tools = [{"type": "function", "function": {"name": "Read", "parameters": {}}}]
    r1 = d.parse_request({"model": "m", "tools": tools, "messages": msgs})
    b1, _ = compaction.sticky_hints(jit.store, d, r1, r1.raw, "HINT")
    assert b1["messages"][3]["content"] == "out\n\nHINT"
    more = msgs + [{"role": "assistant", "content": "done?"}, {"role": "user", "content": "go on"}]
    r2 = d.parse_request({"model": "m", "tools": tools, "messages": more})
    b2, h2 = compaction.sticky_hints(jit.store, d, r2, r2.raw, None)
    assert h2 is None and b2["messages"][:4] == b1["messages"]
    # Responses: a hint is its own input item, re-inserted after the item it followed
    d = dialect("responses")
    items = [{"role": "user", "content": "t"},
             {"type": "function_call", "call_id": "call_abcdefghijkl", "name": "Read", "arguments": "{}"},
             {"type": "function_call_output", "call_id": "call_abcdefghijkl", "output": "out"}]
    rtools = [{"type": "function", "name": "Read", "parameters": {}}]
    r1 = d.parse_request({"model": "m", "tools": rtools, "input": items})
    b1, _ = compaction.sticky_hints(jit.store, d, r1, r1.raw, "HINT")
    assert b1["input"][3] == {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "HINT"}]}
    later = items + [{"type": "function_call", "call_id": "call_bcdefghijklm", "name": "Read", "arguments": "{}"},
                     {"type": "function_call_output", "call_id": "call_bcdefghijklm", "output": "out2"}]
    r2 = d.parse_request({"model": "m", "tools": rtools, "input": later})
    b2, _ = compaction.sticky_hints(jit.store, d, r2, r2.raw, "HINT2")
    assert b2["input"][:4] == b1["input"] and b2["input"][-1]["content"][0]["text"] == "HINT2"
    assert strict_append_only([b1, b2], key="input") == []


def test_hints_are_pruned_when_idle(tmp_path):
    jit = TreeJIT(str(tmp_path / "h.db"))
    d = dialect("anthropic")
    r = d.parse_request(_body(_conv(1)))
    compaction.sticky_hints(jit.store, d, r, r.raw, "H")
    assert jit.store.q1("SELECT COUNT(*) n FROM hints")["n"] == 1
    jit.store.x("UPDATE hints SET ts=0")
    jit.store._hint_prune_at = 0.0
    r2 = d.parse_request(_body(_conv(2)))
    compaction.sticky_hints(jit.store, d, r2, r2.raw, None, retention_days=7)
    assert jit.store.q1("SELECT COUNT(*) n FROM hints")["n"] == 0
    jit.close()


# ------------------------------------------------------------------ the thinking drop (X2 c)


def test_thinking_model_classes():
    for m in ("claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5", "claude-fable-5", "us.anthropic.claude-opus-5-5"):
        assert thinking_always_on(m) and thinking_default_on(m), m
    for m in ("claude-opus-5", "claude-sonnet-5"):
        assert thinking_default_on(m) and not thinking_always_on(m), m
    for m in ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5", "m", "", "claude-opus-4-5-20251101"):
        assert not thinking_default_on(m) and not thinking_always_on(m), m


@pytest.mark.parametrize("model,dropped", [("claude-sonnet-4-6", True), ("claude-opus-4-8", True), ("m", True),
                                           ("claude-opus-5-5", False), ("claude-fable-5-1", False),
                                           ("claude-opus-5", False)])
def test_thinking_drop_after_replayed_turns_is_model_aware(model, dropped):
    d = dialect("anthropic")
    msgs = [{"role": "user", "content": "t"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_tj_abcdef012345_ff0123456789",
                                               "name": "Read", "input": {"file_path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_tj_abcdef012345_ff0123456789",
                                          "content": "x"}]}]
    body = {"model": model, "max_tokens": 10, "tools": TOOLS, "messages": msgs, "thinking": {"type": "adaptive"}}
    fwd = d.prepare_forward(d.parse_request(body))
    assert ("thinking" not in fwd) is dropped
    if not dropped:
        assert fwd is body   # untouched: removing it changes nothing on these models but the cache prefix
    # without a replayed step nothing is dropped anywhere
    msgs[1]["content"][0]["id"] = msgs[2]["content"][0]["tool_use_id"] = "toolu_model000000001"
    assert "thinking" in d.prepare_forward(d.parse_request(body))
