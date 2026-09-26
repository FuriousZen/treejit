"""X1: T2/T3 subcalls on current Claude models: no forced tool_choice (a 400 on Fable 5.1, Mythos 5.1
and Opus 5.5), structured outputs, no `thinking`, effort only where the model takes it."""

from __future__ import annotations

import json

import pytest
from conftest import Model, calls_of, replayed_ids
from test_tiers import FILES, _commit_setup, commit_policy, fill_all, fs_exec, pick_like_policy, train

from treejit.config import Config
from treejit.dialects import get as dialect
from treejit.model import ToolCall
from treejit.replay import Option, Subcall
from treejit.subcalls import (CHOOSE_TOOL, FILL_TOOL, THINKING_MAX_TOKENS, answer_content, build, resolve, strict_schema,
                              subcall_tool, tool_input)
from treejit.templates import anti_unify
from treejit.tree import EdgeInfo, NodeEdge

CURRENT = ["claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5",
           "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5", "anthropic.claude-opus-5-5", "m"]


def _opt(holes=("command#0",), cmd="git commit -m 'old'"):
    ne = NodeEdge("n1", "e1", 3, 3, 0, 3, 0.0, False, True, False, "live", 1.0, 0.8, 0.8, {h: None for h in holes},
                  list(holes), {}, {}, {"command": cmd}, [], False, 0.0, 0.0, 0.0, True)
    other = "git commit -m 'x y'" if holes else cmd
    tpl = anti_unify([ToolCall("a", "Bash", {"command": cmd}), ToolCall("b", "Bash", {"command": other})])
    return Option(ne, EdgeInfo("e1", "Bash", "s", tpl, "Bash(...)"), {}, list(holes),
                  args=None if holes else {"command": cmd})


def _req(model="m"):
    return dialect("anthropic").parse_request({"model": model, "max_tokens": 9, "tools": [],
                                               "thinking": {"type": "adaptive"},
                                               "messages": [{"role": "user", "content": "commit it"}]})


FILL = Subcall("fill", "n1", "r1", [_opt()], "holes")
CHOOSE = Subcall("choose", "n1", "r1", [_opt(), _opt((), "ls -la")], "ambiguous")


def _walk(x):
    yield x
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


@pytest.mark.parametrize("model", CURRENT)
@pytest.mark.parametrize("sub", [FILL, CHOOSE], ids=["fill", "choose"])
def test_anthropic_subcall_body_is_valid_on_current_models(model, sub):
    b = build("anthropic", sub, _req(model), Config())
    assert "tool_choice" not in b and "tools" not in b          # forced tool use: 400 on Fable 5.1 / Opus 5.5
    assert "thinking" not in b                                  # `disabled` is a 400 where thinking is always on
    fmt = b["output_config"]["format"]
    assert fmt["type"] == "json_schema" and subcall_tool(b) == (FILL_TOOL if sub is FILL else CHOOSE_TOOL)
    schema = fmt["schema"]
    for node in _walk(schema):
        if isinstance(node, dict):
            assert "minimum" not in node and "maximum" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False and set(node["required"]) <= set(node["properties"])
    if sub is CHOOSE:
        assert schema["properties"]["choice"]["enum"] == [0, 1, 2] and schema["required"] == ["choice"]
    else:
        assert schema["required"] == ["command_0"] and "not_this_step" in schema["properties"]
    effort = ("haiku" not in model and model != "m")
    assert (b["output_config"].get("effort") == "low") is effort
    if any(k in model for k in ("fable", "mythos", "opus-5", "sonnet-5")):
        assert b["max_tokens"] >= THINKING_MAX_TOKENS             # thinking counts toward max_tokens
    else:
        assert b["max_tokens"] == Config().subcall_max_tokens


def test_effort_and_format_knobs():
    b = build("anthropic", FILL, _req("claude-opus-5-5"), Config(subcall_effort=""))
    assert "effort" not in b["output_config"]
    b = build("anthropic", FILL, _req("claude-opus-5-5"), Config(subcall_format="tool_auto"))
    assert b["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert b["tools"][0]["strict"] is True and b["tools"][0]["input_schema"]["additionalProperties"] is False
    assert "format" not in b["output_config"] and subcall_tool(b) == FILL_TOOL and FILL_TOOL in b["system"]
    # no call with tool_choice auto: the subcall failed, T4 as always
    assert resolve(FILL, tool_input("anthropic", FILL, {"content": [{"type": "text", "text": "I think..."}]}))[2] == "no_tool_call"
    ok = {"content": [{"type": "tool_use", "id": "t", "name": FILL_TOOL, "input": {"command_0": "Msg"}}]}
    assert resolve(FILL, tool_input("anthropic", FILL, ok))[1]["command#0"].cooked == "Msg"
    # legacy models without structured outputs keep the forced tool call
    legacy = build("anthropic", FILL, _req("claude-3-5-haiku-20241022"), Config())
    assert legacy["tool_choice"] == {"type": "tool", "name": FILL_TOOL} and "output_config" not in legacy
    assert build("anthropic", FILL, _req("claude-opus-5-5"), Config(subcall_format="tool"))["tool_choice"]["type"] == "tool"


@pytest.mark.parametrize("content,expect", [
    ([{"type": "text", "text": '{"command_0": "Fix typo"}'}], "Fix typo"),
    ([{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": '{"command_0": "A"}'}], "A"),
    ([{"type": "text", "text": '```json\n{"command_0": "B"}\n```'}], "B"),
    ([{"type": "text", "text": 'Here you go: {"command_0": "C"} hope that helps'}], "C"),
    ([{"type": "text", "text": '{"command_0": '}, {"type": "text", "text": '"D"}'}], "D"),
    ([{"type": "text", "text": "not json"}], None),
    ([{"type": "text", "text": "[1, 2]"}], None),
    ([{"type": "thinking", "thinking": "", "signature": "s"}], None),   # thinking only (max_tokens hit)
    ([], None),
])
def test_structured_output_parsing(content, expect):
    out = tool_input("anthropic", FILL, {"content": content, "stop_reason": "end_turn"})
    opt, vals, why = resolve(FILL, out)
    assert (vals["command#0"].cooked if vals else None) == expect


def test_strict_schema_keeps_optional_fields_optional():
    s = strict_schema({"type": "object", "properties": {"choice": {"type": "integer", "minimum": 0, "maximum": 2},
                                                        "o1_x": {"type": "string"}}, "required": ["choice"]})
    assert s == {"type": "object", "properties": {"choice": {"type": "integer", "enum": [0, 1, 2]}, "o1_x": {"type": "string"}},
                 "required": ["choice"], "additionalProperties": False}


# ------------------------------------------------------------------ end to end, always-thinking model


class ThinkingModel(Model):
    """Answers like claude-opus-5-5 would: a thinking block, then the JSON text; any forced tool_choice
    or explicit thinking config on a subcall is a 400, as on the real model."""

    def __init__(self, policy, answer):
        super().__init__(policy)
        self.answer, self.small, self.sub_bodies = answer, 0, []

    def __call__(self, body):
        if not subcall_tool(body):
            return super().__call__(body)
        tc = body.get("tool_choice") or {}
        if tc.get("type") in ("tool", "any") or "thinking" in body:
            raise RuntimeError("400: tool_choice: type \"tool\" and \"any\" are not supported for this model.")
        self.small += 1
        self.sub_bodies.append(body)
        content = [{"type": "thinking", "thinking": "", "signature": "sig"}] + answer_content(body, self.answer(body))
        return {"id": "msg_s", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "end_turn",
                "content": content, "usage": {"input_tokens": 30, "output_tokens": 8}}


def test_t3_on_an_always_thinking_model(tmp_path):
    from treejit import TreeJIT

    jit = TreeJIT(str(tmp_path / "t.db"), small_model="claude-opus-5-5")
    model = ThinkingModel(commit_policy, fill_all("Update the docs"))
    _commit_setup(jit, model)
    [m] = train(jit, model, ["commit src/m7.py"], fs_exec(FILES), prefix="x")
    assert calls_of(m)[1] == ("Bash", {"command": "git commit -m 'Update the docs'"})
    assert replayed_ids(m)[1].endswith("_t3") and model.small == 1
    sub = model.sub_bodies[0]
    assert sub["model"] == "claude-opus-5-5" and sub["output_config"]["effort"] == "low"
    row = jit.store.q1("SELECT note, input_tokens FROM requests WHERE tier='T3'")
    assert "ok:" in row["note"] and row["input_tokens"] == 30
    jit.close()


def test_t2_choose_through_structured_output(tmp_path):
    from test_tiers import branch_policy

    from treejit import TreeJIT

    files = {f"status/s{i}": "status report" for i in range(30)}
    files["status/log"] = "log"
    jit = TreeJIT(str(tmp_path / "t.db"), t2=False)
    model = ThinkingModel(branch_policy, pick_like_policy)
    train(jit, model, [f"check s{i}" for i in range(6)], fs_exec(files))
    jit.cfg.t2 = True
    ok, = train(jit, model, ["check s11"], fs_exec(files), prefix="x")
    assert calls_of(ok)[1] == ("Bash", {"command": "ls healthy"}) and replayed_ids(ok)[1].endswith("_t2")
    schema = model.sub_bodies[0]["output_config"]["format"]["schema"]
    assert schema["properties"]["choice"]["enum"][0] == 0 and json.dumps(schema).count("minimum") == 0
    jit.close()
