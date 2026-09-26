"""Unit tests: tokenizer, templates, bindings, features, policy, families, dialects."""

from __future__ import annotations

import json

from treejit import dialects
from treejit.bindings import Sources, eval_rule, find_rule
from treejit.config import Config
from treejit.families import resolve
from treejit.features import eval_decision_list, learn_decision_list, obs_features, task_words
from treejit.model import Observation, ToolCall
from treejit.policy import is_commit_point, is_readonly
from treejit.shellwords import command_heads, tokenize
from treejit.store import Store
from treejit.templates import Val, anti_unify, call_slots, render, shape_of

# ------------------------------------------------------------------ shellwords


def test_tokenize_quotes_ops_and_spans():
    s = """cd app && git commit -m "fix: it's done" 2>&1 | tail -5"""
    toks = tokenize(s)
    vals = [t.val for t in toks]
    assert vals == ["cd", "app", "&&", "git", "commit", "-m", "fix: it's done", "2", ">&", "1", "|", "tail", "-5"]
    msg = toks[6]
    assert s[msg.start:msg.end] == '"fix: it\'s done"'


def test_tokenize_heredoc_in_command_substitution():
    s = "git commit -m \"$(cat <<'EOF'\nFix (parens) \"and quotes\"\nEOF\n)\" && git push"
    toks = tokenize(s)
    assert [t.val for t in toks if t.op] == ["&&"]
    assert toks[3].val.startswith("$(cat <<'EOF'")
    assert [t.val for t in toks][-2:] == ["git", "push"]


def test_command_heads():
    assert command_heads("cd app && FOO=1 npm test | tail -5; ls -la") == ["cd", "npm test", "tail", "ls"]
    assert command_heads("python -m pytest -q") == ["python -m pytest"]
    assert command_heads("git -C x status") == ["git"]  # flag before subcommand: program only


# ------------------------------------------------------------------ templates


def test_anti_unify_and_render_shell():
    calls = [ToolCall("1", "Bash", {"command": "git add src/a.py"}), ToolCall("2", "Bash", {"command": "git add 'b c.py'"})]
    tpl = anti_unify(calls)
    assert tpl["args"]["command"]["items"] == [["c", "git", False], ["c", "add", False], ["v", 0]]
    assert call_slots(tpl, calls[1])["command#0"].cooked == "b c.py"
    assert call_slots(tpl, ToolCall("3", "Bash", {"command": "git rm x"})) is None
    out = render(tpl, calls[1].args, {"command#0": Val("new file.py")})
    assert out == {"command": "git add 'new file.py'"}
    assert render(tpl, calls[0].args, {"command#0": Val("z.py")}) == {"command": "git add z.py"}


def test_render_preserves_rest_of_command():
    ref = {"command": 'cd /repo && git commit -m "old message" --quiet'}
    tpl = anti_unify([ToolCall("1", "Bash", ref), ToolCall("2", "Bash", {"command": 'cd /repo && git commit -m "other" --quiet'})])
    out = render(tpl, ref, {"command#0": Val("new msg")})
    assert out["command"] == 'cd /repo && git commit -m "new msg" --quiet'


def test_shape_groups_by_segment_heads():
    a = shape_of(ToolCall("1", "Bash", {"command": "git status --short"}))
    b = shape_of(ToolCall("2", "Bash", {"command": "git status"}))
    c = shape_of(ToolCall("3", "Bash", {"command": "git log -3"}))
    assert a == b != c


def test_json_args_const_and_var():
    calls = [ToolCall("1", "get_order", {"id": "#W1", "verbose": True}), ToolCall("2", "get_order", {"id": "#W2", "verbose": True})]
    tpl = anti_unify(calls)
    assert tpl["args"]["id"] == {"k": "v"} and tpl["args"]["verbose"] == {"k": "c", "v": True}


# ------------------------------------------------------------------ bindings


def _src(task, obs=(), calls=()):
    return Sources(task, list(calls), [Observation(o) for o in obs], [None] * len(calls))


def test_bind_from_task_regex_and_json_obs():
    tasks = ["Cancel order #W123 for bob@x.com", "please cancel #W999 (me: al@y.org)"]
    vals = [Val("#W123"), Val("#W999")]
    rule = find_rule(vals, [_src(t) for t in tasks])
    assert rule[0] == "x" and rule[1] == "task"
    assert eval_rule(rule, _src("cancel order #W555 now")).cooked == "#W555"

    obs = [json.dumps({"user": {"id": "u1"}, "n": 2}), json.dumps({"user": {"id": "u2"}, "n": 5})]
    rule = find_rule([Val("u1"), Val("u2")], [_src("t", [o]) for o in obs])
    assert rule == ["x", ["obs", 1], ["json", ["user", "id"]]]


def test_bind_arg_copy_and_fmt():
    c1 = [ToolCall("a", "Read", {"file_path": "src/a.py"})]
    c2 = [ToolCall("b", "Read", {"file_path": "lib/b.py"})]
    rule = find_rule([Val("src/a.py"), Val("lib/b.py")], [_src("x", ["..."], c1), _src("y", ["..."], c2)])
    assert rule == ["arg", 1, "file_path"]
    fmt = find_rule([Val("Bump version to 1.2.3"), Val("Bump version to 2.0.1")],
                    [_src("release 1.2.3 now"), _src("please ship v2.0.1")])
    assert fmt[0] == "fmt"
    assert eval_rule(fmt, _src("go to 9.9.9")).cooked == "Bump version to 9.9.9"


def test_hole_when_unbindable():
    assert find_rule([Val("free text one"), Val("totally different")], [_src("a"), _src("b")]) is None


def test_partial_rule_abstains_on_minority():
    # 3 of 4 instances quote the value in the task; the 4th doesn't. The rule is never wrong.
    tasks = ["fix 'a1' to 'b1'", "fix 'a2' to 'b2'", "fix 'a3' to 'b3'", "bump to 9.9.9"]
    vals = [Val("b1"), Val("b2"), Val("b3"), Val("zzz")]
    rule = find_rule(vals, [_src(t) for t in tasks])
    assert rule is not None and eval_rule(rule, _src("bump to 1.0.0")) is None


# ------------------------------------------------------------------ features


def test_obs_features():
    f = obs_features(Observation('{"status": "pending", "items": [1, 2]}'))
    assert f["json.status"] == "pending" and f["err"] is False
    assert obs_features(Observation("boom\nExit code 2"))["err"] is True
    assert obs_features(Observation("", True))["empty"] is True


def test_decision_list_branches_on_json_and_has_no_default():
    ex = []
    for st, lab in [("pending", "cancel")] * 3 + [("delivered", "return")] * 3:
        o = Observation(json.dumps({"status": st}))
        ex.append((lab, obs_features(o), o.text, set()))
    dl = learn_decision_list(ex, 0.8)
    pick = lambda st: eval_decision_list(dl, obs_features(Observation(json.dumps({"status": st}))), "", set())  # noqa: E731
    assert pick("pending")["edge"] == "cancel" and pick("delivered")["edge"] == "return"
    assert pick("processed") is None  # unseen value: no catch-all, goes to the model


def test_task_words():
    assert "typo" in task_words("Fix the typo please") and "the" not in task_words("the")


# ------------------------------------------------------------------ policy


def test_policy_readonly_and_commit_points():
    cfg = Config()
    ro = lambda cmd: is_readonly("Bash", {"command": cmd}, cfg)  # noqa: E731
    assert ro("git status --short") and ro("ls -la | head") and ro("grep -rn foo src 2>/dev/null")
    assert not ro("git commit -m x") and not ro("rm -rf x") and not ro("echo hi > f") and not ro("sed -i s/a/b/ f")
    assert not ro("find . -delete") and not ro("cat $(evil)")
    assert is_readonly("Read", {"file_path": "x"}, cfg) and is_readonly("get_user_details", {"user_id": "u"}, cfg)
    assert not is_readonly("Edit", {}, cfg)
    assert is_commit_point("Bash", {"command": "git add x && git push origin main"}, cfg)
    assert is_commit_point("cancel_pending_order", {}, cfg)
    assert not is_commit_point("Bash", {"command": "git commit -m x"}, cfg)


# ------------------------------------------------------------------ families


def test_stable_prefix_learning_merges_dynamic_system_prompts(tmp_path):
    st = Store(str(tmp_path / "f.db"))
    base = "You are Claude Code.\n" + "Static instructions line.\n" * 20
    tools = [{"name": "Bash"}]
    f1 = resolve(st, base + "Today's date: 2026-09-01\ncwd: /a\n", tools, "anthropic")
    f2 = resolve(st, base + "Today's date: 2026-09-02\ncwd: /b\n", tools, "anthropic")
    f3 = resolve(st, base + "Today's date: 2026-09-03\n", tools, "anthropic")
    assert f1 == f2 == f3
    assert st.q1("SELECT prefix FROM families WHERE id=?", (f1,))["prefix"] == base
    other = resolve(st, "A completely different agent prompt. " * 5, tools, "anthropic")
    assert other != f1
    assert resolve(st, base, [{"name": "Other"}], "anthropic") != f1


# ------------------------------------------------------------------ dialects


def _anthropic_body(stream=False):
    return {"model": "m", "stream": stream, "system": [{"type": "text", "text": "sys"}], "tools": [{"name": "Bash", "input_schema": {}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "a\nb"}]}]},
            ]}


def test_anthropic_parse_and_sse_roundtrip():
    d = dialects.get("anthropic")
    req = d.parse_request(_anthropic_body())
    assert req.system == "sys" and req.episode.task == "do it" and req.episode.ready
    assert req.episode.steps[0].call.args == {"command": "ls"} and req.episode.steps[0].obs.text == "a\nb"
    calls = [ToolCall(d.new_call_id("abcdef123456", 0.5), "Read", {"file_path": "x"})]
    acc = d.stream_accumulator()
    for chunk in d.build_sse("m", calls, {}):
        acc.feed(chunk)
    info = acc.result()
    assert info.calls[0].args == {"file_path": "x"} and info.stop_reason == "tool_use"
    assert d.parse_response(d.build_response("m", calls)).calls[0].name == "Read"


def test_replayed_call_id_encodes_node_and_confidence():
    d = dialects.get("anthropic")
    from treejit.model import Step

    st = Step(ToolCall(d.new_call_id("0123456789ab", 0.8), "Read", {}))
    assert st.replayed_node == "0123456789ab" and abs(st.replayed_conf - 0.8) < 0.01
    assert Step(ToolCall("toolu_01abc", "Read", {})).replayed_node is None


def test_openai_parse_and_sse_roundtrip():
    d = dialects.get("openai")
    body = {"model": "m", "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}], "messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "res"}]}
    req = d.parse_request(body)
    assert req.system == "sys" and req.episode.steps[0].call.args == {"a": 1} and req.episode.steps[0].obs.text == "res"
    calls = [ToolCall("call_x", "f", {"b": [1, 2]})]
    acc = d.stream_accumulator()
    for chunk in d.build_sse("m", calls, {"stream_options": {"include_usage": True}}):
        acc.feed(chunk)
    assert acc.result().calls[0].args == {"b": [1, 2]}
    hinted = d.inject_hint(body, "<treejit-hints>x</treejit-hints>")
    assert "treejit-hints" in hinted["messages"][-1]["content"] and "treejit-hints" not in body["messages"][-1]["content"]


def test_thinking_dropped_only_after_replayed_turns():
    d = dialects.get("anthropic")
    body = _anthropic_body()
    body["thinking"] = {"type": "enabled", "budget_tokens": 1000}
    assert "thinking" in d.prepare_forward(d.parse_request(body))
    body["messages"][1]["content"][1]["id"] = "toolu_tj_0123456789ab_ffabc"
    body["messages"][2]["content"][0]["tool_use_id"] = "toolu_tj_0123456789ab_ffabc"
    assert "thinking" not in d.prepare_forward(d.parse_request(body))
