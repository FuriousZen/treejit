"""M1: multi-turn episodes. A later user turn that answers the agent stays in the episode as a user
step ($user:<kind>); a new request after a finished turn starts a new episode. Both dialects."""

from __future__ import annotations

import json
import re

from conftest import SYSTEM, text_msg, tool_msg

from treejit.dialects import asks, episode_of, get as dialect, reply_kind
from treejit.model import USER, Observation, ToolCall

# ------------------------------------------------------------------ parsing


def _a(*blocks):
    return {"role": "assistant", "content": list(blocks)}


def _use(cid, name="Bash", **args):
    return {"type": "tool_use", "id": cid, "name": name, "input": args}


def _res(cid, text="ok", err=False):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": text, "is_error": err}]}


def _txt(text):
    return {"type": "text", "text": text}


def _parse(msgs, mode="auto", **body):
    return dialect("anthropic").parse_request({"model": "m", "messages": msgs, **body}, mode=mode).episode


def test_confirmation_reply_stays_in_the_episode():
    msgs = [{"role": "user", "content": "cancel order #W1"},
            _a(_use("toolu_aaaaaaaaaaaaaaaa01", "get_order_details", order_id="#W1")), _res("toolu_aaaaaaaaaaaaaaaa01", '{"status": "pending"}'),
            _a(_txt("Order #W1 is pending. Do you want me to cancel it? (yes/no)")),
            {"role": "user", "content": "yes"}]
    ep = _parse(msgs)
    assert ep.task == "cancel order #W1" and ep.index == 0 and ep.ready
    assert [s.call.name for s in ep.steps] == ["get_order_details", f"{USER}:yes"]
    assert ep.steps[1].obs.text == "yes" and ep.steps[1].is_user and not ep.steps[1].replayed_node
    # the id of a user step is stable across requests (it names the message and its text)
    msgs2 = msgs + [_a(_use("toolu_bbbbbbbbbbbbbbbb02", "cancel_pending_order", order_id="#W1")), _res("toolu_bbbbbbbbbbbbbbbb02")]
    ep2 = _parse(msgs2)
    assert ep2.steps[1].call.id == ep.steps[1].call.id and len(ep2.steps) == 3
    # 'no' is its own kind, so what follows a 'yes' is never learned for a 'no'
    assert _parse(msgs[:-1] + [{"role": "user", "content": "No, don't."}]).steps[1].call.name == f"{USER}:no"
    # an answer to a question is a text step
    ask = msgs[:-2] + [_a(_txt("What is your email?")), {"role": "user", "content": "it's a@b.com"}]
    ep = _parse(ask)
    assert ep.task == "cancel order #W1" and ep.steps[-1].call.name == f"{USER}:text" and ep.steps[-1].obs.text == "it's a@b.com"


def test_new_prompt_after_a_finished_turn_is_a_new_episode():
    """Claude Code: one conversation, several prompts; each prompt after a final answer is a task."""
    msgs = [{"role": "user", "content": [_txt("<system-reminder>ctx</system-reminder>"), _txt("add a --verbose flag")]},
            _a(_use("toolu_aaaaaaaaaaaaaaaa01", command="ls")), _res("toolu_aaaaaaaaaaaaaaaa01"),
            _a(_txt("Added the flag and updated the help text.")),
            {"role": "user", "content": "now write tests for it"},
            _a(_use("toolu_aaaaaaaaaaaaaaaa02", command="pytest")), _res("toolu_aaaaaaaaaaaaaaaa02")]
    ep = _parse(msgs)
    assert ep.task == "now write tests for it" and ep.index == 1 and [s.call.args for s in ep.steps] == [{"command": "pytest"}]
    assert ep.origin == "add a --verbose flag" and ep.anchor_ids == ["toolu_aaaaaaaaaaaaaaaa01"]
    # a generic closing question doesn't make the next prompt a reply
    closer = msgs[:3] + [_a(_txt("Done. Is there anything else I can help you with?")), {"role": "user", "content": "now write tests for it"}]
    assert _parse(closer).task == "now write tests for it"
    # a local slash-command transcript before the next prompt is not part of it
    local = msgs[:4] + [{"role": "user", "content": "<command-name>/cost</command-name>\n<command-message>cost</command-message>"},
                        {"role": "user", "content": "<local-command-stdout>$0.12</local-command-stdout>"},
                        {"role": "user", "content": "now write tests for it"}]
    ep = _parse(local)
    assert ep.task == "now write tests for it" and ep.index == 1 and ep.steps == []
    # a user message with only harness text is not a turn
    ep = _parse(msgs[:4] + [{"role": "user", "content": "<system-reminder>todo list changed</system-reminder>"}])
    assert ep.task == "add a --verbose flag" and ep.index == 0 and len(ep.steps) == 1


def test_interrupt_continues_the_episode():
    msgs = [{"role": "user", "content": "add a flag"},
            _a(_use("toolu_aaaaaaaaaaaaaaaa01", command="ls")), _res("toolu_aaaaaaaaaaaaaaaa01"),
            _a(_use("toolu_aaaaaaaaaaaaaaaa02", command="rm -rf build")),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_aaaaaaaaaaaaaaaa02", "is_error": True,
                                          "content": "The user doesn't want to proceed with this tool use."},
                                         _txt("[Request interrupted by user for tool use]")]},
            {"role": "user", "content": "don't delete anything, use --verbose instead"}]
    ep = _parse(msgs)
    assert ep.task == "add a flag" and ep.index == 0
    assert [s.call.name for s in ep.steps] == ["Bash", "Bash", f"{USER}:steer"]
    assert ep.steps[-1].obs.text == "don't delete anything, use --verbose instead"
    # interrupted while the model was still writing: the marker as a user message of its own
    msgs2 = msgs[:3] + [_a(_txt("I'll now")), {"role": "user", "content": "[Request interrupted by user]"},
                        {"role": "user", "content": "actually use --verbose"}]
    ep = _parse(msgs2)
    assert ep.task == "add a flag" and ep.steps[-1].call.name == f"{USER}:steer" and ep.steps[-1].obs.text == "actually use --verbose"
    # a bare marker adds no step
    ep = _parse(msgs[:5])
    assert ep.task == "add a flag" and len(ep.steps) == 2


def test_episode_modes():
    msgs = [{"role": "user", "content": "task one"}, _a(_use("toolu_aaaaaaaaaaaaaaaa01", command="ls")),
            _res("toolu_aaaaaaaaaaaaaaaa01"), _a(_txt("Done.")), {"role": "user", "content": "task two please"}]
    assert _parse(msgs).task == "task two please"
    conv = _parse(msgs, mode="conversation")
    assert conv.task == "task one" and conv.index == 0 and conv.steps[-1].call.name == f"{USER}:text"
    # 'turn': every user message starts an episode, even a confirmation (the old behaviour)
    yes = msgs[:3] + [_a(_txt("Shall I push?")), {"role": "user", "content": "yes"}]
    assert _parse(yes).task == "task one" and _parse(yes, mode="turn").task == "yes"
    assert _parse(yes, mode="turn").steps == []


def test_openai_episodes():
    d = dialect("openai")
    tc = {"id": "call_Xy12abCD34efGH56", "type": "function", "function": {"name": "get_order_details", "arguments": '{"order_id": "#W1"}'}}
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "cancel order #W1"},
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": tc["id"], "content": '{"status": "pending"}'},
            {"role": "assistant", "content": "It is pending. Should I cancel it?"},
            {"role": "user", "content": "Yes, please proceed with the cancellation."}]
    ep = d.parse_request({"model": "m", "messages": msgs, "prompt_cache_key": "conv-1", "user": "u1"}).episode
    assert ep.task == "cancel order #W1" and [s.call.name for s in ep.steps] == ["get_order_details", f"{USER}:yes"]
    assert ep.ready and ep.session == "conv-1" and ep.user == "u1"
    new = msgs[:4] + [{"role": "assistant", "content": "It is cancelled."}, {"role": "user", "content": "what's the weather in Paris"}]
    ep = d.parse_request({"model": "m", "messages": new}).episode
    assert ep.task == "what's the weather in Paris" and ep.index == 1 and ep.steps == []
    # a user message between tool results and the next assistant turn steers
    steer = msgs[:4] + [{"role": "user", "content": "use the other order"}]
    ep = d.parse_request({"model": "m", "messages": steer}).episode
    assert ep.task == "cancel order #W1" and ep.steps[-1].call.name == f"{USER}:steer"


def test_reply_classes():
    for t in ("yes", "y", "Yes, please proceed with the cancellation.", "Yes, I confirm.", "ok", "sure", "go ahead", "do it"):
        assert reply_kind(t) == "yes", t
    for t in ("no", "No thanks", "nope", "do not do that", "stop"):
        assert reply_kind(t) == "no", t
    for t in ("yes, but only order #W2", "ok now write tests for it", "Go ahead and cancel order #W123 and #W456",
              "my email is a@b.com", "i changed my mind", "please"):
        assert reply_kind(t) == "text", t
    assert asks("Should I also commit?") and asks("Please confirm (yes) to proceed.")
    assert asks("Could you provide your email or name and zip code?")
    assert not asks("Done. The tests pass.") and not asks("Is there anything else I can help you with?")
    assert not asks("How can I help you today?")


def test_episode_of_events_directly():
    ev = [("u", "task", 0, False), ("a", [ToolCall("toolu_x0000000000000001", "Bash", {"command": "ls"})], ""),
          ("r", "toolu_x0000000000000001", Observation("a")), ("a", [], "Proceed with the deploy?"), ("u", "yes", 4, False)]
    ep = episode_of(ev)
    assert [s.call.name for s in ep.steps] == ["Bash", f"{USER}:yes"] and ep.anchor_salt is not None


# ------------------------------------------------------------------ end to end: T0 after "yes"

ORDER_TOOLS = [
    {"name": "get_order_details", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    {"name": "cancel_pending_order", "input_schema": {"type": "object", "properties": {
        "order_id": {"type": "string"}, "reason": {"type": "string"}}}},
]


class ConfirmModel:
    """Looks the order up, asks before the write, cancels only after a yes (tau-bench's protocol)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, body):
        self.calls += 1
        msgs = body["messages"]
        oid = re.search(r"#W\d+", msgs[0]["content"]).group(0)
        names = [b["name"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
        last = msgs[-1]["content"]
        if "cancel_pending_order" in names:
            return text_msg("Your order is cancelled.")
        if "get_order_details" not in names:
            return tool_msg("get_order_details", {"order_id": oid})
        if not isinstance(last, str):
            return text_msg(f"Order {oid} is pending. Do you want me to cancel it? (yes/no)")
        if last.lower().startswith("yes"):
            return tool_msg("cancel_pending_order", {"order_id": oid, "reason": "no longer needed"})
        return text_msg("OK, I won't cancel it.")


def _exec(name, args):
    if name == "get_order_details":
        return json.dumps({"order_id": args["order_id"], "status": "pending"}), False
    return json.dumps({"order_id": args["order_id"], "status": "cancelled"}), False


def converse(call, task, replies, dialect_name="anthropic"):
    msgs = [{"role": "user", "content": task}]
    replies = list(replies)
    for _ in range(12):
        r = call({"model": "m", "max_tokens": 100, "system": SYSTEM, "tools": ORDER_TOOLS, "messages": msgs})
        msgs.append({"role": "assistant", "content": r["content"]})
        uses = [b for b in r["content"] if b["type"] == "tool_use"]
        if uses:
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": u["id"], "content": _exec(u["name"], u["input"])[0]} for u in uses]})
            continue
        if not replies:
            break
        msgs.append({"role": "user", "content": replies.pop(0)})
    return msgs


def _uses(msgs):
    return [b for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]


def test_confirm_flow_is_one_episode_and_t0_replays_the_write_after_yes(jit):
    model = ConfirmModel()
    client = jit.wrap(model, dialect="anthropic")
    for i in range(3):
        rid = f"conf-{i}"
        msgs = converse(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), f"please cancel order #W10{i}", ["yes"])
        assert [u["name"] for u in _uses(msgs)] == ["get_order_details", "cancel_pending_order"]
        jit.outcome(rid, "pass")
    runs = jit.store.q("SELECT id, task, n_steps FROM runs ORDER BY id")
    assert [(r["id"], r["task"], r["n_steps"]) for r in runs] == [(f"conf-{i}", f"please cancel order #W10{i}", 3) for i in range(3)]
    assert [s["tool"] for s in jit.store.steps("conf-0")] == ["get_order_details", f"{USER}:yes", "cancel_pending_order"]
    # the user step is context, never a replayable choice
    assert not jit.store.q("SELECT 1 FROM node_edges ne JOIN edges e ON e.id=ne.edge WHERE e.tool LIKE '$user%'")
    jit.store.x("INSERT INTO approvals(edge, node, ts) VALUES('*', '', 0)")
    jit.rebuild()

    before = model.calls
    msgs = converse(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "conf-9"}), "please cancel order #W109", ["yes"])
    uses = _uses(msgs)
    assert [u["name"] for u in uses] == ["get_order_details", "cancel_pending_order"]
    assert "_tj_" in uses[0]["id"] and "_tj_" in uses[1]["id"]            # both replayed: the write right after "yes"
    assert uses[1]["input"] == {"order_id": "#W109", "reason": "no longer needed"}
    assert model.calls - before == 2                                     # the question and the final answer only
    tiers = [r["tier"] for r in jit.store.q("SELECT tier FROM requests WHERE run_id='conf-9' ORDER BY id")]
    assert tiers == ["T0", "T4", "T0", "T4"]

    # a "no" lands on another node: nothing is replayed after it
    before = model.calls
    msgs = converse(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "conf-no"}), "please cancel order #W108", ["no"])
    assert [u["name"] for u in _uses(msgs)] == ["get_order_details"] and model.calls - before == 2
    # and a "yes" nobody asked for doesn't count as the confirmation (the agent asked nothing)
    view = jit.view(jit.store.q1("SELECT id FROM families")["id"])
    assert any(e.tool == f"{USER}:yes" for e in view.edges.values())


def test_confirm_flow_header_less_is_one_run_per_conversation(jit):
    model = ConfirmModel()
    client = jit.wrap(model, dialect="anthropic")
    for i in range(3):
        converse(client, f"please cancel order #W20{i}", ["yes"])
        assert jit.outcome("latest", "pass")
    runs = jit.store.q("SELECT id, task, n_steps, outcome FROM runs ORDER BY created")
    assert len(runs) == 3 and all(r["n_steps"] == 3 and r["outcome"] == "pass" for r in runs)
    assert [r["task"] for r in runs] == [f"please cancel order #W20{i}" for i in range(3)]
    assert not jit.store.q("SELECT 1 FROM requests WHERE family IS NOT NULL AND run_id IS NULL")


def test_claude_code_multi_prompt_conversation_splits_into_episodes(jit):
    """One conversation, three prompts, each after a final answer; Claude Code's session id in metadata."""
    tools = [{"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]
    plan = {"add a --verbose flag": ["ls", "sed -i s/x/y/ cli.py"], "now write tests for it": ["pytest"],
            "commit it": ["git commit -am wip"]}

    def model(body):
        msgs = body["messages"]
        k = max(i for i, m in enumerate(msgs) if m["role"] == "user" and isinstance(m["content"], str))
        cmds = plan[msgs[k]["content"]]
        done = sum(1 for m in msgs[k:] if m["role"] == "assistant")
        return tool_msg("Bash", {"command": cmds[done]}) if done < len(cmds) else text_msg("Done.")

    client = jit.wrap(model, dialect="anthropic")
    meta = {"user_id": "user_0f0f_account_1234abcd-0000-0000-0000-000000000000_session_9e8d7c6b-1111-2222-3333-444455556666"}
    msgs: list = []
    for prompt in plan:
        msgs.append({"role": "user", "content": prompt})
        for _ in range(6):
            r = client({"model": "m", "max_tokens": 50, "system": SYSTEM, "tools": tools, "messages": msgs, "metadata": meta})
            msgs.append({"role": "assistant", "content": r["content"]})
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
        jit.outcome("latest", "pass")  # a Stop hook after each turn
    runs = jit.store.q("SELECT id, task, n_steps, outcome FROM runs ORDER BY created")
    assert [(r["task"], r["n_steps"], r["outcome"]) for r in runs] == [
        ("add a --verbose flag", 2, "pass"), ("now write tests for it", 1, "pass"), ("commit it", 1, "pass")]
    # every request of an episode, the first one included (a session id needs no tool call), has its run
    by_run = [r["run_id"] for r in jit.store.q("SELECT run_id FROM requests ORDER BY id")]
    assert None not in by_run and len(set(by_run)) == 3
    assert by_run == [runs[0]["id"]] * 3 + [runs[1]["id"]] * 2 + [runs[2]["id"]] * 2
