"""Commit points where the model has chosen differently (PLAN B1).

tau-bench retail, edges approved: after `get_order_details` on a pending order the model cancels it
or modifies it, and only the task text says which. The decision list learned
`json.status == "pending" -> cancel_pending_order` (4 cancels, 1 modify: purity 0.8, support 4) and
T1 replayed an irreversible cancel into a task that wanted the order modified. Airline test 29: a
read-only task, where the model's choice after the lookup was to stop (END), got a replayed cancel.
"""

from __future__ import annotations

import json
import re

from conftest import calls_of, replayed_ids, run_agent
from test_tiers import SubModel, approve_all

from treejit import TreeJIT
from treejit.config import Config
from treejit.features import learn_decision_list
from treejit.replay import commit_rule_ok

TOOLS = [
    {"name": "get_order_details", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    {"name": "cancel_pending_order",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}}},
    {"name": "modify_pending_order_address",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "city": {"type": "string"}}}},
]

# What each task wants is not in its words (every task says the same thing apart from the order id),
# so no task-word rule separates the kinds; the observation is identical too (a pending order).
# That is the tau-bench situation: the model reads the intent from text treejit can't key on.
INTENT: dict[str, str] = {}


def task_for(oid: str, intent: str) -> str:
    INTENT[oid] = intent
    return f"Please handle order {oid} as we agreed on the phone."


def policy(task, hist, body):
    oid = re.search(r"#W\d+", task).group(0)
    if not hist:
        return "get_order_details", {"order_id": oid}
    if len(hist) == 1:
        want = INTENT[oid]
        if want == "cancel":
            return "cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}
        if want == "modify":
            return "modify_pending_order_address", {"order_id": oid, "city": "Austin"}
    return None  # "lookup" tasks stop after reading the order


def pick(body):
    """T2 answer: the option that is what `policy` would do next, else 0 (something else)."""
    prompt = body["messages"][0]["content"]
    task = re.search(r"<task>\n(.*?)\n</task>", prompt, re.S).group(1)
    oid = re.search(r"#W\d+", task).group(0)
    want = {"cancel": "cancel_pending_order", "modify": "modify_pending_order_address"}.get(INTENT[oid], "(stop)")
    for line in prompt.splitlines():
        m = re.match(r"(\d+)\. (.*)", line)
        if m and f'"tool": "{want}"' in m.group(2):
            return {"choice": int(m.group(1))}
    return {"choice": 0}


class Orders:
    def __init__(self) -> None:
        self.done: dict[str, str] = {}

    def __call__(self, name, args):
        oid = args["order_id"]
        if name == "get_order_details":
            status = "cancelled" if self.done.get(oid) == "cancel" else "pending"
            return json.dumps({"order_id": oid, "status": status, "items": 2}), False
        if oid in self.done:
            return "Error: non-pending order cannot be modified", True
        self.done[oid] = "cancel" if name.startswith("cancel") else "modify"
        return json.dumps({"order_id": oid, "status": "cancelled" if name.startswith("cancel") else "pending"}), False


def run(jit, model, orders, oid, intent):
    client = jit.wrap(model, dialect="anthropic")
    rid = f"r-{oid}"
    msgs = run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), task_for(oid, intent), orders, tools=TOOLS)
    ok = orders.done.get(oid) == (None if intent == "lookup" else intent)
    jit.outcome(rid, "pass" if ok else "fail", None if ok else "wrong database state")
    return msgs


def _train(jit, intents):
    model, orders = SubModel(policy, pick), Orders()
    jit.cfg.t2 = False  # training: the model decides every step itself
    for i, intent in enumerate(intents):
        run(jit, model, orders, f"#W{1000 + i}", intent)
    jit.cfg.t2 = True
    approve_all(jit)  # `treejit approve '*'`
    return model, orders


def test_contested_commit_point_is_not_replayed_on_an_observation_rule(jit):
    model, orders = _train(jit, ["cancel", "cancel", "modify", "cancel", "cancel"])
    fam = jit.store.q1("SELECT id FROM families")["id"]
    cancel = jit.store.q1("SELECT id FROM edges WHERE tool='cancel_pending_order'")["id"]
    rules = [r for d in jit.view(fam).stumps.values() for r in d["rules"] if r["edge"] == cancel]
    # like the rule that misrouted tau-bench retail train 238 (`json.status == "pending"`): an observation
    # predicate (here `err == false`), 4 of 5, one leak
    assert any(r["pred"][0] == "feat" and r["leak"] == 1 and r["support"] == 4 and r["purity"] == 0.8 for r in rules)

    small = model.small
    m = run(jit, model, orders, "#W2000", "modify")
    # before the fix: T1 replayed cancel_pending_order(#W2000) here, conf 0.70, and the modify failed
    assert ("cancel_pending_order", {"order_id": "#W2000", "reason": "no longer needed"}) not in calls_of(m)
    assert orders.done["#W2000"] == "modify"
    assert model.small - small == 1  # the task went to the model once (T2 over the known choices), then T4
    assert len(replayed_ids(m)) == 1  # only the read replayed

    # a cancel task still gets its cancel, chosen by the model in the same small call (no full call)
    calls = model.calls
    m = run(jit, model, orders, "#W2001", "cancel")
    assert orders.done["#W2001"] == "cancel" and replayed_ids(m)[1].endswith("_t2")
    assert model.calls - calls == 1  # the final answer only


def test_end_is_a_sibling_of_a_commit_point(jit):
    # 8 cancels and 1 read-only task: T0's 8-to-1 majority used to replay the cancel into the next
    # read-only task (airline test 29). Stopping is a choice the model made here, so the cancel is contested.
    model, orders = _train(jit, ["cancel"] * 4 + ["lookup"] + ["cancel"] * 4)
    m = run(jit, model, orders, "#W3000", "lookup")
    assert [c[0] for c in calls_of(m)] == ["get_order_details"] and "#W3000" not in orders.done


def test_uncontested_commit_point_still_replays(jit):
    # the model has only ever cancelled after this lookup: T0 replays it (with approval + evidence), no small call
    model, orders = _train(jit, ["cancel"] * 4)
    small = model.small
    m = run(jit, model, orders, "#W4000", "cancel")
    assert orders.done["#W4000"] == "cancel" and len(replayed_ids(m)) == 2 and model.small == small


def test_commit_rule_needs_zero_leak_support_and_resemblance():
    cfg = Config()
    words = {"cancel", "order", "please", "needed"}
    ex = [("c", {"s": "p"}, "", words | {f"w{i}"}) for i in range(5)] + [("m", {"s": "d"}, "", {"modify", "order"})]
    dl = learn_decision_list(ex, 0.8, class_sets=True, gated={"c"})
    [rule] = [r for r in dl["rules"] if r["edge"] == "c"]
    assert rule["leak"] == 0 and rule["support"] == 5 and rule["ex"]  # obs rule on a commit label keeps its sets
    assert commit_rule_ok(cfg, rule, words | {"w9"})
    assert not commit_rule_ok(cfg, rule, {"modify", "address", "order", "street", "city"})  # a new kind of task
    assert not commit_rule_ok(cfg, dict(rule, leak=1), words)
    assert not commit_rule_ok(cfg, dict(rule, support=4), words)
    assert not commit_rule_ok(cfg, {k: v for k, v in rule.items() if k != "leak"}, words)  # unknown leak: no


def test_no_commit_replay_without_gate_when_t2_is_off(tmp_path):
    jit = TreeJIT(str(tmp_path / "t.db"), t2=False)
    model, orders = _train(jit, ["cancel", "cancel", "modify", "cancel", "cancel"])
    jit.cfg.t2 = False
    m = run(jit, model, orders, "#W5000", "modify")
    assert orders.done["#W5000"] == "modify" and len(replayed_ids(m)) == 1
    jit.close()
