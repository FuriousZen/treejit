"""Engine behaviour through inline mode with a scripted model."""

from __future__ import annotations

import json

from conftest import Model, calls_of, replayed_ids, run_agent

from treejit import TreeJIT
from treejit.replay import hints
from treejit.tree import node_id


def fs_exec(files: dict[str, str]):
    def ex(name, args):
        if name == "Read":
            p = args["file_path"]
            return (files[p], False) if p in files else (f"no such file {p}", True)
        if name == "Bash":
            return f"ran: {args['command']}", False
        return "sent", False
    return ex


def inspect_policy(task, hist, body):
    path = task.split()[-1]
    plan = [("Bash", {"command": "git status --short"}), ("Read", {"file_path": path}), ("Bash", {"command": f"wc -l {path}"})]
    return plan[len(hist)] if len(hist) < len(plan) else None


def train(jit, model, tasks, exec_, outcome=True, run_prefix="run"):
    client = jit.wrap(model, dialect="anthropic")
    out = []
    for i, t in enumerate(tasks):
        rid = f"{run_prefix}-{i}"
        msgs = run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), t, exec_)
        jit.outcome(rid, outcome if not callable(outcome) else outcome(t, msgs))
        out.append(msgs)
    return out


FILES = {f"src/m{i}.py": f"line\n{i}" for i in range(10)}


def test_promotion_needs_two_passing_runs_then_t0_replays_with_bindings(jit):
    model = Model(inspect_policy)
    ex = fs_exec(FILES)
    [m1] = train(jit, model, ["inspect src/m1.py"], ex, run_prefix="a")
    assert model.calls == 4 and not replayed_ids(m1)
    [m2] = train(jit, model, ["inspect src/m2.py"], ex, run_prefix="b")
    assert not replayed_ids(m2), "one passing run must not promote"
    before = model.calls
    [m3] = train(jit, model, ["inspect src/m3.py"], ex, run_prefix="c")
    assert calls_of(m3) == [("Bash", {"command": "git status --short"}), ("Read", {"file_path": "src/m3.py"}),
                            ("Bash", {"command": "wc -l src/m3.py"})]
    assert len(replayed_ids(m3)) >= 2
    assert model.calls - before < 4
    # the whole path is eventually served, leaving only the final answer to the model
    [m4] = train(jit, model, ["inspect src/m4.py"], ex, run_prefix="d")
    [m5] = train(jit, model, ["inspect src/m5.py"], ex, run_prefix="e")
    before = model.calls
    [m6] = train(jit, model, ["inspect src/m6.py"], ex, run_prefix="f")
    assert model.calls - before == 1 and len(replayed_ids(m6)) == 3
    assert calls_of(m6)[1] == ("Read", {"file_path": "src/m6.py"})


def test_failed_and_errored_runs_do_not_promote(jit):
    model = Model(inspect_policy)
    ex = fs_exec(FILES)
    train(jit, model, [f"inspect src/m{i}.py" for i in range(3)], ex, outcome=False, run_prefix="f")
    train(jit, model, [f"inspect src/m{i}.py" for i in range(3, 6)], ex, outcome="error", run_prefix="e")
    [m] = train(jit, model, ["inspect src/m7.py"], ex, run_prefix="p")
    assert not replayed_ids(m)


def _branch_setup(jit, down_action):
    def policy(task, hist, body):
        name = task.split()[-1]
        if not hist:
            return "Read", {"file_path": f"status/{name}"}
        if len(hist) == 1:
            return ("Bash", {"command": "ls healthy"}) if json.loads(hist[0][2])["state"] == "ok" else down_action
        return None

    files = {f"status/s{i}": json.dumps({"state": "ok" if i % 2 else "down", "id": i}) for i in range(20)}
    files["status/log"] = "log"
    model = Model(policy)
    ex = fs_exec(files)
    train(jit, model, [f"check s{i}" for i in range(6)], ex)
    before = model.calls
    [ok] = train(jit, model, ["check s11"], ex, run_prefix="ok")
    [down] = train(jit, model, ["check s12"], ex, run_prefix="down")
    assert calls_of(ok)[1] == ("Bash", {"command": "ls healthy"})
    assert calls_of(down)[1] == down_action
    assert len(replayed_ids(ok)) == 2 and len(replayed_ids(down)) == 2
    assert model.calls - before == 2  # final answers only


def test_t1_branch_between_edges_on_observation(jit):
    _branch_setup(jit, ("Read", {"file_path": "status/log"}))
    notes = [r["note"] for r in jit.store.q("SELECT note FROM requests WHERE tier='T1'")]
    assert notes and all("T1@" in n for n in notes)


def test_value_branch_within_one_edge_uses_case_binding(jit):
    _branch_setup(jit, ("Bash", {"command": "ls broken"}))
    rows = jit.store.q("SELECT bindings FROM node_edges WHERE bindings LIKE '%case%'")
    assert rows


def test_side_exit_when_postcondition_fails(tmp_path):
    jit = TreeJIT(str(tmp_path / "t.db"), batch=False)
    model = Model(inspect_policy)
    ex = fs_exec(FILES)
    train(jit, model, [f"inspect src/m{i}.py" for i in range(4)], ex)
    [m] = train(jit, model, ["inspect src/missing.py"], ex, run_prefix="x")
    notes = [r["note"] for r in jit.store.q("SELECT note FROM requests WHERE run_id='x-0' ORDER BY id")]
    assert any(n.startswith("side_exit") for n in notes), notes
    # after the side exit the model decided the next step
    assert "_tj_" not in [b for b in m[-3]["content"] if b["type"] == "tool_use"][-1]["id"]
    jit.close()


def test_hard_cap_limits_consecutive_replays(tmp_path):
    jit = TreeJIT(str(tmp_path / "t.db"), hard_cap=3, theta=0.0)

    def policy(task, hist, body):
        return ("Bash", {"command": f"ls dir{len(hist)}"}) if len(hist) < 9 else None

    model = Model(policy)
    ex = fs_exec({})
    train(jit, model, [f"walk {i}" for i in range(4)], ex)
    [m] = train(jit, model, ["walk 9"], ex, run_prefix="cap")
    run, longest = 0, 0
    for msg in m:
        if msg["role"] != "assistant":
            continue
        for b in msg["content"]:
            if b["type"] == "tool_use":
                run = run + 1 if "_tj_" in b["id"] else 0
                longest = max(longest, run)
    assert longest == 3
    jit.close()


def test_commit_point_needs_approval_and_extra_evidence(jit):
    def policy(task, hist, body):
        who = task.split()[-1]
        plan = [("Read", {"file_path": f"people/{who}"}), ("send_email", {"to": f"{who}@example.com"})]
        return plan[len(hist)] if len(hist) < 2 else None

    files = {f"people/p{i}": f"person {i}" for i in range(20)}
    model = Model(policy)
    ex = fs_exec(files)
    train(jit, model, [f"email p{i}" for i in range(4)], ex)
    [m] = train(jit, model, ["email p10"], ex, run_prefix="noapp")
    ids = replayed_ids(m)
    assert len(ids) == 1 and calls_of(m)[1][0] == "send_email" and "_tj_" not in m[3]["content"][0]["id"]
    edge = jit.store.q1("SELECT id FROM edges WHERE tool='send_email'")["id"]
    jit.store.x("INSERT INTO approvals(edge, node, ts) VALUES(?, '', 0)", (edge,))
    jit.rebuild()
    [m2] = train(jit, model, ["email p11"], ex, run_prefix="app")
    assert len(replayed_ids(m2)) == 2 and calls_of(m2)[1] == ("send_email", {"to": "p11@example.com"})


def test_tombstone_and_hints_at_frontier(jit):
    def policy(task, hist, body):
        who = task.split()[-1]
        if not hist:
            return "Read", {"file_path": f"people/{who}"}
        if len(hist) == 1:
            if "BAD" in task and "AVOID" not in json.dumps(body["messages"][-1]):
                return "Bash", {"command": "rm -rf build"}
            return "send_email", {"to": f"{who}@example.com"}
        return None

    files = {f"people/p{i}": f"person {i}" for i in range(20)}
    model = Model(policy)
    ex = fs_exec(files)
    jit.cfg.hints = "off"
    train(jit, model, ["email p1", "email p2", "email p3"], ex)
    ran_rm = lambda t, m: "fail" if any(c[0] == "Bash" for c in calls_of(m)) else "pass"  # noqa: E731
    train(jit, model, ["BAD email p4", "BAD email p5"], ex, outcome=ran_rm, run_prefix="bad")
    jit.cfg.hints = "failures"
    jit.store.x("UPDATE runs SET reason='deleted the build dir' WHERE outcome='fail'")
    jit.rebuild()
    tomb = jit.store.q("SELECT tier, reasons FROM node_edges ne JOIN edges e ON e.id=ne.edge WHERE e.tool='Bash' AND ne.tier='tomb'")
    assert tomb and "deleted the build dir" in tomb[0]["reasons"]
    fam = jit.store.q1("SELECT id FROM families")["id"]
    view = jit.view(fam)
    read_edge = jit.store.q1("SELECT id FROM edges WHERE tool='Read'")["id"]
    text = hints(view, jit.cfg, node_id(fam, "r", (read_edge,)))
    assert "AVOID Bash(rm -rf build)" in text and "deleted the build dir" in text and "worked before: send_email" in text
    # at the frontier (send_email isn't approved) the model sees the hint and avoids the bad edge
    [m] = train(jit, model, ["BAD email p9"], ex, run_prefix="hinted")
    assert "treejit-hints" in json.dumps(model.bodies[-2]["messages"][-1])
    assert calls_of(m)[1][0] == "send_email"


def test_run_id_derivation_and_latest_outcome(jit):
    model = Model(inspect_policy)
    client = jit.wrap(model, dialect="anthropic")
    for i in range(3):
        run_agent(client, f"inspect src/m{i}.py", fs_exec(FILES))
        assert jit.outcome("latest", "pass")
    runs = jit.store.q("SELECT id, n_steps, outcome FROM runs")
    assert len(runs) == 3 and all(r["id"].startswith("r_") and r["n_steps"] == 3 and r["outcome"] == "pass" for r in runs)
    m = run_agent(client, "inspect src/m5.py", fs_exec(FILES))
    assert replayed_ids(m)


def test_openai_dialect_inline(jit):
    def model(body):
        msgs = body["messages"]
        task = msgs[1]["content"]
        n = sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant")
        path = task.split()[-1]
        if n == 0:
            tc = {"id": f"call_{len(msgs)}{task[-4:]}", "type": "function", "function": {"name": "Read", "arguments": json.dumps({"file_path": path})}}
            return {"id": "c", "object": "chat.completion", "model": "m", "usage": {"prompt_tokens": 50, "completion_tokens": 5},
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        return {"id": "c", "object": "chat.completion", "model": "m", "usage": {"prompt_tokens": 60, "completion_tokens": 3},
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}

    client = jit.wrap(model, dialect="openai")
    tools = [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]

    def run(task, rid):
        msgs = [{"role": "system", "content": "sys " * 30}, {"role": "user", "content": task}]
        first = client({"model": "m", "tools": tools, "messages": msgs}, extra_headers={"X-TreeJIT-Run": rid})
        tc = first["choices"][0]["message"]["tool_calls"][0]
        msgs += [first["choices"][0]["message"], {"role": "tool", "tool_call_id": tc["id"], "content": "contents"}]
        client({"model": "m", "tools": tools, "messages": msgs}, extra_headers={"X-TreeJIT-Run": rid})
        jit.outcome(rid, "pass")
        return tc

    run("read src/a.py", "o1")
    run("read src/b.py", "o2")
    tc = run("read src/c.py", "o3")
    assert "_tj_" in tc["id"] and json.loads(tc["function"]["arguments"]) == {"file_path": "src/c.py"}


def test_stats_and_rebuild_are_idempotent(jit):
    model = Model(inspect_policy)
    train(jit, model, [f"inspect src/m{i}.py" for i in range(4)], fs_exec(FILES))
    a = jit.store.q("SELECT node, edge, tier, pass_runs FROM node_edges ORDER BY node, edge")
    jit.rebuild()
    b = jit.store.q("SELECT node, edge, tier, pass_runs FROM node_edges ORDER BY node, edge")
    assert [tuple(r) for r in a] == [tuple(r) for r in b]
    s = jit.stats()
    assert s["T4"]["requests"] > 0 and s.get("T0", {}).get("tool_calls", 0) > 0
