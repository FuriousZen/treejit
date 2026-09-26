"""Frontier prefix compaction: eligibility, keep-last, dependencies, determinism, dialects."""

from __future__ import annotations

import copy
import itertools
import json

import pytest
from conftest import Model, run_agent

from treejit import TreeJIT
from treejit.compaction import depended_on, digest, obs_reads
from treejit.config import Config
from treejit.model import Observation, ToolCall
from treejit.tree import NodeEdge, TreeView, node_id

JUNK = "".join(f"?? build/obj/unit_{i:03d}.o\n" for i in range(40))
DOCS = "\n".join(f"docs/page_{i:02d}.md" for i in range(40))


def module(n: int) -> str:
    return "\n".join([f"# src/m{n}.py", ""] + [f"def f{j}(x):\n    return x + {j}" for j in range(30)])


def status(n: int) -> str:
    return f" M src/m{n}.py\n" + JUNK


def policy(task, hist, body):
    """git status (big) -> Read the modified file (path taken from the status output) -> wc (small)
    -> ls docs (big) -> pwd (small) -> answer."""
    path = hist[0][2].split()[1] if hist else None
    plan = [("Bash", {"command": "git status --short"}), ("Read", {"file_path": path}), ("Bash", {"command": f"wc -l {path}"}),
            ("Bash", {"command": "ls docs"}), ("Bash", {"command": "pwd"})]
    return plan[len(hist)] if len(hist) < len(plan) else None


def execute(task):
    n = int(task.split()[-1])

    def ex(name, args):
        if name == "Read":
            return module(n), False
        cmd = args["command"]
        if cmd.startswith("git status"):
            return status(n), False
        if cmd.startswith("ls"):
            return DOCS, False
        return ("/work" if cmd == "pwd" else f"31 src/m{n}.py"), False
    return ex


def train(jit, tasks, prefix="t"):
    model = Model(policy)
    client = jit.wrap(model, dialect="anthropic")
    convs = []
    for i, t in enumerate(tasks):
        rid = f"{prefix}{i}"
        convs.append(run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), t, execute(t)))
        jit.outcome(rid, "pass")
    return model, convs


def body_of(msgs):
    from conftest import SYSTEM, TOOLS
    return {"model": "m", "max_tokens": 100, "system": SYSTEM, "tools": TOOLS, "messages": msgs}


def results_by_id(body):
    out = {}
    for m in body["messages"]:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    out[b["tool_use_id"]] = b
    return out


def uses(msgs):
    return [b for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]


@pytest.fixture
def cjit(tmp_path):
    j = TreeJIT(str(tmp_path / "c.db"), compact=True, theta=0.0)
    yield j
    j.close()


def replayed_conv(jit):
    """Train until the whole 5-step path replays; return the last conversation (answer included)."""
    _, convs = train(jit, [f"inspect {i}" for i in range(6)])
    msgs = convs[-1]
    assert all("_tj_" in u["id"] for u in uses(msgs)), "the whole path should replay by now"
    return msgs


def test_default_off_forwards_body_unchanged(tmp_path):
    jit = TreeJIT(str(tmp_path / "off.db"), theta=0.0)
    assert jit.cfg.compact is False
    model, _ = train(jit, [f"inspect {i}" for i in range(6)])
    last = model.bodies[-1]
    assert all(b["content"] in (status(5), module(5), DOCS, "/work", "31 src/m5.py") for b in results_by_id(last).values())
    assert jit.store.q1("SELECT COALESCE(SUM(compacted_chars),0) n FROM requests")["n"] == 0
    jit.close()


def test_verified_replayed_step_is_compacted_and_harness_history_untouched(cjit):
    msgs = replayed_conv(cjit)
    ids = [u["id"] for u in uses(msgs)]
    req = body_of(msgs[:-1])  # the request that produced the final answer
    snapshot = copy.deepcopy(req)
    res = cjit.handle("anthropic", req, {"X-TreeJIT-Run": "probe"})
    assert res.kind == "forward"
    assert req == snapshot, "the harness's body must not be mutated"
    got = results_by_id(res.body)
    # step 1 (Read, big, verified, nothing depends on it) is compacted
    read = got[ids[1]]
    assert read["content"].startswith("[treejit: replayed & verified step — Read(file_path=\"src/m5.py\") → ok, ")
    assert "\n" in read["content"] and read["content"].count("\n") == 1
    assert read["tool_use_id"] == ids[1] and read["is_error"] is False
    # step 0 (git status) is big and verified, but the Read's file_path binding read it: kept
    assert got[ids[0]]["content"] == status(5)
    # the last 3 observations are always full (ls docs is big but within keep-last)
    assert [got[i]["content"] for i in ids[2:]] == ["31 src/m5.py", DOCS, "/work"]
    row = cjit.store.q1("SELECT note, compacted_chars FROM requests ORDER BY id DESC LIMIT 1")
    assert "compacted 1 obs/" in row["note"] and row["compacted_chars"] == len(module(5)) - len(read["content"])


def test_compacted_block_list_keeps_cache_control(cjit):
    """A tool_result whose content is a list of text blocks with a prompt-cache breakpoint on an inner
    block: the digest replaces the list with one block, and the breakpoint moves onto it."""
    msgs = replayed_conv(cjit)
    ids = [u["id"] for u in uses(msgs)]
    req = body_of(copy.deepcopy(msgs[:-1]))
    mark = {"type": "ephemeral"}
    for m in req["messages"]:
        for b in m["content"] if isinstance(m["content"], list) else []:
            if b.get("type") == "tool_result" and b["tool_use_id"] == ids[1]:
                text = b["content"]
                b["content"] = [{"type": "text", "text": text[:100]}, {"type": "text", "text": text[100:], "cache_control": mark}]
    res = cjit.handle("anthropic", req, {"X-TreeJIT-Run": "probe-cc"})
    [block] = results_by_id(res.body)[ids[1]]["content"]
    assert block["type"] == "text" and block["text"].startswith("[treejit: replayed & verified step")
    assert block["cache_control"] == mark
    # without a breakpoint, none is invented
    from treejit.compaction import _replace
    assert _replace([{"type": "text", "text": "a"}], "d") == [{"type": "text", "text": "d"}]


def test_unverified_and_model_chosen_steps_are_kept(cjit):
    msgs = replayed_conv(cjit)
    ids = [u["id"] for u in uses(msgs)]
    # a replayed step whose result broke the postcondition (an error) is not verified
    bad = copy.deepcopy(msgs[:-1])
    for m in bad:
        if isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("tool_use_id") == ids[1]:
                    b["is_error"] = True
    got = results_by_id(cjit.handle("anthropic", body_of(bad), {"X-TreeJIT-Run": "bad"}).body)
    assert got[ids[1]]["content"] == module(5)
    # the same step chosen by the model (no replay marker) is never compacted
    plain = json.loads(json.dumps(msgs[:-1]).replace(ids[1], "toolu_model000001"))
    got = results_by_id(cjit.handle("anthropic", body_of(plain), {"X-TreeJIT-Run": "plain"}).body)
    assert got["toolu_model000001"]["content"] == module(5)


def test_min_chars_and_keep_last_knobs(tmp_path):
    jit = TreeJIT(str(tmp_path / "k.db"), compact=True, compact_keep_last=1, theta=0.0)
    msgs = replayed_conv(jit)
    ids = [u["id"] for u in uses(msgs)]
    got = results_by_id(jit.handle("anthropic", body_of(msgs[:-1]), {"X-TreeJIT-Run": "k"}).body)
    assert got[ids[3]]["content"].startswith("[treejit: replayed & verified step — Bash(ls docs) → ok, 40 lines")
    assert got[ids[2]]["content"] == "31 src/m5.py"          # below compact_min_chars
    assert got[ids[4]]["content"] == "/work"                  # last one
    jit.cfg.compact_min_chars = 10 ** 6
    got = results_by_id(jit.handle("anthropic", body_of(msgs[:-1]), {"X-TreeJIT-Run": "k2"}).body)
    assert got[ids[3]]["content"].startswith("[treejit:"), "sticky: once compacted, a step stays compacted"
    jit.close()


def test_deterministic_and_monotonic_across_requests(cjit):
    msgs = replayed_conv(cjit)[:-1]
    counter = itertools.count()
    forwarded = []
    # the model keeps working in the same conversation: every request is a frontier call
    for _ in range(4):
        res = cjit.handle("anthropic", body_of(msgs), {"X-TreeJIT-Run": "mono"})
        assert res.kind == "forward"
        forwarded.append(res.body["messages"])
        cid = f"toolu_model{next(counter):06d}"
        msgs = msgs + [{"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": "send_email", "input": {"to": "x"}}]},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": "sent " * 200}]}]
    for a, b in zip(forwarded, forwarded[1:]):
        # everything but the newest message of request N is byte-identical in request N+1, or went full -> compacted
        for ma, mb in zip(a[:-1], b):
            if ma != mb:
                ra, rb = results_by_id({"messages": [ma]}), results_by_id({"messages": [mb]})
                for k in ra:
                    assert ra[k] == rb[k] or (not str(ra[k]["content"]).startswith("[treejit:")
                                              and rb[k]["content"].startswith("[treejit:")), "compacted steps never go back to full"
    # once compacted, a step's bytes never change again
    compacted: dict[str, str] = {}
    for f in forwarded:
        for k, b in results_by_id({"messages": f}).items():
            if str(b["content"]).startswith("[treejit:"):
                assert compacted.setdefault(k, b["content"]) == b["content"]
    ids = [u["id"] for u in uses(msgs)]
    last = results_by_id({"messages": forwarded[-1]})
    assert last[ids[3]]["content"].startswith("[treejit: replayed & verified step — Bash(ls docs)"), "left the keep-last window"
    assert last[ids[1]]["content"].startswith("[treejit:") and last[ids[0]]["content"] == status(5)
    assert last[ids[2]]["content"] == "31 src/m5.py" and last[ids[4]]["content"] == "/work"  # small: never compacted
    # a fresh engine on the same DB (e.g. after a restart or a rebuild) reproduces the same bytes
    cjit.rebuild()
    again = cjit.handle("anthropic", body_of(msgs[: len(forwarded[-1])]), {"X-TreeJIT-Run": "mono"}).body["messages"]
    assert json.dumps(again[:-1]) == json.dumps(forwarded[-1][:-1])


def test_digest_is_deterministic_and_tells_the_model():
    cfg = Config()
    call = ToolCall("toolu_tj_x_ff", "Bash", {"command": "git  status\n --short"})
    d = digest(call, Observation(" M a.py\n?? b\n"), cfg)
    assert d == digest(call, Observation(" M a.py\n?? b\n"), cfg)
    assert d.startswith('[treejit: replayed & verified step — Bash(git status --short) → ok, 2 lines, 13 chars; first line: "M a.py"; last line: "?? b"]')
    js = json.dumps({"order_id": "#W1", "status": "pending", "items": [1, 2, 3], "meta": {"a": 1}, "total": 59.97})
    d = digest(ToolCall("i", "get_order_details", {"order_id": "#W1"}), Observation(js), cfg)
    assert 'get_order_details(order_id="#W1")' in d
    assert 'json {"order_id": "#W1", "status": "pending", "items": [3], "meta": {…}, "total": 59.97}' in d
    d = digest(call, Observation("F\n1 failed\nExit code 1"), cfg)
    assert "→ error (exit 1), 3 lines" in d


def _ne(node, edge, bindings=None, guard=None):
    return NodeEdge(node, edge, 3, 3, 0, 3, 0.0, False, True, True, "hot", 1.0, 1.0, 1.0, bindings or {}, [], guard or {},
                    {"err": False}, {}, [], False, 0.0, 0.0, 0.0)


def test_depended_on_frontier_children_and_path_bindings():
    cfg = Config(ngram=(1,))
    view = TreeView("fam")
    eids = ["a", "b", "c", "d", "e"]
    # the frontier (after e) has a child whose binding reads obs[-4] (step 1), through a fmt rule
    front = node_id("fam", "r", tuple(eids))
    view.children[front] = [_ne(front, "f", {"x": ["fmt", ["v=", ["x", ["obs", 4], ["line", 0]]]]})]
    # on the path, step 3's edge read obs[-3] (step 0) at its n-gram context
    g = node_id("fam", "g", ("c",))
    view.children[g] = [_ne(g, "d", {"y": ["x", ["obs", 3], ["kv", "id"]], "z": ["arg", 1, "p"]})]
    current, path = depended_on(view, cfg, eids, [None] * 5)
    assert current == {1, 4} and path == {0}
    assert obs_reads(["case", {}, {}]) == {1} and obs_reads(["arg", 2, "x"]) == set() and obs_reads(["x", "task", ["whole"]]) == set()


def test_schema_migration_adds_compacted_chars(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE requests(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, family TEXT, run_id TEXT, dialect TEXT, "
               "tier TEXT, node TEXT, n_calls INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, "
               "cache_read INTEGER DEFAULT 0, cache_write INTEGER DEFAULT 0, latency_ms REAL, status INTEGER, note TEXT, call_ids TEXT)")
    db.execute("INSERT INTO requests(tier) VALUES('T4')")
    db.commit()
    db.close()
    jit = TreeJIT(path)
    assert jit.store.q1("SELECT compacted_chars FROM requests")["compacted_chars"] == 0
    jit.close()
    TreeJIT(path).close()  # idempotent


# ---------------------------------------------------------------- OpenAI


def oa_model(pol):
    ids = itertools.count()
    bodies = []

    def model(body):
        bodies.append(body)
        msgs = body["messages"]
        task = msgs[1]["content"]
        res = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
        hist = [(tc["function"]["name"], json.loads(tc["function"]["arguments"]), res.get(tc["id"], ""), False)
                for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
        act = pol(task, hist, body)
        if act is None:
            msg = {"role": "assistant", "content": "done"}
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_m{next(ids):06d}", "type": "function", "function": {"name": act[0], "arguments": json.dumps(act[1])}}]}
        return {"id": "c", "object": "chat.completion", "model": "m", "usage": {"prompt_tokens": 50, "completion_tokens": 5},
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if act else "stop"}]}
    model.bodies = bodies
    return model


OA_TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in ("Bash", "Read")]


def test_openai_dialect_compacts_tool_messages(cjit):
    model = oa_model(policy)
    client = cjit.wrap(model, dialect="openai")
    convs = []
    for i in range(6):
        task, rid = f"inspect {i}", f"o{i}"
        ex = execute(task)
        msgs = [{"role": "system", "content": "sys " * 30}, {"role": "user", "content": task}]
        for _ in range(12):
            resp = client({"model": "m", "tools": OA_TOOLS, "messages": msgs}, extra_headers={"X-TreeJIT-Run": rid})
            msg = resp["choices"][0]["message"]
            msgs.append(msg)
            if not msg.get("tool_calls"):
                break
            for tc in msg["tool_calls"]:
                out, _ = ex(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        cjit.outcome(rid, "pass")
        convs.append(msgs)
    last = model.bodies[-1]["messages"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert len(tool_msgs) == 5 and all("_tj_" in m["tool_call_id"] for m in tool_msgs)
    assert tool_msgs[1]["content"].startswith("[treejit: replayed & verified step — Read(")
    assert tool_msgs[0]["content"] == status(5)
    # the harness still holds the full observation
    assert [m for m in convs[-1] if m["role"] == "tool"][1]["content"] == module(5)
