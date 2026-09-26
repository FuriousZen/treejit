"""E4b: a T2 subcall on the FIRST request of a header-less run that fails (model answers 0)
-> the subcall's request row keeps run_id NULL, so `explain` drops its tokens."""
from __future__ import annotations

import sys

sys.path[:0] = ["/home/user/treejit/src", "/home/user/treejit/bench/src"]
from treejit import TreeJIT, operate  # noqa: E402

TOOLS = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]
SYSTEM = "You are an agent.\n" * 5
n_small = {"n": 0}


def model(body):
    forced = (body.get("tool_choice") or {}).get("name", "")
    if forced == "treejit_choose":
        n_small["n"] += 1
        return {"id": "m", "type": "message", "role": "assistant", "model": "x", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": f"toolu_s{n_small['n']}", "name": forced, "input": {"choice": 0}}],
                "usage": {"input_tokens": 300, "output_tokens": 20}}
    msgs = body["messages"]
    task = msgs[0]["content"]
    n = sum(1 for m in msgs if m["role"] == "assistant")
    plan = {"alpha": ["ls", "pwd"], "beta": ["date", "pwd"], "gamma": ["whoami", "pwd"]}[task.split()[0]]
    if n < len(plan):
        content = [{"type": "tool_use", "id": f"toolu_{len(msgs)}_{task.split()[1]}", "name": "Bash", "input": {"command": plan[n]}}]
        stop = "tool_use"
    else:
        content, stop = [{"type": "text", "text": "done"}], "end_turn"
    return {"id": "m", "type": "message", "role": "assistant", "model": "x", "stop_reason": stop, "content": content,
            "usage": {"input_tokens": 1000, "output_tokens": 50}}


jit = TreeJIT(":memory:")
w = jit.wrap(model, dialect="anthropic")
tasks = [f"{k} task{i}" for i in range(6) for k in ("alpha", "beta")] + ["gamma task99"]
for t in tasks:
    msgs = [{"role": "user", "content": t}]
    for _ in range(5):
        r = w({"model": "x", "max_tokens": 100, "system": SYSTEM, "tools": TOOLS, "messages": msgs})  # no X-TreeJIT-Run
        msgs.append({"role": "assistant", "content": r["content"]})
        uses = [b for b in r["content"] if b["type"] == "tool_use"]
        if not uses:
            break
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
    jit.outcome("latest", "pass")
print("small calls made:", n_small["n"])
for r in jit.store.q("SELECT id, run_id, tier, note, input_tokens+output_tokens tok FROM requests WHERE tier IN ('T2','T3')"):
    print(dict(r))
rid = jit.store.q1("SELECT id FROM runs WHERE task='gamma task99'")["id"]
d = operate.explain(jit.store, rid)
print(operate.explain_text(d))
tot = jit.store.q1("SELECT SUM(input_tokens+output_tokens) n FROM requests WHERE family IS NOT NULL")["n"]
print("tokens in requests table:", tot, " sum over explain of all runs:",
      sum(operate.explain(jit.store, r["id"])["tokens"] for r in jit.store.q("SELECT id FROM runs")))
