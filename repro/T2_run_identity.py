"""T2: run identity without X-TreeJIT-Run (run id = r_ + h(family, task_hash, first tool-call id))."""
import json
import sys
import tempfile

sys.path.insert(0, "/home/user/treejit/tests")
from conftest import SYSTEM, TOOLS, Model, run_agent  # noqa: E402

from treejit import TreeJIT  # noqa: E402


def policy(task, hist, body):
    plan = [("Bash", {"command": "ls"}), ("Bash", {"command": "cat README.md"})]
    return plan[len(hist)] if len(hist) < len(plan) else None


def ex(name, args):
    return ("README.md" if args["command"] == "ls" else "# readme\n" * 3), False


def runs(jit):
    return [(r["id"], r["task"][:30], r["n_steps"], r["outcome"]) for r in jit.store.q("SELECT * FROM runs ORDER BY created")]


tmp = tempfile.mkdtemp()

print("== 1. same task, two conversations (model-chosen first call) ==")
jit = TreeJIT(tmp + "/a.db")
m = Model(policy)
client = jit.wrap(m, dialect="anthropic")
for _ in range(2):
    run_agent(client, "fix the readme", ex)
for r in runs(jit):
    print("  ", r)
print("  -> two runs (distinct model tool ids). A harness that *resumes* a conversation keeps its history, so the id is stable;")
print("     a fresh conversation with the same task is a new run (by design or not).")
jit.close()

print("\n== 2. identical task, first call REPLAYED in both conversations ==")
jit = TreeJIT(tmp + "/b.db")
client = jit.wrap(Model(policy), dialect="anthropic")
for i in range(3):
    rid = f"t{i}"
    run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), f"fix the readme {i}", ex)
    jit.outcome(rid, "pass")
a = run_agent(client, "fix the readme", ex)
b = run_agent(client, "fix the readme", ex)
fa = a[1]["content"][0]["id"]; fb = b[1]["content"][0]["id"]
print(f"   first ids: {fa}  {fb}")
print("   runs:", [r for r in runs(jit) if not r[0].startswith("t")])
print("  -> no collision: replayed ids carry rand_id(10) (62^10 space)")
jit.close()

print("\n== 3. OpenAI-compatible backend with deterministic tool-call ids (e.g. 'call_0', common in local servers) ==")
jit = TreeJIT(tmp + "/c.db")


def oai_model(body):
    msgs = body["messages"]
    n = sum(1 for x in msgs if x["role"] == "tool")
    task = next(x["content"] for x in msgs if x["role"] == "user")
    cmds = ["ls", "cat README.md"] if CONV[0] == 0 else ["ls", "rm -rf build"]
    if n >= 2:
        return {"id": "c", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
    return {"id": "c", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{n}", "type": "function", "function": {"name": "Bash", "arguments": json.dumps({"command": cmds[n]})}}]}, "finish_reason": "tool_calls"}]}


CONV = [0]
oclient = jit.wrap(oai_model, dialect="openai")
OTOOLS = [{"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def oai_run(task):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(5):
        r = oclient({"model": "m", "messages": msgs, "tools": OTOOLS})
        msg = r["choices"][0]["message"]
        msgs.append(msg)
        if not msg.get("tool_calls"):
            break
        for tc in msg["tool_calls"]:
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "ok"})


oai_run("task A: inspect the repo")
rid = runs(jit)[0][0]
jit.outcome(rid, "pass")
CONV[0] = 1
oai_run("task A: inspect the repo")        # same task text, a *different* conversation whose 2nd step differs
oai_run("task A: inspect the repo")
print("   runs:", runs(jit))
print("   steps of", rid, ":", [(s["idx"], s["args"]) for s in jit.store.steps(rid)])
print("  -> all three conversations share ONE run id; a later conversation inherits the first one's 'pass'")
jit.close()

print("\n== 4. Claude-Code-like session: 2 prompts, an interrupt text message mid-episode, then /compact ==")
jit = TreeJIT(tmp + "/d.db")
seen = []


def up(body):
    seen.append(body)
    msgs = body["messages"]
    k = len(msgs)
    return {"id": "m", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": f"toolu_cc{k:04d}", "name": "Bash", "input": {"command": f"echo {k}"}}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}


cc = jit.wrap(up, dialect="anthropic")
msgs = [{"role": "user", "content": "prompt 1: add a flag"}]


def step():
    r = cc({"model": "m", "max_tokens": 10, "system": SYSTEM, "tools": TOOLS, "messages": msgs})
    msgs.append({"role": "assistant", "content": r["content"]})
    msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["content"][0]["id"], "content": "ok"}]})


step(); step(); step()
msgs.append({"role": "user", "content": "[Request interrupted by user] actually use --verbose"})  # same task, user steers
step(); step()
msgs.append({"role": "user", "content": "prompt 2: now write tests"})
step(); step()
# /compact: history replaced by a summary user message; the episode continues
msgs[:] = [{"role": "user", "content": "This session is being continued from a previous conversation... Summary: ..."}]
step(); step()
for r in runs(jit):
    print("  ", r)
print("  -> 4 runs for one session: the interrupt split prompt 1's work into two runs (the first 3 steps never")
print("     get an outcome if the Stop hook posts 'latest'), and /compact restarts the root path under a new 'task'.")
print("   metadata/headers treejit ignores: body.metadata.user_id (Claude Code puts a session id there), x-claude-code-session-id")
jit.close()
