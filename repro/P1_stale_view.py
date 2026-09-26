"""Side finding: an outcome reported by another process (`treejit outcome ...` CLI) rebuilds the tree there and
clears `dirty`, so a running proxy's cached TreeView (engine.TreeJIT._views) is never reloaded.

Two TreeJIT instances on one db file stand for the proxy and the CLI.
"""
import os, sys, tempfile
sys.path.insert(0, "/home/user/treejit/tests")
from conftest import Model, run_agent
from treejit import TreeJIT

db = os.path.join(tempfile.mkdtemp(), "t.db")
proxy, cli = TreeJIT(db), TreeJIT(db)
model = Model(lambda task, hist, body: ("Read", {"file_path": "a.txt"}) if not hist else None)
client = proxy.wrap(model, dialect="anthropic")
ex = lambda n, a: ("hello", False)
for i in range(3):
    rid = f"r{i}"
    run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), f"read a.txt {i}", ex)
    (proxy if i == 0 else cli).outcome(rid, "pass")   # first outcome in-process, the rest via the "CLI"
fam = proxy.store.q1("SELECT id FROM families")["id"]
cached = proxy.view(fam)
fresh = TreeJIT(db).view(fam)
print("proxy cached view: hot edges =", sum(ne.tier == "hot" for kids in cached.children.values() for ne in kids),
      "| fresh view from db: hot edges =", sum(ne.tier == "hot" for kids in fresh.children.values() for ne in kids))
calls0 = model.calls
run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "r9"}), "read a.txt 9", ex)
print("model calls for a now-proven task through the running proxy:", model.calls - calls0, "(a fresh process would replay: 1 = final answer only)")
fresh_jit = TreeJIT(db)
c2 = fresh_jit.wrap(model, dialect="anthropic")
calls0 = model.calls
run_agent(lambda b: c2(b, extra_headers={"X-TreeJIT-Run": "r10"}), "read a.txt 10", ex)
print("same task through a freshly started process:", model.calls - calls0, "model call(s)")
