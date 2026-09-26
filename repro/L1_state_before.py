"""Rebuild the seed-3 tree as it was right before task 13 (runs 0..12) and print the decision list at r1."""
import shutil, sys, json
from treejit import TreeJIT
from treejit.features import task_words, eval_decision_list
from treejit.tree import node_id
src, dst = sys.argv[1], sys.argv[2]
shutil.copy(src, dst)
jit = TreeJIT(dst)
st = jit.store
keep = {f"task-3-{i}" for i in range(13)}
drop = [r["id"] for r in st.q("SELECT id FROM runs") if r["id"] not in keep]
for rid in drop:
    st.x("DELETE FROM steps WHERE run_id=?", (rid,)); st.x("DELETE FROM runs WHERE id=?", (rid,))
fam = st.q1("SELECT family FROM runs WHERE id='task-3-13'") or st.q1("SELECT family FROM runs WHERE id='task-3-1'")
fam = fam["family"]
jit.rebuild(fam)
v = jit.view(fam)
labels = {e.id: e.label for e in v.edges.values()}
gs = next(e.id for e in v.edges.values() if e.label.startswith("Bash(git status"))
r1 = node_id(fam, "r", (gs,))
print("node r1 =", r1, "n_pass =", v.node_pass.get(r1))
for k in v.children.get(r1, []):
    print(f"  child {labels[k.edge]:40s} pass_n={k.pass_n} purity={k.purity} tier={k.tier}")
dl = v.stumps.get(r1)
for r in dl["rules"]:
    print("  rule", r["pred"], "->", labels.get(r["edge"], r["edge"]), {k: r[k] for k in r if k not in ("pred", "edge")})
t13 = "Please correct 'definately' to 'definitely' in src/utils/timeparse.py and ship it (tests, commit, push)."
w = task_words(t13)
leaf = eval_decision_list(dl, {}, "", w)
print("task 13 words:", sorted(w))
print("fires:", leaf["pred"], "->", labels[leaf["edge"]], "support", leaf["support"], ">= task_rule_support", jit.cfg.task_rule_support)
# which runs support it
print("examples supporting task~src at r1:")
for r in st.q("SELECT id, task FROM runs ORDER BY created"):
    ws = task_words(r["task"])
    if "src" in ws:
        print("  ", r["id"], sorted(ws))
