"""L1: a chance task-word rule that reached task_rule_support still replays once into the input that breaks it.

Scripted, deterministic; mirrors seed 3 task 13. After `git status`, delete tasks go `git rm <path>`, typo tasks
go `Read <path>`. The first 5 delete tasks all name src/ paths (with distinct file names, so every task has a
distinct word set), typo tasks so far only docs/. The decision list learns task~src -> git rm with
support 5 == task_rule_support, so the first src/ typo task is replayed (T1) into `git rm` without a T2 check.

Scenario A (the bug): novel src/ typo task  -> want: T2 asks, model picks Read, file survives.
Scenario B (no regression): a new delete task of the known kind -> want: T1 replays git rm, 0 small calls.
"""
import os, sys, tempfile
sys.path.insert(0, "/home/user/treejit/tests")
from conftest import calls_of, replayed_ids
from test_learning import Repo, kind_policy, pick_like_kind_policy, train, src_rules
from test_tiers import SubModel, approve_all
from treejit import TreeJIT

NAMES = ["alpha", "bravo", "charlie", "delta", "echo"]
TASKS = ["delete src/legacy/alpha.py", "fix the typo in docs/guide.md", "delete src/legacy/bravo.py",
         "fix the typo in docs/intro.md", "delete src/legacy/charlie.py", "delete src/legacy/delta.py",
         "delete src/legacy/echo.py"]


def setup():
    jit = TreeJIT(os.path.join(tempfile.mkdtemp(), "t.db"))  # defaults: task_rule_support=5
    model = SubModel(kind_policy, pick_like_kind_policy)
    repo = Repo()
    repo.files |= {f"src/legacy/{n}.py": "old" for n in NAMES + ["foxtrot"]} | {"src/net/timeparse.py": "recieve"}
    jit.cfg.t2 = False
    train(jit, model, TASKS, repo)
    jit.cfg.t2 = True
    approve_all(jit)
    return jit, model, repo


def tiers(jit, rid):
    return [(r["tier"], r["note"]) for r in jit.store.q("SELECT tier, note FROM requests WHERE run_id=? ORDER BY id", (rid,))]


jit, model, repo = setup()
print("rule(s) task~src -> git rm:", [{k: v for k, v in r.items() if k != "ex"} for r in src_rules(jit)])
small = model.small
ok = "src/net/timeparse.py"
[m] = train(jit, model, ["fix the typo in " + ok], repo, prefix="x", outcome=ok in repo.files)
print("A calls:", calls_of(m))
print("A small calls:", model.small - small, "| tiers:", tiers(jit, "x-0"))
print("A: L1 REPRODUCED (file deleted by T1 replay)" if ok not in repo.files else "A: L1 not reproduced (asked first)")

jit, model, repo = setup()
small = model.small
[m] = train(jit, model, ["delete src/legacy/foxtrot.py"], repo, prefix="y")
t = tiers(jit, "y-0")
print("B calls:", calls_of(m), "| small calls:", model.small - small, "| tiers:", [x[0] for x in t])
print("B: known class still replays via T1" if t[1][0] == "T1" and model.small == small else "B: REGRESSION (known class not replayed by T1)")
