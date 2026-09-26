"""Learning from outcomes: task-word branch rules, failed replays, and where the model stops."""

from __future__ import annotations

import re

from conftest import calls_of, replayed_ids, run_agent
from test_tiers import SubModel, approve_all

from treejit import TreeJIT
from treejit.features import excess_negatives, learn_decision_list, rule_resembles
from treejit.model import ResponseInfo

# ------------------------------------------------------------------ misrouted task-word rule (seed 3)
# Two task kinds share the first step. "delete <path>" goes on with `git rm`, "fix the typo in <path>"
# with `Read`. The first delete tasks all name paths under src/ and the only typo tasks seen so far
# name docs/, so "src" in the task separates the kinds perfectly in the evidence: a decision list
# learns `task~src -> git rm` and, before the fix, replayed it into the first src/ typo task.


def kind_policy(task, hist, body):
    path = task.split()[-1]
    if not hist:
        return "Bash", {"command": "git status"}
    if len(hist) == 1:
        return ("Bash", {"command": f"git rm {path}"}) if task.startswith("delete") else ("Read", {"file_path": path})
    return None


def pick_like_kind_policy(body):
    """T2 answer: the option that matches what kind_policy would do next."""
    prompt = body["messages"][0]["content"]
    task = re.search(r"<task>\n(.*?)\n</task>", prompt, re.S).group(1)
    n_steps = len(re.findall(r'<step n="\d+">', prompt))
    act = kind_policy(task, [None] * n_steps, None)
    if act is None:
        return {"choice": 0}
    want = act[1].get("command") or act[1].get("file_path")
    for line in prompt.splitlines():
        m = re.match(r"(\d+)\. (.*)", line)
        if m and f'"tool": "{act[0]}"' in m.group(2) and want in m.group(2):
            return {"choice": int(m.group(1))}
    return {"choice": 0}


class Repo:
    def __init__(self) -> None:
        self.files = {f"src/legacy/a{i}.py": "old" for i in range(9)} | {f"src/net/c{i}.py": "recieve" for i in range(9)}
        self.files |= {"docs/guide.md": "recieve", "docs/intro.md": "seperate"}

    def __call__(self, name, args):
        if name == "Read":
            p = args["file_path"]
            return (self.files[p], False) if p in self.files else (f"no such file {p}", True)
        cmd = args["command"]
        if cmd.startswith("git rm "):
            p = cmd.split()[-1]
            if self.files.pop(p, None) is None:
                return f"fatal: pathspec '{p}' did not match any files", True
            return f"rm '{p}'", False
        return "nothing to commit, working tree clean", False


def train(jit, model, tasks, exec_, prefix="t", outcome=True):
    client = jit.wrap(model, dialect="anthropic")
    out = []
    for i, t in enumerate(tasks):
        rid = f"{prefix}-{i}"
        out.append(run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), t, exec_))
        jit.outcome(rid, "pass" if outcome else "fail", None if outcome else "target file was deleted")
    return out


def src_rules(jit) -> list[dict]:
    """Decision-list rules on the task word "src" that send the task down the `git rm` edge."""
    fam = jit.store.q1("SELECT id FROM families")["id"]
    rm = jit.store.q1("SELECT id FROM edges WHERE label LIKE '%git rm%'")["id"]
    return [r for d in jit.view(fam).stumps.values() for r in d["rules"] if r["pred"] == ["task", "src"] and r["edge"] == rm]


def _misroute_setup(jit):
    model = SubModel(kind_policy, pick_like_kind_policy)
    repo = Repo()
    jit.cfg.t2 = False  # training runs: the model decides every step itself
    tasks = ["delete src/legacy/a0.py", "fix the typo in docs/guide.md", "delete src/legacy/a1.py", "fix the typo in docs/intro.md",
             "delete src/legacy/a2.py"]
    train(jit, model, tasks, repo)
    jit.cfg.t2 = True
    approve_all(jit)
    assert any(r["purity"] == 1.0 and r["n"] >= 2 for r in src_rules(jit))  # the misleading rule exists
    return model, repo


def test_task_word_rule_asks_before_replaying_until_proven(jit):
    model, repo = _misroute_setup(jit)
    small = model.small
    [m] = train(jit, model, ["fix the typo in src/net/c1.py"], repo, prefix="x")
    # before the fix: T1 replayed `git rm src/net/c1.py` and deleted the file the task was about
    assert calls_of(m) == [("Bash", {"command": "git status"}), ("Read", {"file_path": "src/net/c1.py"})]
    assert "src/net/c1.py" in repo.files
    assert replayed_ids(m)[1].endswith("_t2") and model.small - small == 1
    # the pick is the model's choice, so it is a counterexample that breaks the rule
    assert not src_rules(jit)


def test_failed_replay_counts_against_the_rule_that_chose_it(tmp_path):
    # With the support requirement off, the rule replays into the typo task and the run fails.
    # That failure must count against the rule, so the next src/ typo task is not sent down `git rm` again.
    jit = TreeJIT(str(tmp_path / "t.db"), task_rule_support=1, task_rule_similarity=0)
    model, repo = _misroute_setup(jit)
    [bad] = train(jit, model, ["fix the typo in src/net/c1.py"], repo, prefix="x", outcome=False)
    assert calls_of(bad)[1] == ("Bash", {"command": "git rm src/net/c1.py"}) and len(replayed_ids(bad)) == 2
    assert all(r.get("neg", 0) > 0 for r in src_rules(jit))  # the rule is gone, or penalized and unproven
    [m] = train(jit, model, ["fix the typo in src/net/c2.py"], repo, prefix="y")
    assert calls_of(m)[1] == ("Read", {"file_path": "src/net/c2.py"}) and "src/net/c2.py" in repo.files
    jit.close()


# ------------------------------------------------------------------ similarity gate on task-word rules
# A chance rule can reach its support: the first 5 delete tasks all name src/ paths, the typo tasks so far
# only docs/ (seed 3, task 13). Support alone then lets T1 replay `git rm` into the first src/ typo task.
# The rule is trusted only on tasks that resemble its supporting examples (Jaccard of task words).

DELETES = ["alpha", "bravo", "charlie", "delta", "echo"]


def _proven_rule_setup(jit):
    model = SubModel(kind_policy, pick_like_kind_policy)
    repo = Repo()
    repo.files |= {f"src/legacy/{n}.py": "old" for n in DELETES + ["foxtrot"]} | {"src/net/timeparse.py": "recieve"}
    jit.cfg.t2 = False
    tasks = [f"delete src/legacy/{n}.py" for n in DELETES]
    tasks[1:1], tasks[3:3] = ["fix the typo in docs/guide.md"], ["fix the typo in docs/intro.md"]
    train(jit, model, tasks, repo)
    jit.cfg.t2 = True
    approve_all(jit)
    rules = src_rules(jit)
    assert rules and all(r["support"] >= jit.cfg.task_rule_support for r in rules)  # proven by count alone
    return model, repo


def _tiers(jit, rid):
    return [r["tier"] for r in jit.store.q("SELECT tier FROM requests WHERE run_id=? ORDER BY id", (rid,))]


def test_proven_task_word_rule_asks_on_a_new_kind_of_task(jit):
    model, repo = _proven_rule_setup(jit)
    small = model.small
    [m] = train(jit, model, ["fix the typo in src/net/timeparse.py"], repo, prefix="x")
    # before the fix: T1 replayed `git rm src/net/timeparse.py`, deleting the file the task was about
    assert calls_of(m)[1] == ("Read", {"file_path": "src/net/timeparse.py"}) and "src/net/timeparse.py" in repo.files
    assert model.small - small == 1 and replayed_ids(m)[1].endswith("_t2")
    assert not src_rules(jit)  # the pick is a counterexample


def test_proven_task_word_rule_still_replays_on_its_own_kind(jit):
    model, repo = _proven_rule_setup(jit)
    small = model.small
    [m] = train(jit, model, ["delete src/legacy/foxtrot.py"], repo, prefix="y")
    assert calls_of(m)[1] == ("Bash", {"command": "git rm src/legacy/foxtrot.py"})
    assert _tiers(jit, "y-0")[1] == "T1" and model.small == small


# Negatives per input class (kNN), not per rule. After N passing T1 replays on class A ("delete P"), a
# word-similar class B ("delete P carefully", which needs Read) used to need about N/4 failed runs before
# the pooled excess showed, and the demotion then hit class A too. Failed replays keep their task-word
# sets, and T1 needs the input to be closer to a supporting example than to any of them.

def _class_policy(task, hist, body):
    path = next(w for w in task.split() if "/" in w)
    if not hist:
        return "Bash", {"command": "git status"}
    if len(hist) == 1:
        a = task.startswith("delete") and "carefully" not in task
        return ("Bash", {"command": f"git rm {path}"}) if a else ("Read", {"file_path": path})
    return None


def test_failures_count_per_input_class_not_per_rule(jit, monkeypatch):
    monkeypatch.setitem(globals(), "kind_policy", _class_policy)  # pick_like_kind_policy answers T2 with it
    model = SubModel(_class_policy, pick_like_kind_policy)
    repo = Repo()
    names = ["alpha", "bravo", "charlie", "delta", "echo", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike",
             "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform"]
    a = [f"src/legacy/a{w}{k}.py" for k in range(2) for w in names][:26]
    b = [f"src/legacy/b{w}.py" for w in names[:6]]
    repo.files |= {p: "old" for p in a} | {p: "recieve" for p in b}
    jit.cfg.t2 = False
    train(jit, model, [f"delete {p}" for p in a[:5]] + ["fix the typo in docs/guide.md", "fix the typo in docs/intro.md"], repo)
    jit.cfg.t2 = True
    approve_all(jit)
    for k, p in enumerate(a[5:25]):  # N = 20 passing T1 replays on class A
        train(jit, model, [f"delete {p}"], repo, prefix=f"a{k}")
        assert _tiers(jit, f"a{k}-0")[1] == "T1"
    client = jit.wrap(model, dialect="anthropic")
    failures = 0
    for k, p in enumerate(b):
        rid = f"b{k}"
        run_agent(lambda body: client(body, extra_headers={"X-TreeJIT-Run": rid}), f"delete {p} carefully", repo)
        ok = p in repo.files
        jit.outcome(rid, "pass" if ok else "fail", None if ok else "target file was deleted")
        if ok:
            break
        failures += 1
    assert failures <= 1  # base: 6
    small = model.small
    train(jit, model, [f"delete {a[25]}"], repo, prefix="post")
    assert _tiers(jit, "post-0")[1] == "T1" and model.small == small  # class A keeps T1 (base: T2)


def test_excess_negatives_has_no_float_residue():
    assert excess_negatives(1, 4, 0.8) == 0.0  # 1 - 0.2 * 5 was 2.2e-16
    assert excess_negatives(2, 4, 0.8) > 0.0


def test_task_word_rule_keeps_its_input_classes():
    ex = [("rm", {}, "", {"delete", "src", f"f{i}"}) for i in range(3)] + [("rm", {}, "", {"delete", "src", "f0"})]
    ex += [("read", {}, "", {"fix", "typo", "docs"}), ("read", {}, "", {"fix", "typo", "readme"})]
    neg = [("rm", {}, "", {"delete", "src", "f9", "carefully"}), ("rm", {}, "", {"delete", "src", "f0"})]
    ok = [("rm", {}, "", {"delete", "src", f"g{i}"}) for i in range(10)]  # passing replays: no excess
    dl = learn_decision_list(ex, 0.8, negatives=neg, confirmed=ok, class_sets=True)
    [rule] = [r for r in dl["rules"] if r["pred"][0] == "task" and r["edge"] == "rm"]
    # distinct sets, cut to the words two supporting examples share (f1, f2 are one-off names; f0 recurs)
    assert rule["ex"] == [["delete", "src"], ["delete", "f0", "src"]]
    assert rule["nx"] == [["carefully", "delete", "f9", "src"]]  # a set that also passed is not held against it
    assert rule_resembles(rule, {"delete", "src", "f7"}, 0.5)
    assert not rule_resembles(rule, {"fix", "typo", "src"}, 0.5)                # a new kind of task
    assert not rule_resembles(rule, {"delete", "src", "f8", "carefully"}, 0.5)  # closer to a failed one
    assert rule_resembles(rule, {"fix", "typo", "src"}, 0)                      # gate off
    assert all("ex" not in r for r in learn_decision_list(ex, 0.8)["rules"])     # only when asked (not case rules)
    assert not rule_resembles({"pred": ["task", "src"]}, {"src"}, 0.5)           # no stored sets: not trusted


# ------------------------------------------------------------------ END: where the model stops


def test_final_answer_is_recorded_as_end_and_not_proposed_again(jit):
    def policy(task, hist, body):
        return ("Read", {"file_path": f"f{len(hist)}"}) if len(hist) < 2 else None

    def ex(name, args):
        return "contents", False

    model = SubModel(policy, lambda body: {"choice": 0, "not_this_step": True})
    train(jit, model, [f"read two files {i}" for i in range(3)], ex)
    assert [r["ended_after"] for r in jit.store.q("SELECT ended_after FROM runs")] == [2, 2, 2]
    view = jit.view(jit.store.q1("SELECT id FROM families")["id"])
    assert view.node_end and all(n == 3 for n in view.node_end.values())  # root path [Read, Read] and its n-grams
    before, small = model.calls, model.small
    [m] = train(jit, model, ["read two files 9"], ex, prefix="x")
    assert len(replayed_ids(m)) == 2 and model.calls - before == 1 and model.small == small  # no subcall at the end
    last = jit.store.q("SELECT tier, note FROM requests WHERE run_id='x-0' ORDER BY id")[-1]
    assert last["tier"] == "T4" and last["note"].startswith("end@")


def test_truncated_response_is_not_an_end(jit):
    msgs = [{"role": "user", "content": "t"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {"file_path": "x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "x"}]}]
    body = {"model": "m", "max_tokens": 5, "system": "s", "tools": [{"name": "Read", "input_schema": {}}], "messages": msgs}
    res = jit.handle("anthropic", body, {"X-TreeJIT-Run": "r1"})
    jit.complete(res, ResponseInfo(text="partial", stop_reason="max_tokens"))
    assert jit.store.run("r1")["ended_after"] is None
    jit.complete(res, ResponseInfo(text="done", stop_reason="end_turn"))
    assert jit.store.run("r1")["ended_after"] == 1
