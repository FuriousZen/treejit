"""Learning from outcomes: task-word branch rules, failed replays, and where the model stops."""

from __future__ import annotations

import re

from conftest import calls_of, replayed_ids, run_agent
from test_tiers import SubModel, approve_all

from treejit import TreeJIT
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
    jit = TreeJIT(str(tmp_path / "t.db"), task_rule_support=1)
    model, repo = _misroute_setup(jit)
    [bad] = train(jit, model, ["fix the typo in src/net/c1.py"], repo, prefix="x", outcome=False)
    assert calls_of(bad)[1] == ("Bash", {"command": "git rm src/net/c1.py"}) and len(replayed_ids(bad)) == 2
    assert all(r.get("neg", 0) > 0 for r in src_rules(jit))  # the rule is gone, or penalized and unproven
    [m] = train(jit, model, ["fix the typo in src/net/c2.py"], repo, prefix="y")
    assert calls_of(m)[1] == ("Read", {"file_path": "src/net/c2.py"}) and "src/net/c2.py" in repo.files
    jit.close()


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
