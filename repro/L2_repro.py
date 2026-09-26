"""L2: negatives are counted per rule, not per input class.

The rule task~src -> git rm is right for class A ("delete src/...") and earns T1 (support 5). It then replays
N times on class A, all passing (these are `confirmed` replays). Class B ("fix the typo in src/...") also
satisfies the predicate but needs Read. Each misrouted class-B run fails. Count class-B failures until the rule
stops sending class B down `git rm`, for several N. Also report collateral: after demotion, does class A lose T1?

class B template "delete P carefully" is word-similar to A (Jaccard 0.5 to the nearest A);
"fix the typo in P" is novel (the L1 situation).
usage: L2_repro.py [N ...]   (default 0 4 8 20 40)
"""
import os, sys, tempfile
sys.path.insert(0, "/home/user/treejit/tests")
from conftest import calls_of
import re
import test_learning as TL
from test_learning import Repo, train


def kind_policy(task, hist, body):
    # class A "delete P" -> git rm P; class B -> Read P. B_SIMILAR: "delete P carefully" (word-similar to A);
    # otherwise "fix the typo in P" (a novel word set, which is L1's situation)
    path = next(w for w in task.split() if "/" in w)
    if not hist:
        return "Bash", {"command": "git status"}
    if len(hist) == 1:
        a = task.startswith("delete") and "carefully" not in task
        return ("Bash", {"command": f"git rm {path}"}) if a else ("Read", {"file_path": path})
    return None


TL.kind_policy = kind_policy  # pick_like_kind_policy (the T2 answer) uses the module global
from test_learning import pick_like_kind_policy
B_SIMILAR = True
B_TEMPLATE = "delete {} carefully"
from test_tiers import SubModel, approve_all
from treejit import TreeJIT
from treejit.features import excess_negatives

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike",
         "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform"]


def names(prefix, n):
    return [f"{prefix}{w}{k}" for k in range(n // len(WORDS) + 1) for w in WORDS][:n]


def run(N, max_b=40, verbose=False):
    jit = TreeJIT(os.path.join(tempfile.mkdtemp(), "t.db"))
    model = SubModel(kind_policy, pick_like_kind_policy)
    repo = Repo()
    a_files = [f"src/legacy/{x}.py" for x in names("a", 5 + N + 5)]
    b_files = [f"src/legacy/{x}.py" for x in names("b", max_b)]
    repo.files |= {p: "old" for p in a_files} | {p: "recieve" for p in b_files}
    jit.cfg.t2 = False
    train(jit, model, [f"delete {p}" for p in a_files[:5]] + ["fix the typo in docs/guide.md", "fix the typo in docs/intro.md"], repo)
    jit.cfg.t2 = True
    approve_all(jit)
    t1_a = 0
    for k, p in enumerate(a_files[5:5 + N]):               # class A: T1 replays, all pass
        train(jit, model, [f"delete {p}"], repo, prefix=f"a{k}")
        t1_a += jit.store.q1("SELECT tier FROM requests WHERE run_id=? ORDER BY id LIMIT 1 OFFSET 1", (f"a{k}-0",))["tier"] == "T1"
    fails = 0
    fixed_at = None
    for k, p in enumerate(b_files):                        # class B: misrouted until demoted
        client = jit.wrap(model, dialect="anthropic")
        from conftest import run_agent
        rid = f"b{k}"
        msgs = run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), B_TEMPLATE.format(p), repo)
        ok = p in repo.files
        jit.outcome(rid, "pass" if ok else "fail", None if ok else "target file was deleted")
        tier = jit.store.q1("SELECT tier FROM requests WHERE run_id=? ORDER BY id LIMIT 1 OFFSET 1", (rid,))["tier"]
        if verbose:
            print(f"   B#{k}: tier={tier} ok={ok}")
        if ok:
            fixed_at = k
            break
        fails += 1
    # collateral on class A after demotion
    small0 = model.small
    p = a_files[5 + N]
    train(jit, model, [f"delete {p}"], repo, prefix="post")
    post_tier = jit.store.q1("SELECT tier FROM requests WHERE run_id='post-0' ORDER BY id LIMIT 1 OFFSET 1")["tier"]
    pred = next((k for k in range(0, 100) if excess_negatives(k, N, jit.cfg.purity) > 0), None)
    print(f"N={N:3d} class-A T1 replays={t1_a:3d} | class-B failures before fixed={fails:3d} (predicted floor(N/4)+1={pred})"
          f" | next class-A tier={post_tier} small={model.small - small0}")
    jit.close()
    return fails


if __name__ == "__main__":
    Ns = [int(x) for x in sys.argv[1:]] or [0, 4, 8, 20, 40]
    for tpl in ("delete {} carefully", "fix the typo in {}"):
        B_TEMPLATE = tpl
        print("class B template:", repr(tpl))
        for N in Ns:
            run(N, verbose=False)
