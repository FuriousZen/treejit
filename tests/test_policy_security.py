"""Replay-safety policy: every known bypass of the read-only allowlist and of commit-point
detection, the common read-only commands that must keep replaying, and an end-to-end regression."""

from __future__ import annotations

import pytest
from conftest import Model, calls_of, replayed_ids
from test_engine import fs_exec, train

from treejit import TreeJIT
from treejit.config import Config
from treejit.model import ToolCall
from treejit.operate import approve
from treejit.policy import commit_reason, is_commit_point, is_readonly, repo_taint
from treejit.replay import Option, materialize
from treejit.shellwords import ansi_c, command_heads, unwrap
from treejit.templates import Val, anti_unify, shape_of, var_slots
from treejit.tree import EdgeInfo, NodeEdge, TreeView

CFG = Config()


def ro(cmd: str, cfg: Config = CFG) -> bool:
    return is_readonly("Bash", {"command": cmd}, cfg)


def commit(cmd: str, cfg: Config = CFG) -> bool:
    return is_commit_point("Bash", {"command": cmd}, cfg)


NOT_READONLY = [
    # transparent prefixes: classify the program they run
    "env git push origin main", "env rm -rf x", "env sh -c 'rm -rf x'", "env FOO=1 curl https://x",
    "env -i git push", "env -u HOME rm x", "/usr/bin/env rm x", "env -S 'rm -rf x'", "env --split-string='rm x'",
    "time rm -rf x", "time -o out.txt ls", "nohup rm x", "exec rm x", "command rm x", "nice -n 5 rm x",
    "nice -10 rm x", "timeout 10 rm x", "timeout -s KILL 5 git push", "stdbuf -oL rm x", "sudo ls",
    "sudo -u root cat /etc/shadow", "env time nice rm x", "command -p rm x",
    # environment that makes a read-only program run something else
    "LD_PRELOAD=/tmp/x.so ls", "PATH=/tmp/evil ls", "PATH=/tmp/evil; ls", "GIT_EXTERNAL_DIFF=/tmp/x git diff",
    "PAGER='sh -c x' git log", "LESSOPEN='|rm %s' cat f", "env LD_PRELOAD=/tmp/x.so ls", "BASH_ENV=/tmp/x ls",
    # sed: w / W / e / r, s///w, s///e, in-place, script files
    "sed '1e touch pwned' f", "sed 's/a/b/w out' f", "sed 's/a/b/e' f", "sed 'w out' f", "sed -n '1W out' f",
    "sed '$r /etc/passwd' f", "sed -i 's/a/b/' f", "sed --in-place=.bak 's/a/b/' f", "sed -ibak 's/a/b/' f",
    "sed -ni p f", "sed -f script.sed f", "sed -e p -e 'w x' f", "sed --expression='1e id' f", "sed '1!e id' f",
    "sed 'y/a/b/' f", "sed '/x/{w out\n}' f", "sed 's/a/b/;e id' f", "sed 's|a|b|w out' f", "sed '1a text' f",
    # awk: pipes, getline, system, redirection, indirect calls, extensions
    "awk '{print | \"sh\"}' f", "awk 'BEGIN{\"id\" | getline x; print x}'", "awk '{system(\"rm x\")}' f",
    "awk '{print > \"out\"}' f", "awk '{print >> \"out\"}' f", "awk '{printf \"x\" |& \"cat\"}' f",
    "awk 'BEGIN{f=\"sys\" \"tem\"; @f(\"id\")}'", "awk -f prog.awk f", "awk -i inplace '{print}' f",
    "awk '@load \"filefuncs\"'", "awk --load=x '{print}'", "awk 'BEGIN{getline line < \"/etc/passwd\"}'",
    # find: every output/exec action
    "find . -delete", "find . -exec rm {} ;", "find . -execdir rm {} +", "find . -ok rm {} ;", "find . -okdir rm {} ;",
    "find . -fprint out", "find . -fprint0 out", "find . -fprintf out %p", "find . -fls out",
    "find . {-delete,-name} x", "find . $ACTION", "find . -name *.py",
    # git: global config/exec options, output files, pagers, writes hidden behind read subcommands
    "git push origin main", "git -c core.pager='sh -c x' log", "git -c alias.x='!rm y' status",
    "git --config-env=core.pager=X log", "git --exec-path=/tmp status", "git -C log push origin main",
    "git log --output=x", "git diff --output=x", "git show --output x HEAD", "git log --out=x", "git diff --outp x",
    "git grep -O foo", "git grep --open-files-in-pager=vim foo", "git grep -nO foo", "git grep --open foo",
    "git config user.name x", "git config --get user.name --add x y", "git config --unset x", "git config -e",
    "git config set user.name x", "git config --global --replace-all a b", "git branch newbranch",
    "git branch -D old", "git branch -m a b", "git branch -r newbranch", "git branch --set-upstream-to=o/m",
    "git tag v1.0", "git tag -a v1 -m x", "git tag -d v1", "git remote add o url", "git remote set-url o url",
    "git remote remove o", "git remote prune o", "git remote update", "git worktree add ../x", "git stash",
    "git stash pop", "git stash drop", "git reflog expire --all", "git reflog delete HEAD@{1}", "git checkout x",
    "git log $X", "git diff *", "git fetch", "git", "git config edit", "git config unset x.y",
    "git reflog -1 expire", "git reflog --all delete HEAD@{1}",
    # output files and program-running options of otherwise read-only tools
    "sort -o out f", "sort --output=out f", "sort -uo out f", "sort -oout f", "sort --out=out f",
    "sort --compress-program=sh f", "uniq in out", "date -s '2020-01-01'", "date --set=x", "date 010100002020",
    "hostname evil", "hostname -F /tmp/h", "hostname -b x", "tree -o out", "tree -aoout", "tree -R",
    "file -C -m magic", "file --compile -m magic", "file -p f", "yq -i '.a=1' f.yaml", "yq --inplace '.a=1' f",
    "yq --in-place . f", "yq -s '.a' f", "yq e -i . f", "fd -x rm", "fd --exec rm", "fd -X rm", "fd -tf -x rm",
    "rg --pre ./x foo", "rg --pre=./x foo", "rg --hostname-bin=./x foo", "rg foo *", "ag --pager 'sh -c x' foo",
    "ag --pag=x foo", "bat --pager 'sh -c x' f", "bat cache --build", "less f", "more f", "xargs rm", "tee f",
    # redirections
    "echo hi > f", "echo hi >> f", "echo hi >| f", "echo hi &> f", "echo hi 2> f", "echo hi 1> f", "echo hi >& f",
    "cat f <> f", "echo x >&out", "ls &>> f", "cat f > $OUT", "cat f >/dev/null2", "exec 3> f",
    "diff <(cat a) b", "cat f > >(tee x)", "cat <<EOF\nhi\nEOF",
    # substitutions and expansions in command position
    "cat $(evil)", "cat `evil`", "echo \"$(rm x)\"", "$CMD", "${CMD} x", "$(echo rm) x", "\"$X\" status",
    "l{s,} -la", "./ls", "/tmp/ls", "ls$IFS-la", "cat <<< \"$(rm x)\"", "sort $'\\055o' out f",
    "$'\\162m' x", "git $'\\160ush'", "sort f - o", "ls\r", "ls\x0b-la",
    # shell structure the policy doesn't model
    "sh -c 'ls'", "bash -c 'git status'", "eval ls", "{ ls; }", "f() { rm x; }; f", "for f in *; do rm $f; done",
    "! rm x", "source x.sh", ". x.sh", "python -c 'print(1)'",
]

READONLY = [
    "git status --short", "git status", "git log --oneline -5", "git diff HEAD~1", "git diff --stat",
    "git show HEAD:src/x.py", "git log --format='%h %s' -n 3", "git -C sub status", "git --no-pager log -3",
    "git branch", "git branch -a", "git branch -vv", "git branch --list 'feat*'", "git branch --show-current",
    "git tag", "git tag -l 'v*'", "git remote -v", "git remote show origin", "git config user.name",
    "git config --get user.email", "git config --list", "git config get user.name", "git stash list",
    "git stash show -p stash@{0}", "git reflog -5", "git rev-parse --show-toplevel", "git ls-files '*.py'",
    "git grep -n foo", "git log --output-indicator-new=x -1", "git blame -L 1,5 f", "git diff -- '*.py'",
    "ls -la src", "ls", "cat a b", "grep -rn foo src", "rg foo", "rg -n --glob '*.py' foo", "head -50 f",
    "tail -n 20 f", "wc -l f", "find . -name '*.py'", "find . -type f -mtime -1 -print0",
    "find src -name '*.py' -not -path './.venv/*'", "jq .x f.json", "jq '.[0] | {a, b}' f.json", "sort f",
    "sort -k2,2n -t: f", "sort -rn f", "uniq f", "uniq -c", "cd x && git status", "ls | head", "grep x f 2>/dev/null",
    "grep x f 2>&1 | head", "ls >/dev/null 2>&1", "cat f &>/dev/null", "echo hi", "printf '%s\\n' x",
    "sed -n '10,20p' f", "sed -n 1,5p f", "sed 's/foo/bar/g' f", "sed -E 's/(a|b)+/x/2' f", "sed -n '/start/,/end/p' f",
    "sed '1d;$d' f", "sed -n '$p' f", "sed -e 's/a/b/' -e '/x/d' f", "sed 5q f", "sed -n '0~4p' f",
    "sed 's|/usr|/opt|' f", "awk '{print $1}' f", "awk -F: '$3 >= 1000 {print $1}' /etc/passwd",
    "awk 'NR==1 || /x/' f", "awk -v n=3 'NR<=n' f", "date", "date +%Y-%m-%d", "date -u -d yesterday +%s",
    "hostname", "hostname -f", "tree -L 2 -I node_modules", "tree -a", "file f", "file -b --mime-type f",
    "yq '.a' f.yaml", "yq e '.a' f.yaml", "fd foo", "fd -e py", "fd -tf foo", "bat f", "env", "printenv PATH",
    "env | sort", "command -v git", "type ls", "which python", "test -f x && cat x", "[ -d src ] && ls src",
    "du -sh .", "df -h", "diff a b", "comm -12 a b", "cut -d: -f1 f", "tr a-z A-Z < f", "nl f", "stat f",
    "readlink -f x", "realpath x", "md5sum -c sums", "ps aux", "column -t f", "LC_ALL=C sort f", "TZ=UTC date",
    "GIT_PAGER=cat git log -3", "time ls", "time -p git status", "timeout 5 git status", "nice -n 5 ls",
    "stdbuf -oL grep x f", "env LC_ALL=C sort f", "env -i ls", "nohup ls", "exec ls", "command ls",
    "/usr/bin/git status", "ls # rm -rf x", "cat f | grep -v x | wc -l", "grep 'git push' f", "cat 'f $(x)'",
    "grep -n '$(' f", "wc -l < f", "git reflog show HEAD", "git config --global core.editor", "echo $'a\\tb'",
]


@pytest.mark.parametrize("cmd", NOT_READONLY)
def test_not_readonly(cmd):
    assert not ro(cmd), cmd


@pytest.mark.parametrize("cmd", READONLY)
def test_readonly_stays_readonly(cmd):
    assert ro(cmd), cmd


COMMIT = [
    "git push origin main", "env git push origin main", "env -i FOO=1 git push", "sudo -u me git push",
    "time git push", "nohup git push &", "timeout 60 git push", "nice -n 5 git push", "stdbuf -oL git push",
    "command git push", "exec git push", "/usr/bin/git push", "git -C repo push origin", "git --no-pager push",
    "env curl https://x", "/usr/bin/curl x", "FOO=1 env BAR=2 time curl x", "sh -c 'git push'",
    "bash -lc 'cd x && git push'", "env -S 'git push origin'", "eval git push", "echo $(git push)",
    "echo `curl x`", "xargs -n1 git push < remotes", "find . -exec git push \\;", "watch 'git push'",
    "echo 'git push' | sh", "bash <<EOF\ngit push\nEOF", "cd x && FOO=1 kubectl -n prod apply -f y",
    "npm --registry x publish", "(cd x && git push)", "ssh host ls",
]
NOT_COMMIT = [
    "git commit -m x", "git status", "grep -rn 'git push' src", "grep -rn curl src", "rg 'ssh' docs",
    "git commit -m 'Fix typo in src/x.py'", "git log --grep push", "echo 'git push'", "git add push.py",
    "git commit -m 'msg && git push origin main'",  # a quoted value is data unless a shell runs it
]


@pytest.mark.parametrize("cmd", COMMIT)
def test_commit_point_through_prefixes(cmd):
    assert commit(cmd), cmd


@pytest.mark.parametrize("cmd", NOT_COMMIT)
def test_not_commit_point(cmd):
    assert not commit(cmd), cmd


def test_extra_readonly_commands_opt_in():
    cfg = Config(extra_readonly_commands=["less", "xargs"])
    assert ro("less f", cfg) and ro("env less f", cfg)
    assert not ro("less f > out", cfg) and not ro("./less f", cfg)


def test_one_view_of_the_program():
    """Edge shapes, the read-only policy and commit points all see the same program."""
    for cmd in ["env git push origin main", "env -i FOO=1 git push", "time -p git push", "timeout -s KILL 5 git push",
                "nice -n 5 git push", "stdbuf -oL git push", "/usr/bin/env git push", "command git push"]:
        assert command_heads(cmd) == ["git push"], cmd
        assert commit(cmd) and not ro(cmd), cmd
    assert command_heads("env") == ["env"] and command_heads("command -v git") == ["command"]
    assert command_heads("cd app && FOO=1 npm test | tail -5; ls -la") == ["cd", "npm test", "tail", "ls"]
    assert unwrap(["env", "-S", "git push"])[:2] == (0, False)
    assert unwrap(["sudo", "ls"])[:2] == (1, False)  # located, but elevation is never side-effect free


def test_env_git_push_edge_never_replays_without_approval(jit):
    """Reviewer's repro: an agent pushes via `env git push origin main` in three passing runs. The edge
    shape is `git push`; it must be a commit point needing approval, and replay must not emit it."""
    def policy(task, hist, body):
        name = task.split()[-1]
        plan = [("Bash", {"command": "git status --short"}), ("Read", {"file_path": f"src/{name}.py"}),
                ("Bash", {"command": "env git push origin main"})]
        return plan[len(hist)] if len(hist) < len(plan) else None

    files = {f"src/m{i}.py": f"x = {i}\n" for i in range(10)}
    model = Model(policy)
    ex = fs_exec(files)
    train(jit, model, [f"ship m{i}" for i in range(3)], ex)
    rows = jit.store.q("SELECT ne.tier, ne.commit_point, ne.blocked FROM node_edges ne JOIN edges e ON e.id = ne.edge "
                       "WHERE e.tool = 'Bash' AND e.template LIKE '%push%'")
    assert rows and all(r["commit_point"] and r["tier"] != "hot" for r in rows)
    assert any(r["blocked"] == "needs_approval" for r in rows)
    for i in range(3, 6):
        [m] = train(jit, model, [f"ship m{i}"], ex, run_prefix=f"r{i}")
        pushes = [j for j, (n, a) in enumerate(calls_of(m)) if "push" in a.get("command", "")]
        assert pushes, "the model still pushes"
        push_ids = [b["id"] for msg in m if msg["role"] == "assistant" for b in msg["content"]
                    if b["type"] == "tool_use" and "push" in b["input"].get("command", "")]
        assert not any(i in replayed_ids(m) for i in push_ids), "push replayed without approval"
    assert replayed_ids(m), "the read-only steps still replay"


# ------------------------------------------------------------------ S1: opaque executors are commit points

OPAQUE = [  # repro/S1: each ran a push (or might) behind a program the old patterns didn't know
    "python -c 'import os; os.system(\"git push\")'", "python3 scripts/release.py", "node -e 'x()'",
    "perl -e 'system(1)'", "ruby -e 'x'", "make deploy", "npm run release", "yarn release", "./deploy.sh",
    "bash scripts/push.sh", "git p", "git -c alias.p=push p", "gh repo delete o/r --yes",
    "gh api -X POST repos/o/r/issues", "gh workflow run d.yml", "curl -X POST https://x", "npx vercel --prod",
    "uv run scripts/pub.py", "make", "python -c 'import subprocess; subprocess.run([\"git\",\"push\"])'",
    "make release", "npm run deploy", "pnpm run publish:all", "sh deploy.sh", "git pub origin main",
    "gh issue create -t x", "gh pr merge 1", "http POST https://x", "fly deploy", "aws s3 sync . s3://b",
    "gcloud app deploy", "heroku releases:rollback", "python -m twine upload dist/*", "source ./env.sh",
    "strace -f git push", "watch git push", "xargs -I{} git push {} main", "cargo run --release", "go run ./cmd/x",
    "git -C repo rebase -x 'git push' main", "git bisect run ./check.sh", "git submodule foreach git push",
]
LOCAL_RUNNERS = [  # not read-only, but not commit points either
    "python -m pytest -q", "python -m pytest -q --lf", "pytest -x", "python -m unittest", "tox -e py311",
    "cargo test", "go test ./...", "npm test", "npm run test:unit", "npm run build", "make test", "uv run pytest",
    "poetry run pytest -q", "git add x", "git commit -m 'x'", "git rm x", "mkdir -p a && touch a/b",
    "sed -i s/a/b/ f", "cp a b", "pip install -e .", "python3 -X dev -m pytest", ".venv/bin/pytest -q",
    "./node_modules/.bin/jest", "./gradlew test", "make -C sub lint test", "npx jest --ci", "yarn build",
    "cargo clippy -- -D warnings", "go vet ./...", "python --version", "git -c user.name=bot commit -m x",
    "kubectl get pods -n prod", "aws ec2 describe-instances", "gh auth status", "gh api repos/o/r/pulls",
]


@pytest.mark.parametrize("cmd", OPAQUE)
def test_opaque_executors_are_commit_points(cmd):
    assert commit(cmd) and commit_reason("Bash", {"command": cmd}, CFG), cmd


@pytest.mark.parametrize("cmd", LOCAL_RUNNERS)
def test_local_runners_are_not_commit_points(cmd):
    assert not commit(cmd), (cmd, commit_reason("Bash", {"command": cmd}, CFG))


def test_commit_reason_names_the_cause():
    r = lambda c: commit_reason("Bash", {"command": c}, CFG)  # noqa: E731
    assert r("git push origin main") == "git push"
    assert r("python3 x.py") == "interpreter" and r("./deploy.sh") == "script" and r("make deploy") == "task runner"
    assert r("sh -c 'git push'") == "shell; git push"
    assert r("ls") == "" and commit_reason("send_email", {"to": "a"}, CFG) == "tool send_*"


def _script_policy(cmd):
    def policy(task, hist, body):
        name = task.split()[-1]
        plan = [("Bash", {"command": "git status --short"}), ("Read", {"file_path": f"src/{name}.py"}),
                ("Bash", {"command": cmd})]
        return plan[len(hist)] if len(hist) < len(plan) else None
    return policy


def _replayed(jit, model, ex, cmd, runs, task="ship"):
    """For each run k: whether the call running `cmd` was replayed."""
    out = []
    for k in runs:
        [m] = train(jit, model, [f"{task} m{k}"], ex, run_prefix=f"r{k}")
        ids = [b["id"] for msg in m if msg["role"] == "assistant" for b in msg["content"]
               if b["type"] == "tool_use" and b["input"].get("command") == cmd]
        assert ids, "the model still runs it"
        out.append(any(i in replayed_ids(m) for i in ids))
    return out


@pytest.mark.parametrize("not_commit", [False, True])
def test_not_commit_is_a_per_edge_override_that_star_never_implies(tmp_path, not_commit):
    """`./run_checks.sh` is opaque: a commit point, so `approve '*'` alone replays it only after
    promote_runs + 1 passing runs. `approve EDGE --not-commit` makes it an ordinary approved write."""
    cmd = "./run_checks.sh"
    jit = TreeJIT(str(tmp_path / "t.db"), theta=0.3, t2=False)
    model, ex = Model(_script_policy(cmd)), fs_exec({f"src/m{i}.py": "x\n" for i in range(10)})
    approve(jit.store, "*", "")
    train(jit, model, [f"ship m{i}" for i in range(2)], ex)
    row = jit.store.q1("SELECT ne.edge, ne.commit_point, ne.commit_reason, ne.blocked FROM node_edges ne "
                       "JOIN edges e ON e.id=ne.edge WHERE e.template LIKE '%run_checks%' AND ne.pass_runs >= 2")
    assert row["commit_reason"] == "script" and row["commit_point"] and row["blocked"] == "commit_point_needs_evidence"
    assert jit.store.not_commit() == set(), "approve '*' never declares anything not-a-commit"
    if not_commit:
        with pytest.raises(ValueError):
            approve(jit.store, "*", "", not_commit=True)
        approve(jit.store, row["edge"], "", not_commit=True)
        jit.rebuild()
    assert _replayed(jit, model, ex, cmd, [2, 3]) == ([True, True] if not_commit else [False, True])
    jit.close()


def test_cli_not_commit_flag(tmp_path):
    from treejit.cli import main as cli
    from treejit.store import Store

    db = str(tmp_path / "c.db")
    with pytest.raises(SystemExit):
        cli(["--db", db, "approve", "*", "--not-commit"])
    st = Store(db)
    st.x("INSERT INTO edges(id, family, tool, shape, template, n, label) VALUES('e1abcdef00', 'f', 'Bash', 's', '{}', 1, 'x')")
    st.close()
    cli(["--db", db, "approve", "e1abcdef", "--not-commit"])
    st = Store(db)
    assert st.not_commit() == {"e1abcdef00"} and ("e1abcdef00", "") in st.approvals()
    st.close()
    cli(["--db", db, "revoke", "e1abcdef", "--not-commit"])
    st = Store(db)
    assert st.not_commit() == set() and ("e1abcdef00", "") in st.approvals()
    st.close()


def test_t3_fill_that_changes_the_commit_reason_is_rejected():
    """A fill is spliced in as one quoted word, but inside `sh -c '...'` that word is a script: turning
    `sh -c 'echo a'` (an approved commit point, reason `shell`) into `sh -c 'git push'` must not replay."""
    calls = [ToolCall("a", "Bash", {"command": "sh -c 'echo a'"}), ToolCall("b", "Bash", {"command": "sh -c 'echo b'"})]
    tpl = anti_unify(calls)
    [slot] = var_slots(tpl)
    ref = calls[1].args
    reason = commit_reason("Bash", ref, CFG)
    assert reason == "shell"
    view = TreeView("f")
    view.edges["e1"] = EdgeInfo("e1", "Bash", shape_of(calls[0]), tpl, "Bash(sh -c $0)")
    view.by_shape[shape_of(calls[0])] = "e1"
    ne = NodeEdge("n1", "e1", 3, 3, 0, 3, 0.0, False, True, False, "live", 1.0, 1.0, 1.0, {slot: None}, [slot], {}, {},
                  ref, [], True, 0.0, 0.0, 0.0, True, commit_reason=reason, approved=True)
    opt = Option(ne, view.edges["e1"], {}, [slot])
    args, why = materialize(view, CFG, opt, {slot: Val("echo c")})
    assert args == {"command": "sh -c 'echo c'"} and why == ""
    args, why = materialize(view, CFG, opt, {slot: Val("git push origin main")})
    assert args is None and why == "commit_reason"


# ------------------------------------------------------------------ S2: bash parity

BASH_PARITY = [  # repro/S2_output.txt: every one of these runs `git push` (or send-pack) in bash
    "git push", r"$'\147it' push", r"$'\x67it' push", r"git $'\160ush'", r"git $'\x70ush'", '$"git" push',
    'git $"push"', "gi\\\nt push", "git \\\npush", "{git,push}", "git {push,}", "g\"\"it push", "gi\\t push",
    '"git" push', "'git' push", "command git push", "exec -a name git push", "env -S 'git push'", "git${IFS}push",
    "git$IFS'push'", "git\tpush", "a=(git push); \"${a[@]}\"", "a=(git); a+=(push); \"${a[@]}\"", "x=push; git $x",
    "p=pu; git ${p}sh", "g=git; $g push", "git \"$(echo push)\"", "git $(printf push)", "git `echo push`",
    "f(){ git \"$@\"; }; f push", "bash <<< 'git push'", "sh<<<'git push'", "echo Z2l0IHB1c2g= | base64 -d | sh",
    "printf 'git push' | bash", "git -c alias.p=push p", "git send-pack origin main", "git --git-dir=.git push",
    "bash -c $'\\147it push'", "bash -c \"\\$'\\\\147it' push\"", "env git${IFS}push", "sh -c 'x=push; git $x'",
    "eval \"git \\${IFS}push\"", "\\git push", "git pus\\h", "git 'pu'sh",
    # shell keywords before the program, and the other remote-writing git subcommands
    "if git push; then echo ok; fi", "{ git push; }", "! git push", "while true; do git push; done",
    "function f { git push; }", "git http-push x", "git svn dcommit", "git p4 submit", "git send-email x.patch",
]


@pytest.mark.parametrize("cmd", BASH_PARITY)
def test_bash_parity_commit_points(cmd):
    assert commit(cmd), cmd


def test_ansi_c_quoting_decodes_like_bash():
    assert command_heads(r"$'\147it' push") == ["git push"]
    assert command_heads(r"$'\x67it' $'\u0070ush'") == ["git push"]
    assert command_heads('$"git" push') == ["git push"]
    assert ansi_c(r"a\tb\n\e\cA\101\x41\u00e9\q") == "a\tb\n\x1b\x01AAé\\q"
    assert ansi_c(r"ab\0cd") == "ab"  # bash ends the word at a NUL
    assert ro("echo $'a\\tb'")


# ------------------------------------------------------------------ S3: patterns only where a program runs

# repro/S3_false_positives.py: (command, commit point). The seven deliberate exceptions stay commit
# points: dry runs (`--dry-run`/`-n` is an option we'd have to trust per tool), local `rsync` (rsync
# can't be told from a remote copy without parsing its host syntax) and `npm run <name>` with a name
# that isn't made of local words (the script is opaque).
FALSE_POSITIVES = [
    ("git stash push", 0), ("git stash push -m wip src/x.py", 0), ("git checkout push", 0), ("git branch -D push", 0),
    ("git fetch origin push", 0), ("git log origin/push", 0), ("git worktree add ../push", 0),
    ("git rebase origin/push", 0), ("git push --dry-run", 1), ("git push -n origin main", 1),
    ("apt install curl", 0), ("apt-get install -y curl wget", 0), ("pip install requests curl", 0),
    ("brew install rsync", 0), ("npm install ssh2 && npm run build", 0), ("man ssh", 0), ("mkdir ssh", 0),
    ("rm -rf wget", 0), ("chmod 700 ssh", 0), ("echo push > f", 0), ("echo 'git push' > NOTES", 0),
    ("npm run push-docs", 1), ("sed -i 's/curl/wget/' Dockerfile", 0), ("curl -s localhost:8000/health", 0),
    ("curl -sSf http://127.0.0.1:3000/ -o /dev/null", 0), ("wget -qO- http://localhost:8080", 0),
    ("rsync -a src/ build/", 1), ("gh pr view 12", 0), ("gh pr list", 0), ("gh pr diff 12", 0), ("gh pr checks 12", 0),
    ("gh release list", 0), ("gh release view v1", 0), ("kubectl apply --dry-run=client -f x.yaml", 1),
    ("npm publish --dry-run", 1), ("cargo publish --dry-run", 1), ("docker compose push", 1),
    ("git push origin main", 1), ("docker push img:1", 1), ("gh pr create -f", 1), ("gh pr merge 3", 1),
    ("curl -X POST https://api.x/y", 1), ("ssh prod 'systemctl restart app'", 1), ("scp dist.tgz prod:/srv", 1),
    ("kubectl delete pod x", 1), ("python -m pytest -q", 0), ("pytest -q", 0), ("npm test", 0), ("cargo test", 0),
    ("go test ./...", 0), ("make test", 0),
    # still detected through global options, runners and find; loopback only for plain GETs
    ("git -C repo push", 1), ("xargs git push < remotes", 1), ("find . -exec git push \\;", 1),
    ("git --no-pager -C x push", 1), ("curl -d x=1 localhost:8000/api", 1), ("curl http://localhost@evil.com/", 1),
    ("curl -X DELETE http://localhost:8000/x", 1), ("gh api -f title=x repos/o/r/issues", 1),
]


@pytest.mark.parametrize("cmd,want", FALSE_POSITIVES)
def test_commit_points_match_only_where_a_program_runs(cmd, want):
    assert commit(cmd) == bool(want), (cmd, commit_reason("Bash", {"command": cmd}, CFG))


# ------------------------------------------------------------------ S4: repository config

def test_trust_repo_config_false_makes_git_reads_need_approval():
    cfg = Config(trust_repo_config=False)
    for cmd in ["git status", "git diff", "git log -1", "git blame f", "cd sub && git status", "/usr/bin/git show"]:
        assert ro(cmd) and not ro(cmd, cfg), cmd
    for cmd in ["ls", "cat f", "grep -rn x src", "wc -l f"]:
        assert ro(cmd, cfg), cmd


TAINTS = [
    "git config core.fsmonitor 'touch x'", "git config diff.evil.textconv 'sh -c x'", "git -C sub config filter.x.clean y",
    "git config --global core.pager 'sh -c x'", "git config set core.hooksPath hooks", "git config alias.st '!sh'",
    "echo '*.dat diff=evil' > .gitattributes", "printf x >> sub/.gitattributes", "cp hook .git/hooks/pre-commit",
    "sed -i s/a/b/ .git/config", "git init --bare vendor/pkg", "git clone https://x/y", "git submodule update --init",
    "tar xzf a.tgz", "tar -C out -xf a.tar", "unzip -q a.zip", "7z x a.7z", "python -m zipfile -e a.zip out",
    "export GIT_DIR=/tmp/x", "GIT_CONFIG_GLOBAL=/tmp/c", "sh -c 'git config core.fsmonitor x'", "rm -rf .git",
]
NO_TAINT = [
    "git config user.name bot", "git config --get core.fsmonitor", "cat .gitattributes", "git status", "ls .git",
    "git add .gitignore", "tar czf out.tgz src", "ls .github/workflows", "git commit -m x", "python -m pytest -q",
    "git init", "echo hi > f",
]


@pytest.mark.parametrize("cmd", TAINTS)
def test_calls_that_taint_the_repository(cmd):
    assert repo_taint("Bash", {"command": cmd}, CFG), cmd


@pytest.mark.parametrize("cmd", NO_TAINT)
def test_calls_that_leave_the_repository_trusted(cmd):
    assert not repo_taint("Bash", {"command": cmd}, CFG), cmd


def test_file_tools_that_write_git_metadata_taint():
    assert repo_taint("Write", {"file_path": ".gitattributes", "content": "*.x diff=e\n"}, CFG)
    assert repo_taint("Edit", {"file_path": "repo/.git/config", "old_string": "a", "new_string": "b"}, CFG)
    assert not repo_taint("Read", {"file_path": ".git/config"}, CFG)
    assert not repo_taint("Write", {"file_path": "README.md", "content": "see .git/config\n"}, CFG)


@pytest.mark.parametrize("config_cmd,replays", [("git config user.name bot", True),
                                                ("git config core.fsmonitor 'touch /tmp/x'", False)])
def test_run_that_sets_exec_config_does_not_replay_later_git_reads(jit, config_cmd, replays):
    """repro/S4: `git status` runs core.fsmonitor. Once the run sets it, git reads stop replaying."""
    def policy(task, hist, body):
        name = task.split()[-1]
        plan = [("Bash", {"command": config_cmd}), ("Bash", {"command": "git status --short"}),
                ("Read", {"file_path": f"src/{name}.py"})]
        return plan[len(hist)] if len(hist) < len(plan) else None

    model, ex = Model(policy), fs_exec({f"src/m{i}.py": "x\n" for i in range(10)})
    train(jit, model, [f"fix m{i}" for i in range(2)], ex)
    replayed = _replayed(jit, model, ex, "git status --short", [2, 3, 4], task="fix")
    assert any(replayed) == replays, replayed
    if not replays:
        notes = " ".join(r["note"] or "" for r in jit.store.q("SELECT note FROM requests WHERE run_id LIKE 'r%'"))
        assert "repo_tainted" in notes
