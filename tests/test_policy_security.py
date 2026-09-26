"""Replay-safety policy: every known bypass of the read-only allowlist and of commit-point
detection, the common read-only commands that must keep replaying, and an end-to-end regression."""

from __future__ import annotations

import pytest
from conftest import Model, calls_of, replayed_ids
from test_engine import fs_exec, train

from treejit.config import Config
from treejit.policy import is_commit_point, is_readonly
from treejit.shellwords import command_heads, unwrap

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
