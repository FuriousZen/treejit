"""Prototype (scratch only) of the proposed commit-point redesign, installed by monkeypatching.
- full bash ANSI-C decoding and $"..." in the tokenizer (S2)
- commit patterns matched at *program positions* only (unwrapped program, after xargs/find -exec/parallel),
  with git's subcommand located past its global options (S3)
- non-literal program / subcommand words => commit (S2)
- opaque executors (interpreters, shells, scripts, task runners, package-script runners, git aliases,
  gh writes) => commit unless on a known-local runner list (S1)
- curl/wget GET of loopback => not commit (S3)"""
from __future__ import annotations
import re, sys
sys.path.insert(0, "/home/user/treejit/src")
import treejit.shellwords as sw
import treejit.policy as P

_OCT = re.compile(r"[0-7]{1,3}"); _HEX = re.compile(r"[0-9A-Fa-f]{1,2}")
def _ansi_c(body: str) -> str:
    out, i, n = [], 0, len(body)
    simple = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v",
              "\\": "\\", "'": "'", '"': '"', "?": "?"}
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c); i += 1; continue
        d = body[i + 1]
        if d in simple: out.append(simple[d]); i += 2
        elif (m := _OCT.match(body, i + 1)): out.append(chr(int(m.group(), 8) & 0xFF)); i = m.end()
        elif d == "x" and (m := _HEX.match(body, i + 2)): out.append(chr(int(m.group(), 16))); i = m.end()
        elif d in "uU" and (m := re.compile(r"[0-9A-Fa-f]{1,%d}" % (4 if d == "u" else 8)).match(body, i + 2)):
            out.append(chr(int(m.group(), 16))); i = m.end()
        elif d == "c" and i + 2 < n: out.append(chr(ord(body[i + 2]) & 0x1F)); i += 3
        else: out.append("\\" + d); i += 2
    return "".join(out)

def _read_word(s: str, i: int) -> tuple[str, int]:
    out: list[str] = []
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace() or c in sw._OP_CHARS:
            break
        if c == "\\":
            if i + 1 < n:
                if s[i + 1] != "\n":
                    out.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == "'":
            j = s.find("'", i + 1)
            j = n if j < 0 else j
            out.append(s[i + 1 : j])
            i = j + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "'":
            j = i + 2
            while j < n and s[j] != "'":
                j += 2 if s[j] == "\\" else 1
            out.append(_ansi_c(s[i + 2 : j]))
            i = j + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == '"':  # $"..." (locale translation) cooks like "..."
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n and s[j] != '"':
                if s[j] == "\\" and j + 1 < n and s[j + 1] in '"\\$`\n':
                    if s[j + 1] != "\n":
                        buf.append(s[j + 1])
                    j += 2
                elif s[j] == "$" and j + 1 < n and s[j + 1] == "(":
                    k = sw._skip_subst(s, j + 2)
                    buf.append(s[j:k])
                    j = k
                elif s[j] == "`":
                    k = sw._skip_backtick(s, j + 1)
                    buf.append(s[j:k])
                    j = k
                else:
                    buf.append(s[j])
                    j += 1
            out.append("".join(buf))
            i = j + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            k = sw._skip_subst(s, i + 2)
            out.append(s[i:k])
            i = k
            continue
        if c == "`":
            k = sw._skip_backtick(s, i + 1)
            out.append(s[i:k])
            i = k
            continue
        out.append(c)
        i += 1
    return "".join(out), i



sw._read_word = _read_word

# ---- classification
INTERP = re.compile(r"(python|pypy)(\d+(\.\d+)?)?|node|nodejs|perl|ruby|php|lua|Rscript|osascript|pwsh|powershell|tclsh|awk|gawk")
TASK_RUNNERS = {"make", "gmake", "just", "rake", "task", "invoke", "nox", "gradle", "gradlew", "mvn", "ant"}
PKG_RUN = {"npm": {"run", "run-script", "exec", "x", "start"}, "pnpm": {"run", "exec", "dlx", "start"},
           "yarn": {"run", "exec", "dlx", "start"}, "bun": {"run", "x", "start"}, "uv": {"run", "tool"},
           "poetry": {"run"}, "pipx": {"run"}, "cargo": {"run"}, "go": {"run", "generate"}, "deno": {"run", "task"},
           "dotnet": {"run"}}
RUNNER_ONLY = {"npx", "pnpx", "bunx", "uvx"}
LOCAL = {  # known-local runners: not commit points (still not read-only)
    "pytest", "py.test", "tox", "ruff", "mypy", "black", "flake8", "pylint", "isort", "eslint", "prettier", "tsc",
    "jest", "vitest", "mocha", "pyright"}
LOCAL_PY_MODULES = {"pytest", "unittest", "mypy", "ruff", "black", "flake8", "pylint", "isort", "doctest", "compileall",
                    "pip", "venv", "tox", "coverage", "py_compile", "json.tool", "pyright"}
LOCAL_SCRIPTS = re.compile(r"(test|tests|lint|build|check|typecheck|type-check|format|fmt|clean|coverage)([:_-].*)?")
LOCAL_SUBCMDS = {"npm": {"test", "t", "ci", "install", "i"}, "pnpm": {"test", "install", "i"}, "yarn": {"test", "install"},
                 "bun": {"test", "install"}, "cargo": {"test", "build", "check", "clippy", "fmt", "bench", "doc"},
                 "go": {"test", "build", "vet", "fmt"}}
GIT_COMMIT_SUBS = {"push", "send-pack", "http-push", "svn", "p4", "request-pull"}
GIT_BUILTINS = set("""add am annotate apply archive bisect blame branch bundle cat-file check-attr check-ignore check-mailmap
check-ref-format checkout checkout-index cherry cherry-pick citool clean clone column commit commit-graph commit-tree config
count-objects credential describe diff diff-files diff-index diff-tree difftool fast-export fast-import fetch fetch-pack
filter-branch fmt-merge-msg for-each-ref format-patch fsck gc get-tar-commit-id grep hash-object help index-pack init
instaweb interpret-trailers log ls-files ls-remote ls-tree mailinfo mailsplit maintenance merge merge-base merge-file
merge-tree mergetool mktag mktree mv name-rev notes pack-objects pack-refs prune pull push range-diff read-tree rebase
reflog remote repack replace request-pull rerere reset restore rev-list rev-parse revert rm send-email send-pack shortlog
show show-branch show-ref sparse-checkout stash status stripspace submodule switch symbolic-ref tag update-index
update-ref var verify-commit verify-pack verify-tag version whatchanged worktree write-tree http-push svn p4""".split())
GH_READ = {"view", "list", "diff", "checks", "status", "download", "watch"}
LOOPBACK = re.compile(r"^(https?://)?(localhost|127\.\d+\.\d+\.\d+|\[::1\])(:\d+)?(/|$)")
CURL_SAFE = re.compile(r"-[sSfILvkiL]+|--(silent|show-error|fail|head|location|verbose|insecure|include|max-time=?.*|retry=?.*)")

def _git_sub(args):
    i = 0
    while i < len(args):
        a = args[i]
        if a in P._GIT_FLAGS: i += 1
        elif a in P._GIT_ARG: i += 2
        elif a.startswith("--") and "=" in a and a.split("=", 1)[0] in P._GIT_ARG: i += 1
        elif a.startswith("-"): return None, i  # -c / --exec-path / unknown: opaque
        else: return a, i
    return "", i

def classify(words, raws, pats) -> str:
    """'' = not a commit, else the reason it is one."""
    if not words: return ""
    if not P._plain(raws[0]): return "non-literal program"
    w0 = words[0]
    prog = w0.rsplit("/", 1)[-1]
    if "/" in w0 and not P._trusted_path(w0): return "script path"
    rest, rraws = words[1:], raws[1:]
    if prog == "git":
        sub, k = _git_sub(rest)
        if sub is None: return "git -c/--exec-path"
        if sub and not P._plain(rraws[k]): return "non-literal git subcommand"
        if sub in GIT_COMMIT_SUBS: return "git " + sub
        if sub and sub not in GIT_BUILTINS: return "git alias/extension"
        return ""
    if prog == "gh":
        pos = [w for w in rest if not w.startswith("-")]
        if pos[:1] == ["api"]:
            return "gh api write" if any(w.startswith(("-X", "--method", "-f", "-F", "--field", "--raw-field", "--input")) for w in rest) else ""
        return "" if len(pos) >= 2 and pos[1] in GH_READ else "gh write"
    if prog in ("curl", "wget"):
        urls = [w for w in rest if not w.startswith("-")]
        opts = [w for w in rest if w.startswith("-")]
        o_ok = all(CURL_SAFE.fullmatch(o) or o in ("-o", "-O-", "-qO-", "-q") for o in opts)
        dn = [w for w in urls if w != "/dev/null"]
        if dn and all(LOOPBACK.match(u) for u in dn) and o_ok: return ""
        return prog
    for p in pats:  # remaining patterns, anchored at the program
        if P._at(words, 0, p, True) and p[0] not in ("git", "gh", "curl", "wget"): return " ".join(p)
    if prog in LOCAL: return ""
    if INTERP.fullmatch(prog):
        if len(rest) >= 2 and rest[0] == "-m" and rest[1] in LOCAL_PY_MODULES: return ""
        return "interpreter"
    if prog in P._SHELLS: return "shell"  # -c scripts are also checked recursively
    if prog in TASK_RUNNERS:
        tgt = [w for w in rest if not w.startswith("-") and "=" not in w]
        return "" if tgt and all(LOCAL_SCRIPTS.fullmatch(t) for t in tgt) else "task runner"
    if prog in RUNNER_ONLY: return "package runner"
    if prog in PKG_RUN:
        pos = [w for w in rest if not w.startswith("-")]
        if pos and pos[0] in LOCAL_SUBCMDS.get(prog, ()): return ""
        if pos and pos[0] in PKG_RUN[prog]:
            nxt = pos[1:2]
            if prog in ("uv", "poetry") and nxt:
                return classify(pos[1:], pos[1:], pats)
            return "" if nxt and LOCAL_SCRIPTS.fullmatch(nxt[0]) else "package script"
        if prog == "yarn" and pos and pos[0] not in ("add", "remove", "install", "info", "why", "list", "test"):
            return "yarn script"
    return ""

_RUNS_NEXT = {"xargs", "parallel"}
_KEYWORDS = {"{", "}", "!", "if", "then", "elif", "else", "do", "while", "until", "function", "coproc"}
def program_positions(words):
    """program positions: the unwrapped program, the command xargs/parallel/find -exec runs."""
    p = 0
    while p < len(words) and words[p] in _KEYWORDS:  # `{ git "$@"; }`, `if git push; then`, `! git push`
        p += 1
    out = [p + P.unwrap(words[p:])[0]]
    for i, w in enumerate(words):
        b = w.rsplit("/", 1)[-1]
        if b in _RUNS_NEXT:
            j = i + 1
            while j < len(words) and words[j].startswith("-"): j += 1 + (words[j] in ("-n", "-I", "-P", "-L", "-d", "-a", "-s", "-E"))
            out.append(j)
        if w in ("-exec", "-execdir", "-ok", "-okdir"): out.append(i + 1)
    return [p for p in out if p < len(words)]

def _commits2(cmd, pats, cfg, depth=0, certain=True):
    if depth > 4: return "depth"
    extra = set(cfg.extra_readonly_commands)
    cmds = [(w, r, ok and P._program_readonly(w, r, extra)) for w, r, ok in P._commands(cmd)]
    runs_text = any(x.rsplit("/", 1)[-1] in P._RUNS_TEXT for w, _, ro in cmds if not ro for x in w)
    for words, raws, ro in cmds:
        for p in ([] if ro else program_positions(words)):
            why = classify(words[p:], raws[p:], pats)
            if why: return why
        for s, sure in P._scripts(words, raws):
            if sure or runs_text:
                why = _commits2(s, pats, cfg, depth + 1, sure)
                if why: return why
    return ""

def is_commit_point(tool, args, cfg):
    if P._match(tool, cfg.commit_tools): return True
    pats = [p.split() for p in cfg.commit_commands if p.split()]
    return any(k in P.SHELL_KEYS and isinstance(v, str) and bool(_commits2(v, pats, cfg)) for k, v in args.items())

def why(cmd, cfg):
    return _commits2(cmd, [p.split() for p in cfg.commit_commands], cfg)

def install():
    import treejit.builder, treejit.replay
    P.is_commit_point = treejit.builder.is_commit_point = treejit.replay.is_commit_point = is_commit_point
