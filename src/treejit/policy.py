"""Which calls replay may emit on its own: read-only/idempotent tools, and commit points.

A shell command is read-only only if every simple command in it is: the program (after `VAR=val`
assignments and transparent wrappers like `env`, `time`, `timeout`, see `shellwords.unwrap`) must be
on an allowlist, and programs with side-effecting options (`sort -o`, `git diff --output=`, `rg --pre`,
`find -exec`, ...) must pass a per-program check that accepts only what it understands. Anything
unrecognised is not read-only: a false "no" costs one model call, a false "yes" runs a write with no
model call and no human in the loop.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import re

from .config import Config
from .shellwords import _ASSIGN, _skip_subst, program_index, segments, tokenize, unwrap
from .templates import shell_text

# Read-only whatever their arguments (none of them can write a file or run a command).
_ANY_ARGS = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "grep", "egrep", "fgrep", "stat", "which",
    "whereis", "type", "du", "df", "cut", "tr", "jq", "diff", "cmp", "basename", "dirname", "realpath", "readlink",
    "cd", "true", "test", "[", "nl", "column", "id", "whoami", "uname", "printenv", "md5sum", "sha1sum", "sha256sum",
    "ps", "comm",
    "env",  # only reached with no command after it (`unwrap` skips env when it runs one)
}
# Not read-only by default: `sudo` (elevation), `less`/`more` (pagers: LESSOPEN, `-o` log files,
# `!cmd`), `xargs`, `tee`. Operators can opt a program in with `extra_readonly_commands`, which
# skips its argument check entirely.


def _parse(args: list[str], short: str = "", short_arg: str = "", long: set | tuple = (),
           long_arg: set | tuple = ()) -> tuple[list[tuple[str, str | None]], list[str]] | None:
    """GNU-style option parse against an allowlist: (options, positionals), or None if any option is
    not on the list. Long options must be spelled out (getopt would accept `--out` for `--output`)."""
    opts: list[tuple[str, str | None]] = []
    pos: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            pos += args[i + 1 :]
            break
        if a.startswith("--"):
            name, eq, val = a[2:].partition("=")
            if name in long and not eq:
                opts.append((name, None))
            elif name in long_arg:
                if not eq:
                    i += 1
                    if i >= len(args):
                        return None
                    val = args[i]
                opts.append((name, val))
            else:
                return None
        elif a.startswith("-") and a != "-":
            for j, c in enumerate(a[1:], 1):
                if c in short:
                    opts.append((c, None))
                    continue
                if c not in short_arg:
                    return None
                val = a[j + 1 :]
                if not val:
                    i += 1
                    if i >= len(args):
                        return None
                    val = args[i]
                opts.append((c, val))
                break
        else:
            pos.append(a)
        i += 1
    return opts, pos


def _denied_long(args: list[str], names: tuple[str, ...]) -> bool:
    """A long option that is, or abbreviates, one of `names` (getopt_long and git accept prefixes)."""
    for a in args:
        if a == "--":
            return False
        if a.startswith("--"):
            opt = a[2:].split("=", 1)[0]
            if opt and any(n.startswith(opt) for n in names):
                return True
    return False


def _short_has(args: list[str], chars: str) -> bool:
    """A short-option cluster containing any of `chars` (conservative: option values are scanned too)."""
    for a in args:
        if a == "--":
            return False
        if a.startswith("-") and not a.startswith("--") and any(c in a[1:] for c in chars):
            return True
    return False


# ------------------------------------------------------------------ per-program checks

def _sort(args: list[str]) -> bool:  # -o/--output writes, --compress-program runs a program
    return _parse(args, "bdfgiMhnRrVcCmsuz", "ktST",
                  {"ignore-leading-blanks", "dictionary-order", "ignore-case", "general-numeric-sort",
                   "ignore-nonprinting", "month-sort", "human-numeric-sort", "numeric-sort", "random-sort", "reverse",
                   "version-sort", "check", "merge", "stable", "unique", "zero-terminated", "debug"},
                  {"key", "field-separator", "buffer-size", "temporary-directory", "parallel", "sort",
                   "random-source", "check", "files0-from"}) is not None


def _uniq(args: list[str]) -> bool:  # `uniq IN OUT` writes OUT
    p = _parse(args, "cdDiuz", "fsw", {"count", "repeated", "ignore-case", "unique", "zero-terminated", "all-repeated",
                                       "group"},
               {"skip-fields", "skip-chars", "check-chars", "all-repeated", "group"})
    return p is not None and len(p[1]) <= 1


def _date(args: list[str]) -> bool:  # -s/--set and `date MMDDhhmm` set the clock
    p = _parse(args, "uRI", "dr", {"utc", "universal", "rfc-email", "iso-8601", "debug"},
               {"date", "reference", "rfc-3339", "iso-8601"})
    return p is not None and all(a.startswith("+") for a in p[1])


def _hostname(args: list[str]) -> bool:  # `hostname NAME`, -F/-b set the hostname
    p = _parse(args, "aAdfiIsvy", "", {"alias", "all-fqdns", "all-ip-addresses", "domain", "fqdn", "long",
                                        "ip-address", "short", "verbose", "yp", "nis"})
    return p is not None and not p[1]


def _file(args: list[str]) -> bool:  # -C/--compile writes a .mgc, -p resets atimes
    return _parse(args, "bcdEhikLlNnrsvzZ0", "eFfmP",
                  {"brief", "checking-printout", "exclude-quiet", "no-dereference", "mime", "apple", "extension",
                   "mime-type", "mime-encoding", "keep-going", "list", "dereference", "no-pad", "no-buffer", "raw",
                   "special-files", "uncompress", "uncompress-noreport", "print0", "debug"},
                  {"exclude", "separator", "files-from", "magic-file", "parameter"}) is not None


def _tree(args: list[str]) -> bool:  # -o FILE writes, -R re-runs tree with -o in every directory
    return not _short_has(args, "oR")


def _find(args: list[str]) -> bool:
    # find's side-effecting primaries are a closed, documented set (GNU and BSD); everything else
    # is a test, an option, or prints to stdout.
    bad = ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0", "-fprintf")
    return not any(a in bad or a.startswith(("-exec", "-ok", "-fprint", "-fls", "-delete")) for a in args)


def _rg(args: list[str]) -> bool:  # --pre CMD / --hostname-bin CMD run programs
    return not any(a.split("=", 1)[0] in ("--pre", "--hostname-bin") for a in args)


def _ag(args: list[str]) -> bool:
    return not _denied_long(args, ("pager",))


def _fd(args: list[str]) -> bool:  # -x/--exec, -X/--exec-batch
    return not _short_has(args, "xX") and not any(a.split("=", 1)[0] in ("--exec", "--exec-batch") for a in args)


def _bat(args: list[str]) -> bool:  # a pager runs a program; `bat cache --build` writes
    return (not any(a.split("=", 1)[0] in ("--pager", "--paging") for a in args)
            and next((a for a in args if not a.startswith("-")), None) != "cache")


def _yq(args: list[str]) -> bool:  # -i/--inplace (both yqs), -s/--split-exp writes files (mikefarah)
    return not _short_has(args, "is") and not _denied_long(args, ("inplace", "in-place", "split-exp"))


def _command(args: list[str]) -> bool:  # only reached as `command -v/-V NAME` (see unwrap)
    return _parse(args, "pvV") is not None


_SED_SIMPLE = set("pPdDnNgGhHxz={}")


def _sed_script(s: str) -> bool:
    """A conservative sed subset: [addr[,addr]][!] then p P d D n N g G h H x z = { } q Q l or s///
    with flags g p i I m M N. No w/W/r/R/e (write, read, execute), no a/i/c/y/labels/comments."""
    i, n = 0, len(s)

    def ws(i: int) -> int:
        while i < n and s[i] in " \t":
            i += 1
        return i

    def delimited(i: int) -> int:  # s[i] is the delimiter; index past the closing one, or -1
        d = s[i]
        if d in "\n\\":
            return -1
        i += 1
        while i < n:
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == "\n":
                return -1
            if s[i] == d:
                return i + 1
            i += 1
        return -1

    def digits(i: int) -> int:
        while i < n and s[i].isdigit():
            i += 1
        return i

    def address(i: int) -> int:  # index past an address (== i if none), -1 if malformed
        if i < n and s[i].isdigit():
            i = digits(i)
            if i < n and s[i] == "~":
                j = digits(i + 1)
                return -1 if j == i + 1 else j
            return i
        if i < n and s[i] == "$":
            return i + 1
        if i < n and s[i] in "/\\":
            if s[i] == "\\":
                i += 1
                if i >= n:
                    return -1
            i = delimited(i)
            while 0 <= i < n and s[i] in "IM":
                i += 1
        return i

    while True:
        while i < n and s[i] in " \t;\n":
            i += 1
        if i >= n:
            return True
        j = address(i)
        if j < 0:
            return False
        if j != i:
            i = ws(j)
            if i < n and s[i] == ",":
                i = ws(i + 1)
                if i < n and s[i] in "+~":
                    j = digits(i + 1)
                    if j == i + 1:
                        return False
                else:
                    j = address(i)
                    if j <= i:
                        return False
                i = ws(j)
        if i < n and s[i] == "!":
            i = ws(i + 1)
        if i >= n:
            return False
        c = s[i]
        if c in _SED_SIMPLE:
            i += 1
        elif c in "qQl":
            i = digits(ws(i + 1))
        elif c == "s" and i + 1 < n:
            j = delimited(i + 1)
            j = delimited(j - 1) if j > 0 else -1  # the replacement shares the pattern's closing delimiter
            if j < 0:
                return False
            i = j
            while i < n and s[i] in "gpiImM0123456789":
                i += 1
        else:
            return False
        i = ws(i)
        if i < n and s[i] not in ";\n}":
            return False


def _sed(args: list[str]) -> bool:  # -i/--in-place, -f FILE, and w/W/e/r in the script
    p = _parse(args, "nErsuz", "el", {"quiet", "silent", "regexp-extended", "separate", "unbuffered", "null-data",
                                       "posix", "debug", "sandbox", "binary"},
               {"expression", "line-length"})
    if p is None:
        return False
    opts, pos = p
    scripts = [v for k, v in opts if k in ("e", "expression")]
    if not scripts:
        if not pos:
            return False
        scripts = [pos[0]]
    return all(v is not None and _sed_script(v) for v in scripts)


_AWK_BAD = re.compile(r"system|getline|extension|@|\||>")


def _awk(args: list[str]) -> bool:
    # awk programs can run commands (`system()`, `print | "cmd"`, `"cmd" | getline`, gawk `@f()`
    # indirect calls, `@load`/`extension()`) and write files (`print > "f"`). Accept only programs
    # with none of those tokens; `||` and `>=` are fine.
    p = _parse(args, "", "Fv", (), {"field-separator", "assign"})
    if p is None or not p[1]:
        return False
    prog = p[1][0].replace("||", "").replace(">=", "")
    return not _AWK_BAD.search(prog)


# git: global options before the subcommand. `-c`/`--config-env` (core.pager, core.fsmonitor,
# alias.*, diff.external all run programs) and `--exec-path` are not on the list.
_GIT_FLAGS = {"--no-pager", "-P", "-p", "--paginate", "--no-optional-locks", "--literal-pathspecs",
              "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs", "--no-replace-objects", "--bare"}
_GIT_ARG = {"-C", "--git-dir", "--work-tree", "--namespace"}
_GIT_PLAIN = {"status", "diff", "log", "show", "rev-parse", "ls-files", "ls-tree", "blame", "describe", "shortlog",
              "cat-file", "rev-list", "merge-base", "grep"}
_GIT_LIST_BRANCH = {"list", "l", "contains", "no-contains", "merged", "no-merged", "points-at", "show-current"}
_GIT_LIST_TAG = {"list", "l", "v", "verify", "contains", "no-contains", "merged", "no-merged", "points-at"}
_GIT_CONFIG_READ = {"get", "get-all", "get-regexp", "get-urlmatch", "list", "l"}


def _git_split(args: list[str]) -> tuple[str, list[str]] | None:
    i = 0
    while i < len(args):
        a = args[i]
        if a in _GIT_FLAGS:
            i += 1
        elif a in _GIT_ARG:
            i += 2
        elif a.startswith("--") and "=" in a and a.split("=", 1)[0] in _GIT_ARG:
            i += 1
        elif a.startswith("-"):
            return None
        else:
            return a, args[i + 1 :]
    return None


def _git(args: list[str]) -> bool:
    sp = _git_split(args)
    if sp is None:
        return False
    sub, rest = sp
    # `--output=FILE` (diff/log/show) writes; `git grep -O/--open-files-in-pager` runs a program
    if _denied_long(rest, ("output", "open-files-in-pager")) or (sub == "grep" and _short_has(rest, "O")):
        return False
    if sub in _GIT_PLAIN:
        return True
    if sub == "stash":
        return bool(rest) and rest[0] in ("list", "show")
    if sub == "reflog":  # `expire`/`delete`/`drop` rewrite reflogs
        return next((w for w in rest if not w.startswith("-")), "show") in ("show", "list", "exists")
    if sub == "remote":
        while rest and rest[0] in ("-v", "--verbose"):
            rest = rest[1:]
        return not rest or rest[0] in ("show", "get-url")
    if sub == "branch":  # positionals create a branch unless a listing option is given
        both = {"contains", "no-contains", "merged", "no-merged"}
        p = _parse(rest, "arvil", "",
                   {"list", "all", "remotes", "verbose", "show-current", "ignore-case", "color", "no-color",
                    "column", "no-column", "no-abbrev", "omit-empty"} | both,
                   {"points-at", "sort", "format", "abbrev", "color", "column"} | both)
        return p is not None and (not p[1] or any(k in _GIT_LIST_BRANCH for k, _ in p[0]))
    if sub == "tag":  # positionals create a tag unless listing or verifying
        both = {"contains", "no-contains", "merged", "no-merged"}
        p = _parse(rest, "lin0123456789v", "",
                   {"list", "ignore-case", "verify", "color", "no-color", "column", "no-column", "omit-empty"} | both,
                   {"sort", "format", "points-at", "color", "column"} | both)
        return p is not None and (not p[1] or any(k in _GIT_LIST_TAG for k, _ in p[0]))
    if sub == "config":  # `git config KEY` reads, `git config KEY VALUE` writes
        p = _parse(rest, "lz", "f",
                   {"list", "get", "get-all", "get-regexp", "get-urlmatch", "show-origin", "show-scope", "name-only",
                    "global", "system", "local", "worktree", "null", "includes", "no-includes", "bool", "int",
                    "bool-or-int", "path", "expiry-date", "fixed-value", "all", "regexp", "show-names"},
                   {"file", "blob", "type", "default", "value", "url"})
        if p is None:
            return False
        opts, pos = p
        if pos and pos[0] in ("get", "list"):
            return True
        # a key always has a dot, so `git config edit` / `git config set` aren't mistaken for reads
        return any(k in _GIT_CONFIG_READ for k, _ in opts) or (len(pos) == 1 and "." in pos[0])
    return False


_CHECKED = {
    "sort": _sort, "uniq": _uniq, "date": _date, "hostname": _hostname, "file": _file, "tree": _tree,
    "find": _find, "rg": _rg, "ag": _ag, "fd": _fd, "bat": _bat, "yq": _yq, "command": _command, "sed": _sed,
    "awk": _awk, "git": _git,
}
READONLY_PROGRAMS = _ANY_ARGS | set(_CHECKED)

# ------------------------------------------------------------------ words and simple commands

_REDIRECTS = {">", ">>", ">|", "&>", ">&", "<", "<<", "<<-", "<<<", "<&"}
_SINKS = {"/dev/null", "/dev/stdout", "/dev/stderr"}
_FD = re.compile(r"\d+|\{[A-Za-z_]\w*\}")
_BRACE = re.compile(r"\{[^{}]*(,|\.\.)[^{}]*\}")
_STD_BIN = {"/bin", "/usr/bin", "/usr/local/bin", "/sbin", "/usr/sbin", "/opt/homebrew/bin"}
_SAFE_VARS = {"LANG", "LANGUAGE", "TZ", "NO_COLOR", "CLICOLOR", "CLICOLOR_FORCE", "FORCE_COLOR", "TERM", "COLUMNS",
              "LINES", "GIT_PAGER", "PAGER"}


def _scan(raw: str) -> tuple[str, bool, bool]:
    """(the word's unquoted characters, quoted/escaped ones replaced by '_'; whether it expands `$`
    or backticks; whether it contains a command substitution)."""
    out: list[str] = []
    expands = subst = False
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == "\\":
            out.append("_")
            i += 2
            continue
        if c == "'":
            j = raw.find("'", i + 1)
            j = n if j < 0 else j
            out.append("_")
            i = j + 1
            continue
        if c == "$" and raw.startswith("$'", i):
            j = i + 2
            while j < n and raw[j] != "'":
                # escapes (`$'\055o'` is `-o`) are data the checks shouldn't have to decode
                expands = expands or raw[j] == "\\"
                j += 2 if raw[j] == "\\" else 1
            out.append("_")
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n and raw[j] != '"':
                if raw[j] == "\\":
                    j += 2
                    continue
                if raw[j] in "$`":
                    expands = True
                    subst = subst or raw[j] == "`" or raw.startswith("$(", j)
                j += 1
            out.append("_")
            i = j + 1
            continue
        if c in "$`":
            expands = True
            subst = subst or c == "`" or raw.startswith("$(", i)
        out.append(c)
        i += 1
    return "".join(out), expands, subst


def _plain(raw: str) -> bool:
    """The word is exactly what it looks like: no expansion, glob or brace expansion."""
    text, expands, _ = _scan(raw)
    return not expands and not any(ch in text for ch in "*?[") and not _BRACE.search(text)


def _trusted_path(word: str) -> bool:
    """A bare program name (found on PATH) or one in a standard bin dir, not `./ls`."""
    return "/" not in word or word.rsplit("/", 1)[0] in _STD_BIN


def _safe_assignment(word: str) -> bool:
    name, _, val = word.partition("=")
    if name in ("PAGER", "GIT_PAGER"):
        return val in ("", "cat")
    return name in _SAFE_VARS or name.startswith("LC_")


def _commands(cmd: str, toks: list | None = None) -> list[tuple[list[str], list[str], bool]]:
    """Per simple command: cooked words and their raw text (redirections and their targets removed),
    and whether every redirection only reads, duplicates a descriptor, or writes to /dev/null."""
    toks = tokenize(cmd) if toks is None else toks
    out = []
    for seg in segments(toks):
        words: list[str] = []
        raws: list[str] = []
        ok, k = True, 0
        while k < len(seg):
            t = seg[k]
            nxt = seg[k + 1] if k + 1 < len(seg) else None
            if t.op:  # every non-separator operator is a redirection
                target = nxt if nxt is not None and not nxt.op else None
                if target is None:  # `<(...)`/`>(...)` (the paren is a separator), or a dangling operator
                    ok = False
                    k += 1
                    continue
                raw = cmd[target.start : target.end]
                if t.val in (">", ">>", ">|", "&>"):
                    ok = ok and target.val in _SINKS and _plain(raw)
                elif t.val in (">&", "<&"):
                    ok = ok and (target.val.isdigit() or target.val == "-" or target.val in _SINKS) and _plain(raw)
                k += 2
                continue
            if nxt is not None and nxt.op and nxt.val in _REDIRECTS and nxt.start == t.end and _FD.fullmatch(
                    cmd[t.start : t.end]):
                k += 1  # the descriptor of `2>`, `{fd}>`
                continue
            words.append(t.val)
            raws.append(cmd[t.start : t.end])
            k += 1
        out.append((words, raws, ok))
    return out


def _program_readonly(words: list[str], raws: list[str], extra: set[str], trust_git: bool = True) -> bool:
    start, clean, wrappers = unwrap(words)
    if not clean:
        return False
    for k in range(start):  # assignments, wrappers and their options
        if not _plain(raws[k]) or (_ASSIGN.match(words[k]) and not _safe_assignment(words[k])):
            return False
    if not all(_trusted_path(words[k]) for k in wrappers):
        return False
    if start == len(words):
        return True
    if not (_plain(raws[start]) or words[start] == "[") or not _trusted_path(words[start]):
        return False
    prog = words[start].rsplit("/", 1)[-1]
    if prog == "git" and not trust_git:
        # git reads run programs named by the repository's config and attributes (core.fsmonitor,
        # diff.external, textconv, clean filters): read-only only in a checkout we trust
        return False
    if prog in extra or prog in _ANY_ARGS:
        return True
    check = _CHECKED.get(prog)
    # the checks read literal words: a `$VAR` or glob could expand into any option (a file named `--pre=x`)
    return check is not None and all(_plain(r) for r in raws[start + 1 :]) and check(words[start + 1 :])


def shell_readonly(cmd: str, cfg: Config) -> bool:
    # the tokenizer splits words on any str.isspace(); bash only on space/tab/newline
    if any(c.isspace() and c not in " \t\n" for c in cmd):
        return False
    toks = tokenize(cmd)
    if any(not t.op and _scan(cmd[t.start : t.end])[2] for t in toks):
        return False
    extra = set(cfg.extra_readonly_commands)
    trust = cfg.trust_repo_config
    return all(ok and _program_readonly(words, raws, extra, trust) for words, raws, ok in _commands(cmd, toks))


def _match(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _shell_args(args: dict) -> list[str]:
    """The command text of every shell argument: a string, or an argv list (`["bash", "-lc", S]` -> S,
    any other argv -> shlex.join(argv); see templates.shell_text)."""
    return [st[0] for st in (shell_text(k, v) for k, v in args.items()) if st is not None]


def is_readonly(tool: str, args: dict, cfg: Config) -> bool:
    shell = _shell_args(args)
    if tool in cfg.shell_tools or (shell and not _match(tool, cfg.replay_tools)):
        return bool(shell) and all(shell_readonly(c, cfg) for c in shell)
    return _match(tool, cfg.replay_tools)


# ------------------------------------------------------------------ commit points
#
# A commit point is a call whose effects leave the machine or can't be taken back (`git push`, `curl`,
# `send_*`), or whose effects can't be seen from the call at all: an interpreter, a shell, a script, a
# task-runner target or package script, a git alias, a `gh` command that isn't a read, a cloud CLI
# write. `commit_reason` says why ('' = not a commit point). The builder stores it per (node, edge),
# and `replay.materialize` rejects a render whose reason differs, so a value the model fills in (T3)
# can't change it. Patterns only match where a program runs: the unwrapped program of each simple
# command, the command `xargs`/`find -exec`/`watch`/... run, and nested scripts (`sh -c`, `$(...)`).

_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "mksh", "fish", "su", "runuser", "tcsh", "csh", "ash", "busybox"}
_RUNS_TEXT = _SHELLS | {"watch", "eval", "parallel"}
_SCRIPTY = re.compile(r"\s|[;&|]")
_INTERP = re.compile(
    r"(python|pypy)(\d+(\.\d+)*)?|node(js)?|perl(\d+(\.\d+)*)?|ruby|php(\d+(\.\d+)*)?|lua(jit|\d+(\.\d+)*)?|R|"
    r"Rscript|osascript|pwsh|powershell|tclsh|wish|julia|java|groovy|kotlin|scala|elixir|erl|escript|guile|racket|"
    r"sbcl|swift|[gmn]?awk|tsx|ts-node|expect")
_SCRIPT_EXT = (".sh", ".bash", ".zsh", ".py", ".rb", ".pl", ".js", ".mjs", ".cjs", ".ts", ".php", ".lua", ".ps1")
_TOOL_BIN = re.compile(r"(.*/)?(node_modules/\.bin|\.?venv/bin|env/bin|\.tox/[^/]+/bin)")

# Known-local runners: they run the project's tests, linters or build. Not read-only (they write
# caches and build output), but not commit points either.
_LOCAL_TOOLS = {"pytest", "py.test", "tox", "ruff", "mypy", "black", "flake8", "pylint", "isort", "eslint", "prettier",
                "tsc", "jest", "vitest", "mocha", "pyright", "biome", "coverage", "stylelint", "rustfmt", "gofmt",
                "golangci-lint", "shellcheck", "yamllint", "markdownlint", "playwright", "unittest"}
_LOCAL_PY_MODULES = {"pytest", "unittest", "mypy", "ruff", "black", "flake8", "pylint", "isort", "doctest", "compileall",
                     "pip", "venv", "tox", "coverage", "py_compile", "json.tool", "pyright", "build"}
# task and package-script names made only of these words are local: `test`, `test:unit`, `lint-fix`, `build`
_LOCAL_WORDS = {"test", "tests", "unit", "integration", "e2e", "smoke", "lint", "linters", "build", "check", "checks",
                "typecheck", "types", "type", "format", "fmt", "clean", "coverage", "cov", "compile", "package", "verify",
                "validate", "install", "ci", "all", "dev", "debug", "watch", "fast", "quick", "local", "style", "fix",
                "doc", "docs", "html", "assemble", "vet", "bench", "benchmark", "benchmarks", "deps", "py", "js", "ts",
                "mypy", "ruff", "pytest", "eslint", "prettier", "jest", "vitest", "flake8", "pylint", "classes", "jar",
                "spec", "specs"}
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _local_name(name: str, raw: str | None = None) -> bool:
    """A task or script name made only of local words (`test`, `test:unit`, `lint-fix`, `testDebugUnitTest`)."""
    if not name or (raw is not None and not _plain(raw)) or not re.fullmatch(r"[\w:./-]+", name):
        return False
    for seg in re.split(r"[:_./-]+", name):
        if seg and not (seg.lower() in _LOCAL_WORDS or seg.isdigit()
                        or all(p.lower() in _LOCAL_WORDS or p.isdigit() for p in _CAMEL.findall(seg))):
            return False
    return True


def _positionals(args: list[str], raws: list[str], values: set = frozenset()) -> list[tuple[str, str]]:
    """Non-option words (and their raw text), skipping the value of an option in `values`."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == "--":
            out += list(zip(args[i + 1 :], raws[i + 1 :]))
            break
        if a.startswith("-") and a != "-":
            i += 2 if a in values else 1
            continue
        out.append((a, raws[i]))
        i += 1
    return out


# ---- task runners: local only when every target is a local name

_TASK_RUNNERS = {"make", "gmake", "just", "rake", "task", "invoke", "inv", "nox", "gradle", "gradlew", "mvn", "mvnw", "ant"}
_TASK_VALUES = {"-C", "-f", "-I", "-o", "-W", "-d", "-p", "-b", "-x", "-pl", "-rf", "--directory", "--file", "--makefile",
                "--justfile", "--working-directory", "--noxfile", "--python", "--project-dir", "--build-file",
                "--include-dir", "--old-file", "--what-if", "--new-file"}


def _task_reasons(prog: str, args: list[str], raws: list[str]) -> list[str]:
    if any(a.startswith(("--eval", "-E")) for a in args):
        return ["task runner"]  # `make --eval=...` defines rules on the command line
    words, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("-j", "-l") and i + 1 < len(args) and args[i + 1].isdigit():
            i += 2
            continue
        if a.startswith("-"):
            i += 2 if a in _TASK_VALUES else 1
            continue
        if _ASSIGN.match(a):
            return ["task runner"]  # `make test PYTEST='...'` overrides what the recipe runs
        words.append((a, raws[i]))
        i += 1
    if prog in ("just", "task"):
        words = words[:1]  # the rest are the recipe's arguments
    return [] if words and all(_local_name(w, r) for w, r in words) else ["task runner"]


# ---- package managers

_PKG_VALUES = {"--prefix", "-w", "--workspace", "--registry", "--tag", "--userconfig", "--cache", "--loglevel", "--otp",
               "--filter", "-F", "--cwd", "-C", "--dir", "--with", "--with-requirements", "--python", "-p", "--package",
               "--extra", "--group", "--env-file", "--directory", "--project", "--index", "--index-url"}
_PKG_REMOTE = {"publish", "unpublish", "deprecate", "dist-tag", "dist-tags", "owner", "author", "access", "team", "org",
               "token", "star", "unstar", "hook", "tag", "yank", "push", "deploy"}
_PKG_LOCAL = {  # subcommands with local effects (installs run lifecycle scripts; accepted like `pip install`)
    "npm": {"install", "i", "in", "ci", "add", "uninstall", "remove", "rm", "un", "update", "up", "upgrade", "ls", "list",
            "ll", "la", "outdated", "audit", "view", "info", "show", "v", "search", "pack", "version", "init", "link", "ln",
            "config", "get", "set", "cache", "doctor", "explain", "why", "fund", "help", "prefix", "root", "bin", "query",
            "sbom", "dedupe", "prune", "rebuild", "shrinkwrap", "diff", "pkg", "completion", "ping", "whoami", "docs",
            "repo", "bugs", "home", "login", "logout", "install-test", "it", "install-ci-test", "cit"},
    "pnpm": {"install", "i", "add", "update", "up", "upgrade", "remove", "rm", "uninstall", "un", "link", "ln", "unlink",
             "import", "rebuild", "rb", "prune", "fetch", "patch", "patch-commit", "audit", "list", "ls", "ll", "outdated",
             "why", "licenses", "store", "root", "bin", "config", "c", "get", "set", "init", "pack", "env", "setup",
             "ci", "dedupe", "install-test", "it", "help", "login", "logout", "view", "info"},
    "yarn": {"add", "remove", "install", "info", "why", "list", "upgrade", "up", "upgrade-interactive", "init", "config",
             "cache", "audit", "outdated", "pack", "link", "unlink", "bin", "version", "set", "plugin", "constraints",
             "dedupe", "explain", "patch", "rebuild", "stage", "unplug", "global", "check", "licenses", "import",
             "login", "logout", "help", "workspaces"},
    "bun": {"install", "i", "add", "a", "remove", "rm", "update", "link", "unlink", "pm", "outdated", "build", "init",
            "upgrade", "audit", "info", "patch", "help"},
    "cargo": {"build", "b", "check", "c", "clean", "doc", "d", "new", "init", "add", "remove", "rm", "test", "t", "bench",
              "update", "search", "install", "uninstall", "fmt", "clippy", "tree", "metadata", "fetch", "vendor",
              "generate-lockfile", "locate-project", "pkgid", "verify-project", "version", "help", "fix", "rustc",
              "rustdoc", "package", "login", "logout", "report", "config", "info", "nextest", "llvm-cov", "tarpaulin",
              "deny", "audit", "outdated", "machete", "udeps", "hack", "insta", "expand", "miri"},
    "go": {"build", "test", "vet", "fmt", "mod", "get", "install", "list", "env", "version", "doc", "work", "clean", "tool",
           "help", "bug", "telemetry", "fix"},
    "deno": {"test", "lint", "fmt", "check", "compile", "bundle", "info", "doc", "cache", "install", "uninstall", "add",
             "remove", "outdated", "init", "upgrade", "bench", "coverage", "types", "completions", "help"},
    "dotnet": {"test", "build", "restore", "format", "clean", "pack", "new", "add", "remove", "list", "sln", "publish",
               "msbuild", "help", "--info", "--version", "workload", "tool", "dev-certs", "user-secrets"},
}
_PKG_SCRIPT_SHORTHAND = {"pnpm", "yarn", "bun"}   # `yarn build` runs the `build` script
_RUNNER_ONLY = {"npx", "pnpx", "bunx", "uvx"}
_CMD_RUNNERS = {"uv": "run", "poetry": "run", "pipenv": "run", "rye": "run", "conda": "run", "bundle": "exec",
                "pdm": "run", "hatch": "run", "pixi": "run"}   # `uv run CMD ...`: CMD is the program


def _runner_pkg(pos: list[tuple[str, str]]) -> list[str]:
    """`npx jest`, `pnpm dlx x`, `uv tool run ruff`: fetch and run a package."""
    return [] if pos and _plain(pos[0][1]) and pos[0][0] in _LOCAL_TOOLS else ["package runner"]


def _pkg_reasons(prog: str, args: list[str], raws: list[str], ctx: "_Ctx") -> list[str]:
    pos = _positionals(args, raws, _PKG_VALUES)
    if prog in _RUNNER_ONLY:
        return _runner_pkg(pos)
    if not pos:
        return []
    sub, sraw = pos[0]
    if not _plain(sraw):
        return ["non-literal subcommand"]
    if prog in _CMD_RUNNERS:
        if sub == "tool" and len(pos) > 1 and pos[1][0] == "run":
            return _runner_pkg(pos[2:])
        if sub != _CMD_RUNNERS[prog]:
            return []
        # the command after `run`: classified like any program (`uv run pytest`, `uv run ./x.sh`)
        k = args.index(sub) + 1
        while k < len(args) and args[k].startswith("-") and args[k] != "--":
            k += 2 if args[k] in _PKG_VALUES else 1
        k += k < len(args) and args[k] == "--"
        if k >= len(args):
            return []
        if prog in ("pdm", "hatch", "pixi") and _local_name(args[k], raws[k]):
            return []  # a script defined in the project file
        why = _prog_reasons(args, raws, k, ctx)
        if not why and prog in ("pdm", "hatch", "pixi") and not (
                args[k] in _LOCAL_TOOLS or _INTERP.fullmatch(args[k].rsplit("/", 1)[-1])):
            return ["package script"]
        return why
    if prog == "pipx":
        return _runner_pkg(pos[1:]) if sub == "run" else []
    if prog == "go":
        return [f"go {sub}"] if sub in ("run", "generate") else []
    if prog == "cargo":
        if sub in ("run", "r"):
            return ["cargo run"]
        if sub in _PKG_REMOTE:
            return [f"cargo {sub}"]
        return [] if sub in _PKG_LOCAL["cargo"] else ["cargo extension"]
    if prog == "deno":
        if sub == "task":
            return [] if len(pos) > 1 and _local_name(*pos[1]) else ["package script"]
        if sub in _PKG_REMOTE:
            return [f"deno {sub}"]
        return [] if sub in _PKG_LOCAL["deno"] else ["deno run"]
    if prog == "dotnet":
        if sub == "nuget" and len(pos) > 1 and pos[1][0] in ("push", "delete"):
            return ["dotnet nuget push"]
        return [] if sub in _PKG_LOCAL["dotnet"] else ["dotnet run"]
    if prog == "gem":
        return ["gem push"] if sub in ("push", "yank", "owner") else []
    if prog == "bundle":
        return []
    # npm, pnpm, yarn, bun
    if sub in ("run", "run-script", "rum", "urn"):
        return [] if len(pos) < 2 or _local_name(*pos[1]) else ["package script"]
    if sub in ("test", "t", "tst"):
        return []
    if sub in ("exec", "x", "dlx", "create", "node"):
        return _runner_pkg(pos[1:])
    if sub in _PKG_REMOTE or (prog == "yarn" and sub == "npm"):
        return [f"{prog} {sub}"]
    if sub in _PKG_LOCAL.get(prog, ()):
        return []
    if prog == "yarn" and sub == "workspace":  # `yarn workspace NAME SCRIPT`
        return [] if len(pos) < 3 or _local_name(*pos[2]) or pos[2][0] in _PKG_LOCAL["yarn"] else ["package script"]
    if prog in _PKG_SCRIPT_SHORTHAND:
        return [] if _local_name(sub, sraw) else ["package script"]
    return ["package command"]


_PKG = {"npm", "pnpm", "yarn", "bun", "cargo", "go", "deno", "dotnet", "gem", "pipx"} | _RUNNER_ONLY | set(_CMD_RUNNERS)

# ---- interpreters


def _interp_reasons(args: list[str], raws: list[str], ctx: "_Ctx", name: str) -> list[str]:
    if args and all(a in ("-V", "--version", "-version", "-v", "-h", "--help") for a in args):
        return []
    if name.startswith(("python", "pypy")):
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-W", "-X"):
                i += 2
            elif re.fullmatch(r"-[bBdEiIOPqsSuvx]+", a):
                i += 1
            elif a == "-m" and i + 1 < len(args):
                mod = args[i + 1]
                if not _plain(raws[i + 1]):
                    return ["interpreter"]
                if mod in _LOCAL_PY_MODULES:
                    return []
                # `python -m twine upload`: the module is the program
                return _prog_reasons(args, raws, i + 1, ctx) or ["interpreter"]
            else:
                break
    return ["interpreter"]


# ---- git

_GIT_COMMIT_SUBS = {"push", "send-pack", "http-push", "svn", "p4", "send-email", "imap-send", "cvsexportcommit"}
_GIT_BUILTINS = set("""
add am annotate apply archive backfill bisect blame branch bugreport bundle cat-file check-attr check-ignore check-mailmap
check-ref-format checkout checkout-index cherry cherry-pick citool clean clone column commit commit-graph commit-tree config
count-objects credential credential-cache credential-store daemon describe diagnose diff diff-files diff-index diff-tree
difftool fast-export fast-import fetch fetch-pack filter-branch fmt-merge-msg for-each-ref for-each-repo format-patch fsck
gc get-tar-commit-id grep gui hash-object help hook index-pack init init-db instaweb interpret-trailers log ls-files
ls-remote ls-tree mailinfo mailsplit maintenance merge merge-base merge-file merge-index merge-one-file merge-tree
mergetool mktag mktree multi-pack-index mv name-rev notes pack-objects pack-redundant pack-refs patch-id prune
prune-packed pull range-diff read-tree rebase receive-pack reflog refs remote repack replace replay rerere reset restore
rev-list rev-parse revert rm scalar shortlog show show-branch show-index show-ref sparse-checkout stage stash status
stripspace submodule switch symbolic-ref tag unpack-file unpack-objects update-index update-ref update-server-info
upload-archive upload-pack var verify-commit verify-pack verify-tag version whatchanged worktree write-tree""".split())
_GIT_GLOBAL_FLAGS = _GIT_FLAGS | {"--no-advice", "--no-lazy-fetch", "--no-optional-locks"}
_GIT_GLOBAL_ARGS = _GIT_ARG | {"--super-prefix", "--attr-source", "--list-cmds"}
_GIT_INFO = {"--version", "--help", "-h", "-v", "--html-path", "--man-path", "--info-path", "--exec-path"}
# config keys that run nothing and point git at nothing else (for `git -c` and the per-run taint)
_GIT_SAFE_KEY = re.compile(
    r"(user|author|committer|color|advice|column|i18n)\.[\w.-]+|init\.defaultbranch|"
    r"core\.(quotepath|autocrlf|safecrlf|filemode|ignorecase|abbrev|precomposeunicode|longpaths|eol|whitespace)|"
    r"(commit|tag)\.gpgsign|push\.(default|autosetupremote)|pull\.(rebase|ff)|merge\.(conflictstyle|ff)|"
    r"rebase\.(autosquash|autostash|updaterefs)|fetch\.prune|diff\.(renames|algorithm|mnemonicprefix|noprefix|context|"
    r"colormoved|relative)|log\.(date|decorate|abbrevcommit|follow)|format\.pretty|gc\.auto|maintenance\.auto|"
    r"status\.(short|branch|showuntrackedfiles)|branch\.(autosetupmerge|sort)|protocol\.version|help\.autocorrect",
    re.I)


def _git_key_safe(kv: str) -> bool:
    key, _, val = kv.partition("=")
    if key.lower() in ("core.pager", "pager.log", "pager.diff", "pager.show", "pager.status", "pager.branch"):
        return val in ("", "cat")
    return bool(_GIT_SAFE_KEY.fullmatch(key))


def _git_global(args: list[str], raws: list[str]) -> tuple[int | None, list[str]]:
    """Walk git's global options: (index of the subcommand or None, reasons found on the way)."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in _GIT_GLOBAL_FLAGS:
            i += 1
        elif a in _GIT_GLOBAL_ARGS:
            i += 2
        elif a.startswith("--") and "=" in a and a.split("=", 1)[0] in _GIT_GLOBAL_ARGS:
            i += 1
        elif a == "-c" or (a.startswith("-c") and not a.startswith("--")):
            kv, raw = (a[2:], raws[i]) if len(a) > 2 else (args[i + 1], raws[i + 1]) if i + 1 < len(args) else ("", "")
            if not (_plain(raw) and _git_key_safe(kv)):
                out.append("git -c " + kv.split("=", 1)[0])
            i += 1 if len(a) > 2 else 2
        elif a.startswith(("--config-env", "--exec-path=")):
            out.append("git " + a.split("=", 1)[0])
            i += 2 if a == "--config-env" else 1
        elif a in _GIT_INFO:
            return None, out
        elif a.startswith("-"):
            out.append("git option " + a.split("=", 1)[0])
            return None, out
        else:
            return i, out
    return None, out


def _git_reasons(args: list[str], raws: list[str], ctx: "_Ctx") -> list[str]:
    i, out = _git_global(args, raws)
    if i is None:
        return out
    sub = args[i]
    if not _plain(raws[i]):
        return out + ["non-literal git subcommand"]
    rest, rraws = args[i + 1 :], raws[i + 1 :]
    if sub in _GIT_COMMIT_SUBS:
        out.append("git " + sub)
    elif sub not in _GIT_BUILTINS:
        out.append("git alias/extension")  # an alias can be `!sh -c ...`; an extension is any git-* program
    elif sub == "bisect" and rest[:1] == ["run"]:
        out += ["git bisect run"] + _prog_reasons(rest, rraws, 1, ctx)
    elif sub == "submodule" and "foreach" in rest:
        out.append("git submodule foreach")
    elif sub == "rebase" and any(w in ("-x", "--exec") or w.startswith(("--exec=", "-x")) for w in rest):
        out.append("git rebase --exec")
    elif sub in ("filter-branch", "hook"):
        out.append("git " + sub)
    # configured patterns headed by git match from the subcommand on: `git -C repo push`
    for pat in ctx.pats:
        if pat[0] == "git" and len(pat) > 1 and pat[1] == sub:
            it = iter(rest)
            if all(p in it for p in pat[2:]):
                out.append(" ".join(pat))
    return out


# ---- gh, curl/wget, cloud CLIs

_GH_READ = {"view", "list", "ls", "diff", "checks", "status", "download", "watch", "checkout", "clone", "get", "field-list",
            "item-list", "check", "verify", "token"}
_GH_TOP_READ = {"search", "status", "version", "help", "completion", "browse"}


def _gh_reasons(args: list[str], raws: list[str]) -> list[str]:
    pos = _positionals(args, raws)
    if not pos:
        return []
    top, traw = pos[0]
    if not _plain(traw):
        return ["non-literal gh command"]
    if top == "api":  # GET unless a method or fields say otherwise (fields turn it into a POST)
        method, fields = None, False
        for k, a in enumerate(args):
            if a in ("-X", "--method") and k + 1 < len(args):
                method = args[k + 1] if _plain(raws[k + 1]) else "?"
            elif a.startswith("--method="):
                method = a.split("=", 1)[1]
            elif a.startswith("-X") and len(a) > 2:
                method = a[2:]
            elif a in ("-f", "-F", "--field", "--raw-field", "--input") or a.startswith(
                    ("--field=", "--raw-field=", "--input=", "-f", "-F")):
                fields = True
        m = (method or ("POST" if fields else "GET")).upper()
        return [] if m in ("GET", "HEAD") else ["gh api " + m]
    if top in _GH_TOP_READ:
        return []
    if len(pos) >= 2 and _plain(pos[1][1]) and pos[1][0] in _GH_READ:
        return []
    return ["gh write"]


_LOOPBACK = re.compile(r"(https?://)?(localhost|127(\.\d{1,3}){3}|\[::1\]|0\.0\.0\.0)(:\d{1,5})?([/?#][^\s@]*)?", re.I)
_CURL_LONG = {"silent", "show-error", "fail", "fail-with-body", "head", "location", "verbose", "insecure", "include",
              "globoff", "no-buffer", "compressed", "no-progress-meter", "http1.1", "http2", "ipv4", "ipv6", "get"}
_CURL_LONG_ARG = {"output", "header", "max-time", "connect-timeout", "retry", "retry-delay", "retry-max-time",
                  "write-out", "user-agent", "referer", "range", "request", "max-filesize"}
_WGET_LONG = {"quiet", "spider", "server-response", "no-check-certificate", "no-verbose", "verbose"}
_WGET_LONG_ARG = {"output-document", "tries", "timeout", "user-agent", "header", "max-redirect", "method"}


def _http_reasons(prog: str, args: list[str], raws: list[str]) -> list[str]:
    """curl/wget: a GET (or HEAD) of loopback URLs only, with options that send nothing, is local."""
    if prog == "curl":
        p = _parse(args, "sSfILvkiqgN46G", "oHmwAerXY", _CURL_LONG, _CURL_LONG_ARG)
    else:
        p = _parse(args, "qSvNc", "OotTU", _WGET_LONG, _WGET_LONG_ARG)
    if p is None or not all(_plain(r) for r in raws):
        return [prog]
    opts, urls = p
    if any(k in ("X", "request", "method") and (v or "").upper() not in ("GET", "HEAD") for k, v in opts):
        return [prog]
    return [] if urls and all(_LOOPBACK.fullmatch(u) for u in urls) else [prog]


_REMOTE_CLI = {"aws", "gcloud", "gsutil", "az", "doctl", "heroku", "fly", "flyctl", "vercel", "netlify", "firebase",
               "wrangler", "eksctl", "pulumi", "sam", "cdk", "serverless", "sls", "ansible", "ansible-playbook", "oc",
               "kubectl", "helm", "terraform", "tofu", "rclone", "s3cmd"}
_ALWAYS_REMOTE = {"http", "https", "xh", "httpie", "mail", "mailx", "sendmail", "mutt", "msmtp", "swaks", "nc", "ncat",
                  "socat", "telnet", "lftp"}
_REMOTE_READ = {"get", "describe", "list", "ls", "show", "logs", "log", "status", "info", "whoami", "version", "help",
                "explain", "top", "view", "plan", "validate", "fmt", "output", "graph", "providers", "preview", "diff",
                "template", "lint", "search", "history", "cluster-info", "api-resources", "api-versions", "config",
                "init", "inspect", "cat", "tree", "size", "check", "console", "can-i", "current-context", "get-contexts"}
_REMOTE_WRITE = {"delete", "create", "update", "deploy", "apply", "set", "add", "remove", "rm", "destroy", "put", "patch",
                 "scale", "restart", "start", "stop", "run", "exec", "rollout", "push", "upload", "sync", "cp", "mv",
                 "import", "invoke", "publish", "release", "promote", "attach", "detach", "enable", "disable", "reset",
                 "reboot", "ssh", "scp", "login", "new", "select", "install", "upgrade", "uninstall", "rollback", "edit",
                 "label", "annotate", "drain", "cordon", "uncordon", "taint", "untaint", "replace", "expose", "autoscale",
                 "refresh", "up", "down", "kill", "send", "copy", "move", "purge", "rotate", "revoke", "grant"}


def _remote_reasons(prog: str, args: list[str], raws: list[str]) -> list[str]:
    pos = _positionals(args, raws)
    words = {w.lower() for w, r in pos if _plain(r)}
    if len(words) < len(pos) or words & _REMOTE_WRITE:
        return [f"remote CLI {prog}"]
    reads = any(w in _REMOTE_READ or w.startswith(("describe-", "list-", "get-")) for w in words)
    return [] if reads else [f"remote CLI {prog}"]


# ---- runners of a command: `xargs git push`, `watch git status`, `strace -f ./x`
# name -> (short options taking a value, long options taking a value, operands before the command)
_RUNNERS: dict[str, tuple[str, set, int]] = {
    "xargs": ("adEILnPs", {"arg-file", "delimiter", "eof", "replace", "max-lines", "max-args", "max-procs",
                           "max-chars", "process-slot-var"}, 0),
    "parallel": ("jJNnSaIdE", {"jobs", "sshlogin", "arg-file", "colsep", "results", "joblog", "delimiter", "tmpdir",
                               "workdir", "basefile"}, 0),
    "watch": ("n", {"interval"}, 0),
    "strace": ("abeEIoOpPsSuUX", set(), 0), "ltrace": ("aAeEFlnopsSuwxX", set(), 0), "setsid": ("", set(), 0),
    "unbuffer": ("", set(), 0), "ionice": ("cnpPu", set(), 0), "chrt": ("", set(), 1), "taskset": ("", set(), 1),
    "doas": ("uC", set(), 0), "flock": ("wE", set(), 1), "chroot": ("", {"userspec", "groups"}, 1),
    "xvfb-run": ("nsfep", set(), 0), "valgrind": ("", set(), 0), "catchsegv": ("", set(), 0),
    "proxychains": ("f", set(), 0), "proxychains4": ("f", set(), 0), "torsocks": ("uaPp", set(), 0),
    "dbus-run-session": ("", {"config-file", "dbus-daemon"}, 0), "firejail": ("", set(), 0),
    "systemd-run": ("pEu", {"unit", "property", "setenv", "uid", "gid", "description", "slice", "working-directory"}, 0),
    "caffeinate": ("tw", set(), 0),
}


def _runner_target(name: str, words: list[str], i: int) -> int:
    """Index of the command a runner at words[i] runs (len(words) if none)."""
    short_arg, long_arg, operands = _RUNNERS[name]
    i += 1
    while i < len(words):
        w = words[i]
        if w == "--":
            i += 1
            break
        if w.startswith("--"):
            i += 2 if w[2:] in long_arg else 1
        elif w.startswith("-") and len(w) > 1:
            j = next((k for k, c in enumerate(w[1:], 1) if c in short_arg), None)
            i += 2 if j is not None and j == len(w) - 1 else 1
        else:
            break
    return i + operands


@dataclasses.dataclass
class _Ctx:
    pats: list[list[str]]
    heads: set[str]
    depth: int = 0


def _prog_reasons(words: list[str], raws: list[str], p: int, ctx: _Ctx) -> list[str]:
    """Why the command whose program is words[p] is a commit point ([] if it isn't)."""
    if p >= len(words) or ctx.depth > 6:
        return ["nested too deep"] if ctx.depth > 6 else []
    w, raw = words[p], raws[p]
    if raw in ("[", "[[", "((") or w == "[":
        return []  # test expressions; substitutions inside them are nested scripts
    if not _plain(raw):
        return ["non-literal program"]
    name = w.rsplit("/", 1)[-1]
    if "/" in w and not _trusted_path(w) and name not in ("gradlew", "mvnw") and not _TOOL_BIN.fullmatch(w.rsplit("/", 1)[0]):
        return ["script"]
    if "/" not in w and name.endswith(_SCRIPT_EXT):
        return ["script"]
    args, rraws = words[p + 1 :], raws[p + 1 :]
    ctx = _Ctx(ctx.pats, ctx.heads, ctx.depth + 1)
    out = [" ".join(pat) for pat in ctx.pats
           if pat[0] == name and name not in ("git", "gh", "curl", "wget") and _at(words, p, pat, True)]
    if name in _SHELLS:
        out.append("shell")
    elif name in ("source", "."):
        out.append("sources a script")
    elif _INTERP.fullmatch(name):
        out += _interp_reasons(args, rraws, ctx, name)
    elif name in _TASK_RUNNERS:
        out += _task_reasons(name, args, rraws)
    elif name in _PKG:
        out += _pkg_reasons(name, args, rraws, ctx)
    elif name == "git":
        out += _git_reasons(args, rraws, ctx)
    elif name == "gh":
        out += _gh_reasons(args, rraws)
    elif name in ("curl", "wget"):
        out += _http_reasons(name, args, rraws) if name in ctx.heads else []
    elif name in _REMOTE_CLI:
        out += _remote_reasons(name, args, rraws)
    elif name in _ALWAYS_REMOTE:
        out.append(f"network client {name}")
    elif name in _RUNNERS:
        if name == "parallel" and any(a in ("-S", "--sshlogin") or a.startswith(("--sshlogin=", "-S")) for a in args):
            out.append("parallel --sshlogin")
        if name == "flock" and any(a in ("-c", "--command") for a in args):
            out.append("shell")
        out += _prog_reasons(words, raws, _runner_target(name, words, p), ctx)
    elif name == "find":
        for k in range(p + 1, len(words)):
            if words[k] in ("-exec", "-execdir", "-ok", "-okdir"):
                out += _prog_reasons(words, raws, k + 1, ctx)
    return out


def _at(words: list[str], i: int, pat: list[str], loose: bool) -> bool:
    """`pat` starts at words[i]: exactly (`git push ...`), or loosely with its other words anywhere
    after it in order (`docker compose push`, `npm --registry x publish`)."""
    if i >= len(words) or words[i].rsplit("/", 1)[-1] != pat[0]:
        return False
    if not loose:
        return words[i + 1 : i + len(pat)] == pat[1:]
    it = iter(words[i + 1 :])
    return all(p in it for p in pat[1:])


def _scripts(words: list[str], raws: list[str]) -> list[tuple[str, bool]]:
    """Command text nested in a simple command: (text, is it certainly a command)."""
    out: list[tuple[str, bool]] = []
    for raw in raws:  # $(...) and `...`
        i = 0
        while i < len(raw):
            if raw.startswith("$(", i):
                j = _skip_subst(raw, i + 2)
                out.append((raw[i + 2 : j - 1], True))
                i = j
            elif raw[i] == "`":
                j = raw.find("`", i + 1)
                j = len(raw) if j < 0 else j
                out.append((raw[i + 1 : j], True))
                i = j + 1
            else:
                i += 1
    names = [w.rsplit("/", 1)[-1] for w in words]
    for i, name in enumerate(names):
        if name == "eval":
            out.append((" ".join(words[i + 1 :]), True))
        if name in _SHELLS or name == "env":
            for k in range(i + 1, len(words) - 1):
                w = words[k]
                if (name != "env" and w.startswith("-") and not w.startswith("--") and "c" in w) or (
                        name == "env" and w in ("-S", "--split-string")):
                    out.append((words[k + 1], True))
        if name == "env":
            out += [(w.split("=", 1)[1] if w.startswith("--") else w[2:], True) for w in words[i + 1 :]
                    if w.startswith("--split-string=") or (w.startswith("-S") and len(w) > 2)]
    # any other word that looks like a command line: looked at only when something in the line runs
    # text as a script (see _reasons)
    out += [(w, False) for w in words if _SCRIPTY.search(w)]
    return out


def _reasons(cmd: str, ctx: _Ctx, out: list[str]) -> None:
    if ctx.depth > 6:
        out.append("nested too deep")
        return
    # read-only by the built-in allowlist (operators' extra_readonly_commands don't make a program
    # safe to run unattended as a commit)
    cmds = [(words, raws, ok and _program_readonly(words, raws, set())) for words, raws, ok in _commands(cmd)]
    # `echo 'git push' | sh`, `watch 'git push'`, `bash <<EOF`: quoted text may run as a script;
    # otherwise it is data (`git commit -m "then git push"`, `grep "git push" f`)
    runs_text = any(w.rsplit("/", 1)[-1] in _RUNS_TEXT for words, _, ro in cmds if not ro for w in words)
    multi = [p for p in ctx.pats if len(p) > 1 and p[0] not in ("gh", "curl", "wget")]
    for words, raws, ro in cmds:
        if not ro:
            p = program_index(words, raws)[0]
            out += _prog_reasons(words, raws, p, ctx)
            # a wrapper we don't model may still run a known commit command: `mywrap git push`
            names = [w.rsplit("/", 1)[-1] for w in words]
            out += [" ".join(pat) for pat in multi for i in range(p + 1, len(words)) if names[i : i + len(pat)] == pat]
        for s, sure in _scripts(words, raws):
            if sure or runs_text:
                _reasons(s, _Ctx(ctx.pats, ctx.heads, ctx.depth + 1), out)


def commit_reason(tool: str, args: dict, cfg: Config) -> str:
    """Why a call is a commit point ('' if it isn't): the matching commit tool pattern, or the
    reasons found in its shell command (`git push`, `interpreter`, `task runner`, `gh write`, ...)."""
    out = [f"tool {p}" for p in cfg.commit_tools if fnmatch.fnmatchcase(tool, p)][:1]
    pats = [p.split() for p in cfg.commit_commands if p.split()]
    ctx = _Ctx(pats, {p[0] for p in pats})
    for v in _shell_args(args):
        _reasons(v, ctx, out)
    return "; ".join(dict.fromkeys(out))


def is_commit_point(tool: str, args: dict, cfg: Config) -> bool:
    return bool(commit_reason(tool, args, cfg))


# ------------------------------------------------------------------ repository taint
#
# git reads run programs the repository names (core.fsmonitor, diff.external, textconv and clean
# filters, an embedded bare repo's config). With trust_repo_config the checkout is trusted, but a
# call in the run that changes what git will read ends that trust for the rest of the run.

_GIT_META = re.compile(r"(^|[\s/'\"=:<>|;&(])\.git($|[/\s'\"):;&|])|\.git(attributes|modules)\b|(^|[\s/'\"=])\.gitconfig\b")
_ARCHIVE = {"unzip", "7z", "7za", "7zr", "unrar", "unar", "cpio", "pax", "ar", "jar", "bsdtar", "aunpack", "atool",
            "dpkg-deb"}


def _git_config_taints(rest: list[str], raws: list[str]) -> str:
    if _git(["config", *rest]):
        return ""  # a read
    pos = [w for w, r in _positionals(rest, raws, {"-f", "--file", "--blob", "--type", "--default", "--value", "--url"})]
    if pos and pos[0] in ("set", "unset", "replace-all", "add"):
        pos = pos[1:]
    key = pos[0] if pos else ""
    kv = key + "=" + (pos[1] if len(pos) > 1 else "")
    return "" if key and _git_key_safe(kv) and not {"-e", "--edit", "-f", "--file"} & set(rest) else f"git config {key}"


@functools.lru_cache(maxsize=4096)
def _shell_taint(cmd: str, depth: int = 0) -> str:
    if depth > 4:
        return "nested too deep"
    if _GIT_META.search(cmd):
        return "git metadata path"
    for words, raws, _ in _commands(cmd):
        p = program_index(words, raws)[0]
        if p == len(words) or words[p] in ("export", "declare", "typeset", "readonly", "local"):
            if any(w.startswith("GIT_") and "=" in w for w in words):
                return "git environment"
        if p < len(words):
            name, args = words[p].rsplit("/", 1)[-1], words[p + 1 :]
            if name == "git":
                i, _ = _git_global(args, raws[p + 1 :])
                sub = args[i] if i is not None else ""
                if sub == "config":
                    why = _git_config_taints(args[i + 1 :], raws[p + i + 2 :])
                    if why:
                        return why
                elif sub == "clone" or sub == "submodule" or (sub in ("init", "init-db") and any(
                        a.startswith(("--bare", "--separate-git-dir")) for a in args)):
                    return "git " + sub
            elif name == "tar" and args and (
                    ("x" in args[0] and not args[0].startswith("--")) or "--extract" in args or "--get" in args
                    or any(a.startswith("-") and not a.startswith("--") and "x" in a for a in args)):
                return "archive extraction"
            elif name in _ARCHIVE:
                return "archive extraction"
            elif _INTERP.fullmatch(name) and any(a in ("zipfile", "tarfile", "shutil") for a in args):
                return "archive extraction"
        for s, _ in _scripts(words, raws):
            why = _shell_taint(s, depth + 1)
            if why:
                return why
    return ""


def repo_taint(tool: str, args: dict, cfg: Config) -> str:
    """Why this call ends trust in the repository's git config for the rest of its run ('' if it
    doesn't): it writes `.git/*`, `.gitattributes` or `.gitmodules`, sets a git config key that can run
    a program, sets GIT_* variables, runs `init --bare`/`clone`/`submodule`, or extracts an archive."""
    for k, v in args.items():
        st = shell_text(k, v)
        if st is None and not isinstance(v, str):
            continue
        why = _shell_taint(st[0]) if st is not None else (
            "git metadata path" if "\n" not in v and _GIT_META.search(v) else "")  # a path, not file content
        if why:  # reading `.gitattributes` changes nothing
            return "" if is_readonly(tool, args, cfg) else why
    return ""


def untrusted(cfg: Config) -> Config:
    """`cfg` with the repository's git config distrusted (for a tainted run)."""
    return cfg if not cfg.trust_repo_config else dataclasses.replace(cfg, trust_repo_config=False)
