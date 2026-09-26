"""Which calls replay may emit on its own: read-only/idempotent tools, and commit points.

A shell command is read-only only if every simple command in it is: the program (after `VAR=val`
assignments and transparent wrappers like `env`, `time`, `timeout`, see `shellwords.unwrap`) must be
on an allowlist, and programs with side-effecting options (`sort -o`, `git diff --output=`, `rg --pre`,
`find -exec`, ...) must pass a per-program check that accepts only what it understands. Anything
unrecognised is not read-only: a false "no" costs one model call, a false "yes" runs a write with no
model call and no human in the loop.
"""

from __future__ import annotations

import fnmatch
import re

from .config import Config
from .shellwords import _ASSIGN, _skip_subst, segments, tokenize, unwrap
from .templates import SHELL_KEYS

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


def _program_readonly(words: list[str], raws: list[str], extra: set[str]) -> bool:
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
    return all(ok and _program_readonly(words, raws, extra) for words, raws, ok in _commands(cmd, toks))


def _match(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def is_readonly(tool: str, args: dict, cfg: Config) -> bool:
    shell = [v for k, v in args.items() if k in SHELL_KEYS and isinstance(v, str)]
    if tool in cfg.shell_tools or (shell and not _match(tool, cfg.replay_tools)):
        return bool(shell) and all(shell_readonly(c, cfg) for c in shell)
    return _match(tool, cfg.replay_tools)


# ------------------------------------------------------------------ commit points

_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "mksh", "fish", "su", "runuser"}
_RUNS_TEXT = _SHELLS | {"watch", "eval"}
_SCRIPTY = re.compile(r"\s|[;&|]")


def _at(words: list[str], i: int, pat: list[str], loose: bool) -> bool:
    """`pat` starts at words[i]: exactly (`git push ...`), or loosely with its other words anywhere
    after it in order (`git -C repo push`, `xargs -n1 git push`)."""
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
    # text as a script (see _commits), and then by its head only
    out += [(w, False) for w in words if _SCRIPTY.search(w)]
    return out


def _commits(cmd: str, pats: list[list[str]], cfg: Config, depth: int, certain: bool = True) -> bool:
    if depth > 4:
        return True
    extra = set(cfg.extra_readonly_commands)
    cmds = [(words, raws, ok and _program_readonly(words, raws, extra)) for words, raws, ok in _commands(cmd)]
    # `echo 'git push' | sh`, `watch 'git push'`, `bash <<EOF`: quoted text may run as a script;
    # otherwise it is data (`git commit -m "then git push"`, `grep "git push" f`)
    runs_text = any(w.rsplit("/", 1)[-1] in _RUNS_TEXT for words, _, readonly in cmds if not readonly for w in words)
    for words, raws, readonly in cmds:
        if any(_at(words, unwrap(words)[0], p, False) for p in pats):
            return True
        if not certain:
            continue
        # a command we can't show read-only may run a commit command anywhere in it
        if not readonly and any(_at(words, i, p, True) for p in pats for i in range(len(words))):
            return True
        if any(_commits(s, pats, cfg, depth + 1, sure) for s, sure in _scripts(words, raws) if sure or runs_text):
            return True
    return False


def is_commit_point(tool: str, args: dict, cfg: Config) -> bool:
    if _match(tool, cfg.commit_tools):
        return True
    pats = [p.split() for p in cfg.commit_commands if p.split()]
    return any(k in SHELL_KEYS and isinstance(v, str) and _commits(v, pats, cfg, 0) for k, v in args.items())
