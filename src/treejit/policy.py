"""Which calls replay may emit on its own: read-only/idempotent tools, and commit points."""

from __future__ import annotations

import fnmatch

from .config import Config
from .shellwords import segments, tokenize
from .templates import SHELL_KEYS

READONLY_PROGRAMS = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "grep", "egrep", "fgrep", "rg", "ag", "tree", "stat",
    "file", "which", "whereis", "type", "du", "df", "date", "sort", "uniq", "cut", "tr", "jq", "yq", "diff", "cmp",
    "basename", "dirname", "realpath", "readlink", "cd", "true", "test", "[", "nl", "less", "more", "column", "id",
    "whoami", "hostname", "uname", "env", "printenv", "md5sum", "sha1sum", "sha256sum", "ps", "fd", "bat", "comm",
    "find", "sed", "awk", "git",
}
GIT_READONLY = {
    "status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "ls-tree", "blame", "grep", "describe",
    "remote", "config", "shortlog", "reflog", "cat-file", "rev-list", "merge-base", "tag", "stash",
}
GIT_WRITE_FLAGS = {"-d", "-D", "-m", "-M", "--delete", "--move", "--set-upstream-to", "add", "remove", "rename",
                   "set-url", "--unset", "--add", "push", "pop", "drop", "clear", "apply"}


def _program_readonly(words: list[str], extra: set[str]) -> bool:
    if not words:
        return True
    prog = words[0].rsplit("/", 1)[-1]
    if prog in extra:
        return True
    if prog not in READONLY_PROGRAMS:
        return False
    rest = words[1:]
    if prog == "find":
        return not any(w in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls") for w in rest)
    if prog == "sed":
        return not any(w == "-i" or w.startswith("-i") or w == "--in-place" for w in rest)
    if prog == "awk":
        return not any("system(" in w or ">" in w for w in rest)
    if prog == "git":
        sub = next((w for w in rest if not w.startswith("-")), None)
        if sub not in GIT_READONLY:
            return False
        tail = rest[rest.index(sub) + 1 :] if sub else []
        if sub in ("branch", "remote", "config", "tag", "stash"):
            if sub == "stash":
                return bool(tail) and tail[0] in ("list", "show")
            if sub == "tag":
                return not tail or tail[0] in ("-l", "--list")
            if sub == "config":
                return any(w in ("--get", "--list", "-l", "--get-all") for w in tail)
            return not any(w in GIT_WRITE_FLAGS for w in tail)
        return True
    return True


def shell_readonly(cmd: str, cfg: Config) -> bool:
    toks = tokenize(cmd)
    for t in toks:
        if t.op and t.val in (">", ">>", ">|", "&>"):
            # only allow redirects to /dev/null
            idx = toks.index(t)
            nxt = toks[idx + 1] if idx + 1 < len(toks) else None
            if nxt is None or nxt.val != "/dev/null":
                return False
    extra = set(cfg.extra_readonly_commands)
    for seg in segments(toks):
        words = [t.val for t in seg if not t.op]
        # drop redirect targets like 2>&1 remnants and env assignments
        words = [w for w in words if not (w.isdigit() and len(w) == 1)]
        while words and "=" in words[0] and not words[0].startswith("-"):
            words = words[1:]
        if any("$(" in w or "`" in w for w in words):
            return False
        if not _program_readonly(words, extra):
            return False
    return True


def _match(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def is_readonly(tool: str, args: dict, cfg: Config) -> bool:
    shell = [v for k, v in args.items() if k in SHELL_KEYS and isinstance(v, str)]
    if tool in cfg.shell_tools or (shell and not _match(tool, cfg.replay_tools)):
        return bool(shell) and all(shell_readonly(c, cfg) for c in shell)
    return _match(tool, cfg.replay_tools)


def is_commit_point(tool: str, args: dict, cfg: Config) -> bool:
    if _match(tool, cfg.commit_tools):
        return True
    for k, v in args.items():
        if k in SHELL_KEYS and isinstance(v, str):
            for seg in segments(tokenize(v)):
                words = " ".join(t.val for t in seg if not t.op)
                if any(words == p or words.startswith(p + " ") for p in cfg.commit_commands):
                    return True
    return False
