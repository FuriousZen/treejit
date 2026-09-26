"""A span-preserving shell tokenizer.

`shlex` loses positions and mangles operators, so replayed commands could not be
re-rendered faithfully. This tokenizer returns each token's cooked value (quotes
removed, escapes applied) plus its [start, end) span in the original string, so
templated commands are re-rendered by splicing new values into the original text
and everything else (quoting style, heredocs, `$(...)`, comments) is kept byte-for-byte.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

_OPERATORS = sorted(
    ["&&", "||", ";;", "|&", ">>", "<<-", "<<<", "<<", ">&", "<&", "&>", ">|", "|", ";", "&", "<", ">", "(", ")"],
    key=len,
    reverse=True,
)
_OP_CHARS = set("|&;<>()")
# Operators that separate commands (as opposed to redirections).
SEPARATORS = {"&&", "||", ";;", "|&", "|", ";", "&", "\n", "(", ")"}


@dataclass(frozen=True)
class Tok:
    val: str
    start: int
    end: int
    op: bool = False


def tokenize(s: str) -> list[Tok]:
    toks: list[Tok] = []
    pending_heredocs: list[tuple[str, bool]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\n":
            toks.append(Tok("\n", i, i + 1, True))
            i += 1
            for delim, strip_tabs in pending_heredocs:
                body_start = i
                j = _find_heredoc_end(s, i, delim, strip_tabs)
                body_end = j[0]
                toks.append(Tok(s[body_start:body_end], body_start, body_end))
                i = j[1]
            pending_heredocs = []
            continue
        if c.isspace():
            i += 1
            continue
        if c == "#" and (i == 0 or s[i - 1].isspace() or s[i - 1] in _OP_CHARS):
            while i < n and s[i] != "\n":
                i += 1
            continue
        if c in _OP_CHARS:
            op = next(o for o in _OPERATORS if s.startswith(o, i))
            toks.append(Tok(op, i, i + len(op), True))
            i += len(op)
            if op in ("<<", "<<-"):
                # read the delimiter word
                while i < n and s[i] in " \t":
                    i += 1
                start = i
                val, i = _read_word(s, i)
                if val:
                    toks.append(Tok(val, start, i))
                    pending_heredocs.append((val, op == "<<-"))
            continue
        start = i
        val, i = _read_word(s, i)
        toks.append(Tok(val, start, i))
    return toks


def _find_heredoc_end(s: str, i: int, delim: str, strip_tabs: bool) -> tuple[int, int]:
    """Return (body_end, resume_index) for a heredoc body starting at i."""
    pos = i
    while pos <= len(s):
        nl = s.find("\n", pos)
        line_end = len(s) if nl < 0 else nl
        line = s[pos:line_end]
        if (line.lstrip("\t") if strip_tabs else line) == delim:
            return pos, (line_end + 1 if nl >= 0 else len(s))
        if nl < 0:
            return len(s), len(s)
        pos = nl + 1
    return len(s), len(s)


def _read_word(s: str, i: int) -> tuple[str, int]:
    out: list[str] = []
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace() or c in _OP_CHARS:
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
            out.append(ansi_c(s[i + 2 : min(j, n)]))
            i = j + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == '"':
            i += 1  # $"..." (locale translation) cooks like "..."
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
                    k = _skip_subst(s, j + 2)
                    buf.append(s[j:k])
                    j = k
                elif s[j] == "`":
                    k = _skip_backtick(s, j + 1)
                    buf.append(s[j:k])
                    j = k
                else:
                    buf.append(s[j])
                    j += 1
            out.append("".join(buf))
            i = j + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            k = _skip_subst(s, i + 2)
            out.append(s[i:k])
            i = k
            continue
        if c == "`":
            k = _skip_backtick(s, i + 1)
            out.append(s[i:k])
            i = k
            continue
        out.append(c)
        i += 1
    return "".join(out), i


_ANSI_SIMPLE = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
                "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}
_OCT = re.compile(r"[0-7]{1,3}")
_HEX = {"x": re.compile(r"[0-9A-Fa-f]{1,2}"), "u": re.compile(r"[0-9A-Fa-f]{1,4}"),
        "U": re.compile(r"[0-9A-Fa-f]{1,8}")}


def ansi_c(body: str) -> str:
    r"""Decode the body of a bash `$'...'` word as bash does: \a \b \e \E \f \n \r \t \v \\ \' \" \?,
    \NNN (octal), \xHH, \uHHHH, \UHHHHHHHH and \cX. Any other escape stays as written, and a NUL
    ends the word (bash truncates there)."""
    out: list[str] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        d = body[i + 1]
        m = _OCT.match(body, i + 1)
        hx = _HEX[d].match(body, i + 2) if d in _HEX else None
        if d in _ANSI_SIMPLE:
            out.append(_ANSI_SIMPLE[d])
            i += 2
        elif m:
            out.append(chr(int(m.group(), 8) & 0xFF))
            i = m.end()
        elif hx:
            code = int(hx.group(), 16)
            out.append(chr(code) if code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF else "�")
            i = hx.end()
        elif d == "c" and i + 2 < n:
            out.append(chr(ord(body[i + 2]) & 0x1F))
            i += 3
        else:
            out.append(c + d)
            i += 2
    return "".join(out).split("\0", 1)[0]


def _skip_backtick(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] != "`":
        i += 2 if s[i] == "\\" else 1
    return min(i + 1, n)


_HEREDOC_IN_SUBST = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _skip_subst(s: str, i: int) -> int:
    """Index just past the ')' closing a `$(` whose body starts at i."""
    depth, n = 1, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "'":
            j = s.find("'", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                if s[j] == "\\":
                    j += 1
                elif s[j] == "$" and j + 1 < n and s[j + 1] == "(":
                    j = _skip_subst(s, j + 2) - 1
                j += 1
            i = j + 1
            continue
        if c == "<" and s.startswith("<<", i) and not s.startswith("<<<", i):
            m = _HEREDOC_IN_SUBST.match(s, i)
            if m:
                nl = s.find("\n", m.end())
                if nl >= 0:
                    _, resume = _find_heredoc_end(s, nl + 1, m.group(2), s.startswith("<<-", i))
                    i = resume
                    continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


_SAFE = re.compile(r"[\w@%+=:,./-]+")


def quote(val: str, like: str | None = None) -> str:
    """Quote `val` for the shell, preferring the quoting style of `like` (a raw token)."""
    if val and _SAFE.fullmatch(val):
        return val
    if like and like.startswith('"') and not any(ch in val for ch in '"$`\\'):
        return f'"{val}"'
    return shlex.quote(val)


def segments(toks: list[Tok]) -> list[list[Tok]]:
    segs: list[list[Tok]] = [[]]
    for t in toks:
        if t.op and t.val in SEPARATORS:
            if segs[-1]:
                segs.append([])
        else:
            segs[-1].append(t)
    return [s for s in segs if s]


# Programs whose first argument is a subcommand that changes meaning.
MULTI_COMMAND = {
    "git", "npm", "pnpm", "yarn", "pip", "pip3", "uv", "cargo", "docker", "kubectl", "go", "gh", "poetry",
    "brew", "apt", "apt-get", "systemctl", "dotnet", "bun", "deno", "conda", "helm", "terraform", "make",
}
_WORDLIKE = re.compile(r"[A-Za-z][\w.:-]*")
_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

# Transparent wrappers: they run the rest of the line as a command. name -> (flags, flags taking a
# value, long flags, long flags taking a value, operands before the command, options with side
# effects; None = the wrapper itself is never side-effect free). Parsing stops at the first
# non-option, as their own getopt("+...") does.
_WRAPPERS: dict[str, tuple[str, str, set, set, int, set | None]] = {
    "env": ("i0v", "uC", {"ignore-environment", "null", "debug"}, {"unset", "chdir"}, 0, set()),
    # privilege elevation always needs an operator's approval
    "sudo": ("AbEHiknPSs", "CDghpRrTUu",
             {"askpass", "background", "preserve-env", "non-interactive", "preserve-groups", "set-home", "login",
              "shell", "stdin", "reset-timestamp"},
             {"close-from", "chdir", "group", "host", "prompt", "role", "type", "command-timeout", "other-user", "user"},
             0, None),
    "time": ("pvqa", "fo", {"portability", "verbose", "quiet", "append"}, {"format", "output"}, 0,
             {"a", "o", "append", "output"}),  # GNU time -o FILE writes FILE
    "nohup": ("", "", set(), set(), 0, set()),
    "exec": ("cl", "a", set(), set(), 0, set()),
    "command": ("p", "", set(), set(), 0, set()),
    "nice": ("", "n", set(), {"adjustment"}, 0, set()),
    "timeout": ("v", "sk", {"preserve-status", "foreground", "verbose"}, {"signal", "kill-after"}, 1, set()),
    "stdbuf": ("", "ioe", set(), {"input", "output", "error"}, 0, set()),
}
_NUMERIC_OPT = re.compile(r"-\d+")


def _skip_wrapper(name: str, words: list[str], i: int) -> tuple[int, bool] | None:
    """Index just past a wrapper's own options and operands, and whether they are side-effect free.
    None: an option this parser doesn't know (e.g. `env -S STRING`), so the command can't be located."""
    short, short_arg, long, long_arg, operands, dirty = _WRAPPERS[name]
    clean, n = dirty is not None, len(words)
    dirty = dirty or set()
    while i < n:
        w = words[i]
        if w == "--":
            i += 1
            break
        if (name == "env" and w == "-") or (name == "nice" and _NUMERIC_OPT.fullmatch(w)):
            i += 1
            continue
        if w.startswith("--"):
            opt, eq, _ = w[2:].partition("=")
            if opt in long and not eq:
                pass
            elif opt in long_arg:
                i += 0 if eq else 1
            else:
                return None
            clean = clean and opt not in dirty
        elif w.startswith("-") and len(w) > 1:
            for j, c in enumerate(w[1:], 1):
                clean = clean and c not in dirty
                if c in short:
                    continue
                if c not in short_arg:
                    return None
                if j == len(w) - 1:
                    i += 1  # the value is the next word
                break
        else:
            break
        i += 1
    if name in ("env", "sudo"):
        while i < n and _ASSIGN.match(words[i]):
            i += 1
    return i + operands, clean


def _lookup_only(words: list[str], i: int) -> bool:
    """`command -v NAME` / `command -V NAME` describe NAME without running it."""
    while i < len(words) and words[i].startswith("-") and words[i] != "--":
        if set(words[i][1:]) & {"v", "V"}:
            return True
        i += 1
    return False


def unwrap(words: list[str]) -> tuple[int, bool, list[int]]:
    """Locate the program a simple command really runs.

    Skips `VAR=val` assignments and transparent wrappers with their options (`env -i FOO=1`,
    `sudo -u x`, `time -p`, `nohup`, `exec`, `command`, `nice -n 5`, `timeout -s KILL 10`, `stdbuf -oL`).
    Returns (index of the program, whether every skipped word was understood and side-effect free,
    indices of the wrapper words). The index is len(words) when there are only assignments; a wrapper
    with no command after it (bare `env`, `command -v git`) is itself the program. This is the single
    view of "which program runs" shared by edge shapes, the read-only policy and commit points.
    """
    i, clean, n, wrappers = 0, True, len(words), []
    while i < n:
        if _ASSIGN.match(words[i]):
            i += 1
            continue
        name = words[i].rsplit("/", 1)[-1]
        if name not in _WRAPPERS or (name == "command" and _lookup_only(words, i + 1)):
            return i, clean, wrappers
        r = _skip_wrapper(name, words, i + 1)
        if r is None:
            return i, False, wrappers
        j, ok = r
        if j >= n:
            return i, clean and ok, wrappers
        wrappers.append(i)
        i, clean = j, clean and ok
    return i, clean, wrappers


# Reserved words that may precede a command in a simple-command segment (`if git push; then`,
# `{ git push; }`, `! git push`, `do git push; done`). Only unquoted ones are keywords.
KEYWORDS = {"{", "}", "!", "if", "then", "elif", "else", "fi", "do", "done", "while", "until", "esac", "coproc"}


def skip_keywords(words: list[str], raws: list[str] | None = None) -> int:
    """Index of the first word after leading reserved words (and `function NAME`)."""
    raws = words if raws is None else raws
    i = 0
    while i < len(words) and raws[i] == words[i]:
        if words[i] in KEYWORDS:
            i += 1
        elif words[i] == "function" and i + 1 < len(words):
            i += 2
        else:
            break
    return i


def program_index(words: list[str], raws: list[str] | None = None) -> tuple[int, bool, list[int]]:
    """`unwrap` after leading reserved words: (program index, clean, wrapper indices)."""
    k = skip_keywords(words, raws)
    start, clean, wrappers = unwrap(words[k:])
    return start + k, clean, [w + k for w in wrappers]


def command_heads(cmd: str) -> list[str]:
    """Structural signature of a shell command: the program (+ subcommand) of each segment.

    `cd app && git commit -m "x" | cat` -> ["cd", "git commit", "cat"]
    """
    heads = []
    for seg in segments(tokenize(cmd)):
        words = [t for t in seg if not t.op]
        # skip reserved words, env assignments and transparent wrappers: the same view the policy uses
        words = words[program_index([t.val for t in words], [cmd[t.start : t.end] for t in words])[0]:]
        if not words:
            continue
        prog = words[0].val.rsplit("/", 1)[-1] if "/" in words[0].val and not words[0].val.startswith(".") else words[0].val
        head = prog
        if prog in MULTI_COMMAND and len(words) > 1 and _WORDLIKE.fullmatch(words[1].val) and not words[1].val.startswith("-"):
            head += " " + words[1].val
        elif prog in ("python", "python3") and len(words) > 2 and words[1].val == "-m":
            head += " -m " + words[2].val
        heads.append(head)
    return heads
