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
            buf = []
            while j < n and s[j] != "'":
                if s[j] == "\\" and j + 1 < n:
                    buf.append(_ansi_escape(s[j + 1]))
                    j += 2
                else:
                    buf.append(s[j])
                    j += 1
            out.append("".join(buf))
            i = j + 1
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


def _ansi_escape(c: str) -> str:
    return {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(c, c)


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


def command_heads(cmd: str) -> list[str]:
    """Structural signature of a shell command: the program (+ subcommand) of each segment.

    `cd app && git commit -m "x" | cat` -> ["cd", "git commit", "cat"]
    """
    heads = []
    for seg in segments(tokenize(cmd)):
        words = [t for t in seg if not t.op]
        # skip env assignments and a few transparent prefixes
        while words and (_ASSIGN.match(words[0].val) or words[0].val in ("sudo", "env", "time", "nohup", "exec")):
            words = words[1:]
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
