"""Edge templates: structural shapes, anti-unification, matching and rendering.

An edge is identified by its *shape*: tool name, argument keys, and for shell
arguments the program/subcommand of each command segment. All calls with the
same shape in a family are anti-unified into one template where every argument
is a constant or a variable; shell arguments are templated per token, so
`git commit -m "<msg>"` is one edge with one variable.

Variables are resolved per tree node by binding rules (see bindings.py); a
variable with no rule is a hole.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

from .model import ToolCall
from .shellwords import Tok, command_heads, quote, tokenize
from .util import canon, h, short

SHELL_KEYS = ("command", "cmd")
_NOJS = object()


@dataclass
class Val:
    cooked: str
    raw: str | None = None   # exact shell text, when the value came from a command
    js: Any = _NOJS          # original JSON value, when the value came from a JSON argument

    def has_js(self) -> bool:
        return self.js is not _NOJS


def cook(v: Any) -> str:
    return v if isinstance(v, str) else canon(v)


def is_shell_arg(key: str, value: Any) -> bool:
    return shell_text(key, value) is not None


# argv shell arguments (Codex CLI's `shell` tool: {"command": ["bash", "-lc", "git status"]}). The
# command text is what policy and templates read; `wrap` says how to turn text back into the argument:
#   None            a string argument (the text itself)
#   [shell, flag]   ["bash"|"sh"|"zsh"|"dash"|"ksh", "-c"|"-lc"|...] + [text]: the script runs in that shell
#   "argv"          any other list of strings: shlex.join(argv), and shlex.split back
_ARGV_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_ARGV_FLAG = re.compile(r"-[il]*c[il]*")


def shell_text(key: str, value: Any) -> tuple[str, Any] | None:
    """(command text, wrap) of a shell argument, or None if `key: value` isn't one."""
    if key not in SHELL_KEYS:
        return None
    if isinstance(value, str):
        return value, None
    if not (isinstance(value, list) and value and all(isinstance(x, str) for x in value)):
        return None
    if len(value) == 3 and value[0].rsplit("/", 1)[-1] in _ARGV_SHELLS and _ARGV_FLAG.fullmatch(value[1]):
        return value[2], value[:2]
    return shlex.join(value), "argv"


def unwrap(text: str, wrap: Any) -> Any:
    """The argument for command `text` under `wrap` (see shell_text). ValueError if it can't be split."""
    if wrap is None:
        return text
    if wrap == "argv":
        return shlex.split(text)
    return [*wrap, text]


def shape_of(call: ToolCall) -> str:
    parts = []
    for k in sorted(call.args):
        st = shell_text(k, call.args[k])
        if st is None:
            parts.append([k])
        elif st[1] is None:
            parts.append([k, command_heads(st[0])])      # unchanged for string commands
        else:
            parts.append([k, "argv", st[1], command_heads(st[0])])
    return canon([call.name, parts])


def edge_id(family: str, shape: str) -> str:
    return h(family, shape, n=12)


# ------------------------------------------------------------ anti-unification


def _tokseq(toks: list[Tok]) -> list[tuple[str, bool]]:
    return [(t.val, t.op) for t in toks]


def greedy_gaps(consts: list[tuple[str, bool]], toks: list[Tok]) -> tuple[list[list[Tok]], list[Tok]] | None:
    """Match consts as a subsequence (earliest positions). Returns (gaps, const_tokens)."""
    gaps, anchors, p = [], [], 0
    for val, op in consts:
        q = p
        while q < len(toks) and not (toks[q].val == val and toks[q].op == op):
            q += 1
        if q == len(toks):
            return None
        gaps.append(toks[p:q])
        anchors.append(toks[q])
        p = q + 1
    gaps.append(toks[p:])
    return gaps, anchors


def common_subsequence(seqs: list[list]) -> list:
    common = list(seqs[0])
    for seq in seqs[1:]:
        sm = difflib.SequenceMatcher(None, common, seq, autojunk=False)
        common = [x for blk in sm.get_matching_blocks() for x in common[blk.a : blk.a + blk.size]]
    return common


def unify_tokens(toklists: list[list[Tok]]) -> list[list]:
    consts = common_subsequence([_tokseq(t) for t in toklists])
    nonempty: set[int] = set()
    for toks in toklists:
        g = greedy_gaps(consts, toks)
        assert g is not None
        nonempty.update(i for i, gap in enumerate(g[0]) if gap)
    items: list[list] = []
    j = 0
    for i in range(len(consts) + 1):
        if i in nonempty:
            items.append(["v", j])
            j += 1
        if i < len(consts):
            items.append(["c", consts[i][0], consts[i][1]])
    return items


def anti_unify(calls: list[ToolCall]) -> dict:
    tpl: dict = {"tool": calls[0].name, "args": {}}
    for k in sorted(calls[0].args):
        vals = [c.args.get(k) for c in calls]
        sts = [shell_text(k, v) for v in vals]
        if sts[0] is not None and all(st is not None and st[1] == sts[0][1] for st in sts):
            tpl["args"][k] = {"k": "sh", "items": unify_tokens([tokenize(st[0]) for st in sts])}
            if sts[0][1] is not None:
                tpl["args"][k]["wrap"] = sts[0][1]
        elif all(canon(v) == canon(vals[0]) for v in vals):
            tpl["args"][k] = {"k": "c", "v": vals[0]}
        else:
            tpl["args"][k] = {"k": "v"}
    return tpl


# ------------------------------------------------------------ matching


def match_sh(items: list[list], toks: list[Tok]) -> tuple[dict[int, list[Tok]], list[Tok], list[int]] | None:
    """Return ({var: gap tokens}, const anchor tokens, gap index per var) or None."""
    consts = [(it[1], it[2]) for it in items if it[0] == "c"]
    r = greedy_gaps(consts, toks)
    if r is None:
        return None
    gaps, anchors = r
    out, gap_of, assigned, g = {}, {}, set(), 0
    for it in items:
        if it[0] == "v":
            out[it[1]] = gaps[g]
            gap_of[it[1]] = g
            assigned.add(g)
        else:
            if g not in assigned and gaps[g]:
                return None
            g += 1
    if g not in assigned and gaps[g]:
        return None
    return out, anchors, [gap_of[j] for j in sorted(gap_of)]


def call_slots(tpl: dict, call: ToolCall) -> dict[str, Val] | None:
    """Values of every argument (and shell variable gap) of `call` under `tpl`, or None if it doesn't match."""
    if call.name != tpl["tool"] or set(call.args) != set(tpl["args"]):
        return None
    out: dict[str, Val] = {}
    for k, at in tpl["args"].items():
        v = call.args[k]
        kind = at["k"]
        if kind == "c":
            if canon(v) != canon(at["v"]):
                return None
            out[k] = Val(cook(v), js=v)
        elif kind == "v":
            out[k] = Val(cook(v), js=v)
        else:
            st = shell_text(k, v)
            if st is None or st[1] != at.get("wrap"):
                return None
            text = st[0]
            m = match_sh(at["items"], tokenize(text))
            if m is None:
                return None
            out[k] = Val(text, raw=text, js=v)
            for j, gap in m[0].items():
                out[f"{k}#{j}"] = Val(" ".join(t.val for t in gap), raw=text[gap[0].start : gap[-1].end] if gap else "")
    return out


def var_slots(tpl: dict) -> list[str]:
    out = []
    for k, at in tpl["args"].items():
        if at["k"] == "v":
            out.append(k)
        elif at["k"] == "sh":
            out.extend(f"{k}#{it[1]}" for it in at["items"] if it[0] == "v")
    return out


# ------------------------------------------------------------ rendering


def _coerce(cooked: str, like: Any) -> Any:
    if isinstance(like, bool):
        return cooked.strip().lower() in ("true", "1", "yes")
    if isinstance(like, int):
        try:
            return int(cooked)
        except ValueError:
            return cooked
    if isinstance(like, float):
        try:
            return float(cooked)
        except ValueError:
            return cooked
    if isinstance(like, (list, dict)):
        try:
            return json.loads(cooked)
        except json.JSONDecodeError:
            return cooked
    return cooked


def _shell_raw(val: Val, gap: list[Tok], ref: str) -> str:
    if val.raw is not None:
        return val.raw
    if len(gap) > 1 and " " in val.cooked.strip():
        return " ".join(quote(p) for p in val.cooked.split())
    like = ref[gap[0].start : gap[0].end] if gap else None
    return quote(val.cooked, like)


def render(tpl: dict, ref_args: dict, values: dict[str, Val]) -> dict:
    """Instantiate `tpl` using `values` for its variable slots; shell text is spliced into `ref_args`."""
    args: dict = {}
    for k, at in tpl["args"].items():
        kind = at["k"]
        if kind == "c":
            args[k] = copy.deepcopy(at["v"])
        elif kind == "v":
            val = values[k]
            args[k] = copy.deepcopy(val.js) if val.has_js() else _coerce(val.cooked, ref_args.get(k))
        else:
            st = shell_text(k, ref_args[k])
            if st is None:
                raise ValueError("reference call does not match its template")
            ref = st[0]
            toks = tokenize(ref)
            m = match_sh(at["items"], toks)
            if m is None:
                raise ValueError("reference call does not match its template")
            gaps, anchors, _ = m
            # anchor tokens bound each gap: gap g sits between anchors[g-1] and anchors[g]
            gap_index = _gap_indices(at["items"])
            edits = []
            for j, gap in gaps.items():
                val = values[f"{k}#{j}"]
                new = _shell_raw(val, gap, ref)
                if gap:
                    start, end = gap[0].start, gap[-1].end
                    if ref[start:end] == new:
                        continue
                    if not new:  # drop the token and one adjacent space
                        if start > 0 and ref[start - 1] == " ":
                            start -= 1
                        elif end < len(ref) and ref[end] == " ":
                            end += 1
                    edits.append((start, end, new))
                elif new:
                    g = gap_index[j]
                    if g > 0:
                        pos = anchors[g - 1].end
                        edits.append((pos, pos, " " + new))
                    else:
                        pos = anchors[0].start if anchors else 0
                        edits.append((pos, pos, new + (" " if anchors else "")))
            out = ref
            for start, end, new in sorted(edits, key=lambda e: -e[0]):
                out = out[:start] + new + out[end:]
            args[k] = unwrap(out, at.get("wrap"))
    return args


def _gap_indices(items: list[list]) -> dict[int, int]:
    out, g = {}, 0
    for it in items:
        if it[0] == "v":
            out[it[1]] = g
        else:
            g += 1
    return out


def label(tpl: dict, width: int = 70) -> str:
    parts = []
    for k, at in tpl["args"].items():
        if at["k"] == "c":
            parts.append(f"{k}={short(cook(at['v']), 40)}")
        elif at["k"] == "v":
            parts.append(f"{k}=${k}")
        else:
            words = [(it[1] if it[1] != "\n" else "⏎") if it[0] == "c" else f"${it[1]}" for it in at["items"]]
            parts.append(short(" ".join(words), width))
    return short(f"{tpl['tool']}({', '.join(parts)})", width + 20)
