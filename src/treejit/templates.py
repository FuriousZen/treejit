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
    return key in SHELL_KEYS and isinstance(value, str)


def shape_of(call: ToolCall) -> str:
    parts = []
    for k in sorted(call.args):
        v = call.args[k]
        parts.append([k, command_heads(v)] if is_shell_arg(k, v) else [k])
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
        common = _common_step(common, seq)
    return common


def _common_step(common: list, seq: list) -> list:
    sm = difflib.SequenceMatcher(None, common, seq, autojunk=False)
    return [x for blk in sm.get_matching_blocks() for x in common[blk.a : blk.a + blk.size]]


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
        if is_shell_arg(k, vals[0]) and all(isinstance(v, str) for v in vals):
            tpl["args"][k] = {"k": "sh", "items": unify_tokens([tokenize(v) for v in vals])}
        elif all(canon(v) == canon(vals[0]) for v in vals):
            tpl["args"][k] = {"k": "c", "v": vals[0]}
        else:
            tpl["args"][k] = {"k": "v"}
    return tpl


class TemplateFold:
    """`anti_unify` as a resumable left fold, so a rebuild whose instances only grew at the end
    folds just the new calls. `extend` never mutates: it returns a new fold (safe to share).

    Shell arguments fold their common token subsequence one call at a time, exactly as
    `common_subsequence` does; the gaps that are non-empty in some call are recomputed over all
    calls only when that subsequence changed. `extend` returns None where the fold can't follow
    `anti_unify` (a shell argument that isn't a string in a later call); build from scratch then.
    """

    __slots__ = ("keys", "first", "sh_vals", "consts", "nonempty", "canon0", "same")

    def __init__(self) -> None:
        self.keys: list = []                      # identity of each folded call, in order
        self.first: ToolCall | None = None
        self.sh_vals: dict[str, list[str]] = {}   # shell arg -> every value, in order
        self.consts: dict[str, list] = {}         # shell arg -> common token subsequence so far
        self.nonempty: dict[str, set[int]] = {}   # shell arg -> gap indices non-empty in some call
        self.canon0: dict[str, str] = {}          # other arg -> canonical form of the first value
        self.same: dict[str, bool] = {}           # other arg -> every value equals the first so far

    def extend(self, calls: list[ToolCall], keys: list) -> "TemplateFold | None":
        if not calls:
            return self
        new = TemplateFold()
        new.keys = self.keys + list(keys)
        new.first = self.first
        new.sh_vals = {k: list(v) for k, v in self.sh_vals.items()}
        new.consts = dict(self.consts)
        new.nonempty = {k: set(v) for k, v in self.nonempty.items()}
        new.canon0 = dict(self.canon0)
        new.same = dict(self.same)
        rest = calls
        if new.first is None:
            new.first, rest = calls[0], calls[1:]
            for k in sorted(new.first.args):
                v = new.first.args[k]
                if is_shell_arg(k, v):
                    new.sh_vals[k] = [v]
                    new.consts[k] = _tokseq(tokenize(v))
                    new.nonempty[k] = set()
                else:
                    new.canon0[k] = canon(v)
                    new.same[k] = True
            fresh_from = 0
        else:
            fresh_from = len(self.keys)
        for k, vals in new.sh_vals.items():
            old = new.consts[k]
            common = old
            added = [c.args.get(k) for c in rest]
            if any(not isinstance(v, str) for v in added):
                return None
            for v in added:
                common = _common_step(common, _tokseq(tokenize(v)))
            vals.extend(added)
            new.consts[k] = common
            # gaps of calls already folded stay valid while the subsequence is unchanged
            if fresh_from and common == old:
                todo = vals[fresh_from:]
            else:
                todo, new.nonempty[k] = vals, set()
            for v in todo:
                g = greedy_gaps(common, tokenize(v))
                assert g is not None
                new.nonempty[k].update(i for i, gap in enumerate(g[0]) if gap)
        for k in new.canon0:
            if new.same[k]:
                new.same[k] = all(canon(c.args.get(k)) == new.canon0[k] for c in rest)
        return new

    def template(self) -> dict:
        assert self.first is not None
        tpl: dict = {"tool": self.first.name, "args": {}}
        for k in sorted(self.first.args):
            if k in self.sh_vals:
                consts, nonempty = self.consts[k], self.nonempty[k]
                items: list[list] = []
                j = 0
                for i in range(len(consts) + 1):
                    if i in nonempty:
                        items.append(["v", j])
                        j += 1
                    if i < len(consts):
                        items.append(["c", consts[i][0], consts[i][1]])
                tpl["args"][k] = {"k": "sh", "items": items}
            elif self.same[k]:
                tpl["args"][k] = {"k": "c", "v": self.first.args[k]}
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
            if not isinstance(v, str):
                return None
            m = match_sh(at["items"], tokenize(v))
            if m is None:
                return None
            out[k] = Val(v, raw=v, js=v)
            for j, gap in m[0].items():
                out[f"{k}#{j}"] = Val(" ".join(t.val for t in gap), raw=v[gap[0].start : gap[-1].end] if gap else "")
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
            ref = ref_args[k]
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
            args[k] = out
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
