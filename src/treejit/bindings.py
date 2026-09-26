"""Provenance search: bind template variables to $task, $obs[-k] or $arg[-k].

For each variable at a node, candidate rules are proposed from one passing
instance and kept only if they reproduce the value in every other instance.
A variable with no consistent rule is a hole (filled at T3, or served by the model).

Rules (JSON lists):
  ["const", value, raw]                    same value every time
  ["arg", k, slot]                         copy an argument of the call k steps back
  ["x", source, extractor]                 extract from "task" or ["obs", k]
      extractors: ["whole"] ["json", path] ["kv", key] ["after", word] ["re", name, i] ["line", i] ["tok", i]
  ["fmt", [str | rule, ...]]               string template with bound gaps
  ["case", decision_list, {label: [value, raw]}]   value chosen by observation/task predicates
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .features import eval_decision_list, learn_decision_list, obs_features, parse_json, task_words
from .model import Observation, ToolCall
from .templates import Val, common_subsequence, cook

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "url": re.compile(r"https?://[^\s'\"<>)]+"),
    "version": re.compile(r"\bv?(\d+\.\d+\.\d+(?:[-+][\w.]+)?)\b"),
    "path": re.compile(r"(?:\.{0,2}/)?(?:[\w.-]+/)*[\w-][\w.-]*\.[A-Za-z][A-Za-z0-9]{0,7}\b"),
    "sha": re.compile(r"\b[0-9a-f]{7,40}\b"),
    "id": re.compile(r"#?\b[A-Z]{1,4}\d{3,}\b"),
    "snake_id": re.compile(r"\b[a-z]+(?:_[a-z]+)+_\d+\b"),
    "quoted": re.compile(r"(?<!\w)\"([^\"\n]+)\"(?!\w)|(?<!\w)'([^'\n]+)'(?!\w)|`([^`\n]+)`"),
    "number": re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])"),
}
_STRIP = ".,;:!?()[]{}\"'<>"
# format templates are tried at two granularities: paths/versions as one token, then split on punctuation
_FINE = (re.compile(r"#?\w+(?:[.\-/@:]\w+)*|\s+|.", re.S), re.compile(r"\w+|\s+|.", re.S))


@dataclass
class Sources:
    """What a binding rule may read at step i: the task and everything before step i."""

    task: str
    calls: list[ToolCall]
    obs: list[Observation | None]
    slots: list[dict | None]

    def text(self, src: Any) -> str | None:
        if src == "task":
            return self.task
        k = src[1]
        if len(self.obs) >= k and self.obs[-k] is not None:
            return self.obs[-k].text
        return None


def specific(v: str) -> bool:
    """Values worth binding even when constant so far (paths, ids, numbers, long strings)."""
    return len(v) >= 8 or bool(re.search(r"[/.\d@#]", v))


# ------------------------------------------------------------ extractors


def _lines(t: str) -> list[str]:
    return [line.strip() for line in t.splitlines() if line.strip()]


def _toks(t: str) -> list[str]:
    return [w for w in (x.strip(_STRIP) for x in t.split()) if w]


def _json_get(js: Any, path: list) -> Any:
    cur = js
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and -len(cur) <= p < len(cur):
            cur = cur[p]
        else:
            return _MISS
    return cur


_MISS = object()


def _re_all(name: str, t: str) -> list[str]:
    return [m.group(m.lastindex) if m.lastindex else m.group(0) for m in PATTERNS[name].finditer(t)]


def _kv_re(key: str) -> re.Pattern:
    return re.compile(rf"(?im)^[ \t>*-]*{re.escape(key)}[ \t]*[:=][ \t]*(.+?)[ \t]*\.?[ \t]*$")


def extract(t: str, ext: list) -> Val | None:
    kind = ext[0]
    if kind == "whole":
        v = t.strip()
        return Val(v) if v else None
    if kind == "json":
        js = parse_json(t)
        if js is None:
            return None
        v = _json_get(js, ext[1])
        return None if v is _MISS else Val(cook(v), js=v)
    if kind == "kv":
        m = _kv_re(ext[1]).search(t)
        return Val(m.group(1).strip()) if m else None
    if kind == "after":
        toks = _toks(t)
        for j in range(len(toks) - 1):
            if toks[j].lower().rstrip(":#") == ext[1]:
                return Val(toks[j + 1])
        return None
    if kind == "re":
        ms = _re_all(ext[1], t)
        i = ext[2]
        return Val(ms[i]) if -len(ms) <= i < len(ms) else None
    if kind == "line":
        ls = _lines(t)
        i = ext[1]
        return Val(ls[i]) if -len(ls) <= i < len(ls) else None
    if kind == "tok":
        ts = _toks(t)
        i = ext[1]
        return Val(ts[i]) if -len(ts) <= i < len(ts) else None
    return None


def _flat_args(call: ToolCall) -> dict[str, Val]:
    return {k: Val(cook(v), js=v) for k, v in call.args.items()}


def eval_rule(rule: list, S: Sources) -> Val | None:
    kind = rule[0]
    if kind == "const":
        v = rule[1]
        return Val(cook(v), raw=rule[2] if len(rule) > 2 else None, js=v)
    if kind == "arg":
        k, slot = rule[1], rule[2]
        if len(S.calls) < k:
            return None
        slots = S.slots[-k] if S.slots[-k] is not None else _flat_args(S.calls[-k])
        v = slots.get(slot)
        return None if v is None else Val(v.cooked, v.raw, v.js)
    if kind == "x":
        t = S.text(rule[1])
        return None if t is None else extract(t, rule[2])
    if kind == "case":
        feats, text, words = _case_inputs(S)
        leaf = eval_decision_list(rule[1], feats, text, words)
        if leaf is None:
            return None
        js, raw = rule[2][leaf["edge"]]
        return Val(cook(js), raw=raw, js=js)
    if kind == "fmt":
        out = []
        for p in rule[1]:
            if isinstance(p, str):
                out.append(p)
            else:
                v = eval_rule(p, S)
                if v is None or not v.cooked:
                    return None
                out.append(v.cooked)
        return Val("".join(out))
    return None


# ------------------------------------------------------------ candidate generation


def _json_paths(js: Any, target: str, prefix: list | None = None, depth: int = 4) -> list[list]:
    prefix = prefix or []
    out = [prefix] if prefix and cook(js) == target else []
    if isinstance(js, dict) and depth:
        for k, v in js.items():
            out.extend(_json_paths(v, target, prefix + [k], depth - 1))
    elif isinstance(js, list) and depth:
        for i, v in enumerate(js[:20]):
            out.extend(_json_paths(v, target, prefix + [i], depth - 1))
    return out


def _source_candidates(src: Any, t: str, target: str) -> list[list]:
    c: list[list] = []
    js = parse_json(t)
    if t.strip() == target:
        c.append(["x", src, ["whole"]])
    if js is not None:
        for path in _json_paths(js, target)[:3]:
            c.append(["x", src, ["json", path]])
    if target not in t:
        return c
    for m in re.finditer(r"(?im)^[ \t>*-]*([A-Za-z][\w .-]{0,30}?)[ \t]*[:=][ \t]*(.+?)[ \t]*\.?[ \t]*$", t):
        if m.group(2).strip() == target:
            c.append(["x", src, ["kv", m.group(1).strip()]])
    for name in PATTERNS:
        ms = _re_all(name, t)
        if target in ms:
            i = ms.index(target)
            c.append(["x", src, ["re", name, i]])
            if i == len(ms) - 1 and i > 0:
                c.append(["x", src, ["re", name, -1]])
    toks = _toks(t)
    for j, w in enumerate(toks):
        if w == target and j > 0:
            anchor = toks[j - 1].lower().rstrip(":#")
            if re.fullmatch(r"[a-z][\w-]*", anchor):
                c.append(["x", src, ["after", anchor]])
    ls = _lines(t)
    if target in ls:
        i = ls.index(target)
        c.append(["x", src, ["line", i]])
        c.append(["x", src, ["line", i - len(ls)]])
    if target in toks:
        i = toks.index(target)
        c.append(["x", src, ["tok", i]])
        c.append(["x", src, ["tok", i - len(toks)]])
    return c


def candidates(v: Val, S: Sources) -> list[list]:
    target = v.cooked
    if not target:
        return []
    c: list[list] = []
    for k in range(1, min(3, len(S.calls)) + 1):
        slots = S.slots[-k] if S.slots[-k] is not None else _flat_args(S.calls[-k])
        for name, val in slots.items():
            if val.cooked == target:
                c.append(["arg", k, name])
    for src in (["obs", 1], ["obs", 2], ["obs", 3], "task"):
        t = S.text(src)
        if t:
            c.extend(_source_candidates(src, t, target))
    return c


def find_rule(values: list[Val], srcs: list[Sources], allow_fmt: bool = True, shell_slot: bool = False,
              min_share: float = 0.6) -> list | None:
    """A rule reproducing the value in every instance, else one that is never wrong
    (correct or abstaining) and correct on at least `min_share` of them. A rule that
    abstains at replay time sends that step to the model, so a node that mixes two
    task types keeps a rule for the majority instead of becoming a hole."""
    v0 = values[0]
    n = len(values)
    same = all(v.cooked == v0.cooked for v in values)
    const = ["const", v0.js if v0.has_js() else v0.cooked, v0.raw] if v0.raw is not None else ["const", v0.js if v0.has_js() else v0.cooked]
    if same and not specific(v0.cooked):
        return const
    seeds = sorted({0, n - 1, n // 2})
    seen: set[str] = set()
    partial, partial_ok = None, 0
    for si in seeds:
        for rule in candidates(values[si], srcs[si]):
            key = repr(rule)
            if key in seen:
                continue
            seen.add(key)
            ok = wrong = 0
            for s, v in zip(srcs, values):
                got = eval_rule(rule, s)
                if got is None:
                    continue
                if got.cooked == v.cooked:
                    ok += 1
                else:
                    wrong += 1
                    break
            if wrong:
                continue
            if ok == n:
                return rule
            if ok > partial_ok:
                partial, partial_ok = rule, ok
    if same:
        return const
    if allow_fmt:
        r = _find_fmt(values, srcs)
        if r is not None:
            return r
    if partial is not None and partial_ok >= max(2, min_share * n):
        return partial
    r = _case_rule(values, srcs)
    if r is not None:
        return r
    if shell_slot:
        return _mode_flag(values)
    return None


def _case_inputs(S: Sources) -> tuple[dict, str, set]:
    last = S.obs[-1] if S.obs else None
    return obs_features(last) if S.obs else {}, (last.text if last is not None else ""), task_words(S.task)


def _case_rule(values: list[Val], srcs: list[Sources], purity: float = 0.8, max_values: int = 4) -> list | None:
    """A few distinct constant values, chosen by what the model saw: `ls healthy` vs `ls broken`."""
    labels: dict[str, list] = {}
    for v in values:
        labels.setdefault(v.cooked, [v.js if v.has_js() else v.cooked, v.raw])
    if not 1 < len(labels) <= max_values:
        return None
    if any(sum(1 for v in values if v.cooked == k) < 2 for k in labels):
        return None
    examples = [(v.cooked, *_case_inputs(s)) for v, s in zip(values, srcs)]
    dl = learn_decision_list(examples, purity)
    if dl is None or {r["edge"] for r in dl["rules"]} != set(labels):
        return None
    return ["case", dl, labels]


def _mode_flag(values: list[Val], share: float = 0.75) -> list | None:
    """Harmless formatting variance in shell flags (`git status` vs `git status --short`): use the majority form."""
    counts: dict[str, int] = {}
    for v in values:
        counts[v.cooked] = counts.get(v.cooked, 0) + 1
    top, c = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    flaglike = all(not v.cooked or all(p.startswith("-") for p in v.cooked.split()) for v in values)
    if flaglike and c / len(values) >= share:
        raw = next(v.raw for v in values if v.cooked == top)
        return ["const", top, raw]
    return None


def _same(a: Val | None, b: Val) -> bool:
    return a is not None and a.cooked == b.cooked


def _find_fmt(values: list[Val], srcs: list[Sources]) -> list | None:
    for tokenizer in _FINE:
        r = _find_fmt_with(values, srcs, tokenizer)
        if r is not None:
            return r
    return None


def _find_fmt_with(values: list[Val], srcs: list[Sources], tokenizer: re.Pattern) -> list | None:
    seqs = [tokenizer.findall(v.cooked) for v in values]
    if any(len(s) > 80 or len(s) < 2 for s in seqs):
        return None
    consts = common_subsequence(seqs)
    if not consts:
        return None
    gaps_per: list[list[str]] = []
    for seq in seqs:
        gaps, p = [], 0
        for c in consts:
            q = p
            while q < len(seq) and seq[q] != c:
                q += 1
            if q == len(seq):
                return None
            gaps.append("".join(seq[p:q]))
            p = q + 1
        gaps.append("".join(seq[p:]))
        gaps_per.append(gaps)
    var_pos = sorted({i for gaps in gaps_per for i, g in enumerate(gaps) if g})
    if not var_pos or len(var_pos) > 3:
        return None
    parts: list = []
    for i in range(len(consts) + 1):
        if i in var_pos:
            gv = [Val(gaps[i]) for gaps in gaps_per]
            if any(not v.cooked for v in gv):
                return None
            r = find_rule(gv, srcs, allow_fmt=False)
            if r is None or r[0] == "const":
                return None
            parts.append(r)
        if i < len(consts):
            if parts and isinstance(parts[-1], str):
                parts[-1] += consts[i]
            else:
                parts.append(consts[i])
    return ["fmt", parts]


def rule_label(rule: list | None) -> str:
    if rule is None:
        return "?hole"
    kind = rule[0]
    if kind == "const":
        return repr(rule[1])[:40]
    if kind == "arg":
        return f"$arg[-{rule[1]}].{rule[2]}"
    if kind == "x":
        src = "$task" if rule[1] == "task" else f"$obs[-{rule[1][1]}]"
        ext = rule[2]
        return f"{src}.{ext[0]}" + ("" if len(ext) == 1 else "(" + ",".join(str(x) for x in ext[1:]) + ")")
    if kind == "case":
        return "case(" + ", ".join(f"{json.dumps(r['edge'])} if {r['pred'][0]}" for r in rule[1]["rules"]) + ")"
    if kind == "fmt":
        return "f'" + "".join(p if isinstance(p, str) else "{" + rule_label(p) + "}" for p in rule[1]) + "'"
    return str(rule)
