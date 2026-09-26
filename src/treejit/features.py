"""Observation features and the guard predicate set.

Predicates are deliberately few and hand-written: error/exit code, empty,
length bucket, JSON field equals, substring (regex-lite) and task keyword.
Guards are conjunctions of *stable* features (same value across every passing
example); branches are decision lists of single predicates (stumps).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .model import Observation

_EXIT_RE = re.compile(r"(?i)\bexit(?:ed with)?(?:\s+(?:code|status))?\s*[:=]?\s*(-?\d+)\b")
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "into", "please", "you", "your", "are", "was", "have", "has",
         "can", "will", "then", "them", "its", "our", "all", "any", "not", "but", "use", "using", "make", "sure"}


def len_bucket(n: int) -> int:
    return 0 if n == 0 else 1 if n < 100 else 2 if n < 1000 else 3 if n < 10000 else 4


def parse_json(text: str) -> Any:
    t = text.strip()
    if not t or t[0] not in "{[":
        return None
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None


def flatten_json(obj: Any, prefix: str = "", depth: int = 3, out: dict | None = None, limit: int = 60) -> dict:
    """Scalar leaves as {"a.b.0.c": value}."""
    out = {} if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(obj, dict):
        if depth == 0:
            return out
        for k, v in obj.items():
            flatten_json(v, f"{prefix}{k}.", depth - 1, out, limit)
    elif isinstance(obj, list):
        if depth == 0:
            return out
        for i, v in enumerate(obj[:10]):
            flatten_json(v, f"{prefix}{i}.", depth - 1, out, limit)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        out[prefix[:-1]] = obj
    return out


def obs_features(obs: Observation | None) -> dict:
    if obs is None:
        return {}
    text = obs.text or ""
    m = _EXIT_RE.search(text[:4000]) or _EXIT_RE.search(text[-2000:])
    code = int(m.group(1)) if m else None
    f: dict[str, Any] = {
        "err": bool(obs.is_error or (code is not None and code != 0)),
        "empty": not text.strip(),
        "lenb": len_bucket(len(text)),
    }
    if code is not None:
        f["exit"] = code
    js = parse_json(text)
    if isinstance(js, (dict, list)):
        for k, v in flatten_json(js).items():
            if v is None or isinstance(v, bool) or (isinstance(v, (int, float))) or (isinstance(v, str) and len(v) <= 64):
                f["json." + k] = v
    return f


def task_words(task: str) -> set[str]:
    return {w for w in _WORD_RE.findall(task.lower()) if w not in _STOP}


def lines_of(text: str, limit: int = 60) -> list[str]:
    out = []
    for line in text.splitlines()[:limit]:
        s = line.strip()
        if 3 <= len(s) <= 80:
            out.append(s)
    return out


# ---------------------------------------------------------------- predicates
# pred = ["feat", key, value] | ["contains", literal] | ["task", word]


def eval_pred(pred: list, feats: dict, text: str, words: set[str]) -> bool:
    kind = pred[0]
    if kind == "feat":
        return feats.get(pred[1], _MISSING) == pred[2]
    if kind == "contains":
        return pred[1] in text
    if kind == "task":
        return pred[1] in words
    return False


_MISSING = object()


def pred_label(pred: list) -> str:
    if pred[0] == "feat":
        return f"{pred[1]}=={json.dumps(pred[2])}"
    if pred[0] == "contains":
        return f"obs~{json.dumps(pred[1])}"
    return f"task~{pred[1]}"


_POST_KEYS = ("err", "empty", "exit")


def stable(feature_dicts: list[dict], keys: tuple, include_json: bool = True) -> dict:
    """Features with the same value across all dicts (the learned guard)."""
    if not feature_dicts:
        return {}
    out = {}
    for k, v in feature_dicts[0].items():
        if not (k in keys or (include_json and k.startswith("json."))):
            continue
        if all(d.get(k, _MISSING) == v for d in feature_dicts[1:]):
            out[k] = v
    return out


def guard_holds(guard: dict, feats: dict) -> bool:
    return all(feats.get(k, _MISSING) == v for k, v in guard.items())


def postcondition(feature_dicts: list[dict]) -> dict:
    return stable(feature_dicts, keys=_POST_KEYS, include_json=False)


def guard_of(feature_dicts: list[dict]) -> dict:
    return stable(feature_dicts, keys=_POST_KEYS, include_json=True)


# ---------------------------------------------------------------- decision lists


def learn_decision_list(examples: list[tuple[str, dict, str, set]], purity: float, max_rules: int = 6) -> dict | None:
    """examples: (label, feats, obs_text, task_words). Returns a decision list or None.

    Greedily picks the predicate isolating the largest pure-enough subset of the
    remaining examples, preferring predicates that are also false on examples of
    other labels (low leak), removes it and repeats. There is no catch-all
    default: an input no rule fires on goes to the model.
    """
    if len({e[0] for e in examples}) < 2 or len(examples) < 3:
        return None
    rules = []
    remaining = list(examples)
    while remaining and len(rules) < max_rules:
        best = None
        for pred in _candidates(remaining):
            hit = [e for e in remaining if eval_pred(pred, e[1], e[2], e[3])]
            if len(hit) < 2:
                continue
            counts: dict[str, int] = {}
            for e in hit:
                counts[e[0]] = counts.get(e[0], 0) + 1
            label, c = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            p = c / len(hit)
            if p < purity:
                continue
            leak = sum(1 for e in examples if e[0] != label and eval_pred(pred, e[1], e[2], e[3]))
            score = (c * p - leak, -_complexity(pred), pred_label(pred))
            if best is None or score > best[0]:
                best = (score, pred, label, p, len(hit), hit)
        if best is None:
            break
        _, pred, label, p, n, hit = best
        rules.append({"pred": pred, "edge": label, "purity": round(p, 4), "n": n})
        hit_ids = {id(e) for e in hit}
        remaining = [e for e in remaining if id(e) not in hit_ids]
    return {"rules": rules} if rules else None


def eval_decision_list(dl: dict, feats: dict, text: str, words: set[str]) -> dict | None:
    for r in dl.get("rules", []):
        if eval_pred(r["pred"], feats, text, words):
            return r
    return None


def _complexity(pred: list) -> int:
    if pred[0] == "feat":
        return 0 if pred[1] in ("err", "empty", "exit") else 1
    return 2 if pred[0] == "task" else 3


def _candidates(examples: list) -> list[list]:
    feat_vals: dict[tuple, int] = {}
    contains: dict[str, int] = {}
    words: dict[str, int] = {}
    for _, feats, text, tw in examples:
        for k, v in feats.items():
            if k == "lenb" or isinstance(v, float):
                continue
            feat_vals[(k, json.dumps(v))] = feat_vals.get((k, json.dumps(v)), 0) + 1
        for line in set(lines_of(text)):
            contains[line] = contains.get(line, 0) + 1
        for w in tw:
            words[w] = words.get(w, 0) + 1
    n = len(examples)
    preds: list[list] = []
    for (k, v), c in feat_vals.items():
        preds.append(["feat", k, json.loads(v)])
    for line, c in sorted(contains.items(), key=lambda kv: -kv[1])[:150]:
        if c > 1 or n <= 4:
            preds.append(["contains", line])
    for w, c in sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))[:150]:
        preds.append(["task", w])
    return preds
