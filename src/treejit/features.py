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


def excess_negatives(neg: float, confirmed: float, purity: float) -> float:
    """Failed replays beyond the failure rate tolerated among all replays of the same choice.

    A failed run fails every replayed step in it, not just the one that went wrong, so a few
    failures among many passing replays are noise elsewhere in the run; a failure rate above
    1 - purity is evidence against the choice itself."""
    x = neg - (1.0 - purity) * (neg + confirmed)
    return x if x > _EPS else 0.0  # 1 - 0.2*5 is 2.2e-16 in floats, not a negative


_EPS = 1e-9
MAX_CLASS_SETS = 40  # task-word sets kept per task-word rule, for the similarity gate


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _class_sets(rule: dict, examples: list, negatives: list) -> None:
    """Store on a task-word rule what the similarity gate compares an input with (see rule_resembles)."""
    pred, label = rule["pred"], rule["edge"]
    sup = [frozenset(e[3]) for e in examples if e[0] == label and eval_pred(pred, e[1], e[2], e[3])]
    counts: dict[str, int] = {}
    for w in sup:
        for x in w:
            counts[x] = counts.get(x, 0) + 1
    ex: dict[frozenset, None] = {}  # distinct, ordered by most recent occurrence
    for w in sup:
        c = frozenset(x for x in w if counts[x] >= 2)
        ex.pop(c, None)
        if c:
            ex[c] = None
    passed = set(sup)
    nx: dict[frozenset, None] = {}
    for e in negatives:
        w = frozenset(e[3])
        if e[0] == label and eval_pred(pred, e[1], e[2], e[3]) and w not in passed:
            nx.pop(w, None)
            nx[w] = None
    rule["ex"] = [sorted(w) for w in list(ex)[-MAX_CLASS_SETS:]]
    if nx:
        rule["nx"] = [sorted(w) for w in list(nx)[-MAX_CLASS_SETS:]]


def rule_resembles(rule: dict, words: set, threshold: float) -> bool:
    """The similarity gate for a task-word rule: nearest neighbours over task-word sets.

    A task word that separated the evidence so far may be chance (seed 3: `src` in every delete
    task seen, and in no typo task yet). The rule is trusted only on inputs like those that
    support it: the best Jaccard similarity to a supporting example (`ex`) must reach
    `threshold` and beat the best similarity to an input the rule replayed into a failed run
    (`nx`). Anything else is a new input class, which a T2 call labels once.

    Supporting examples keep only the words at least two of them share: a word seen once among
    them (a file name, a typo word, a version) says nothing about the class, and would make
    every task of a known kind look new. The input keeps all its words, since a word the rule
    has never seen is exactly what marks a new kind of task. Failed inputs keep theirs too.
    threshold <= 0 turns the gate off; a rule without stored sets (built by an older version)
    fails it."""
    if threshold <= 0:
        return True
    ex = rule.get("ex")
    if not ex:
        return False
    pos = max(jaccard(words, set(e)) for e in ex)
    neg = max((jaccard(words, set(e)) for e in rule.get("nx", ())), default=None)
    return pos >= threshold and (neg is None or pos > neg)


def learn_decision_list(examples: list[tuple[str, dict, str, set]], purity: float, max_rules: int = 6,
                        negatives: list[tuple[str, dict, str, set]] | None = None,
                        confirmed: list[tuple[str, dict, str, set]] | None = None,
                        class_sets: bool = False) -> dict | None:
    """examples: (label, feats, obs_text, task_words). Returns a decision list or None.

    Greedily picks the predicate isolating the largest pure-enough subset of the
    remaining examples, preferring predicates that are also false on examples of
    other labels (low leak), removes it and repeats. There is no catch-all
    default: an input no rule fires on goes to the model.

    negatives: inputs where replaying `label` here ended in a failed run; confirmed: the
    same for passing runs. Replayed steps never become examples, so without negatives a
    wrong rule could never be refuted: the inputs it misroutes stop producing evidence.
    A rule predicting `label` counts the matching negatives in excess of the tolerated
    failure rate (`excess_negatives`) as misses: lower purity, more leak, and a `neg`
    field that raises the support it needs before T1 replays on it (replay.rule_support).

    class_sets: a task-word rule also keeps, for the similarity gate `rule_resembles`, the
    distinct task-word sets of the examples that support it (`ex`: the most recent
    MAX_CLASS_SETS, each cut to the words at least two of them share) and of the negatives it
    matches (`nx`, minus any set that also supports it: the same words both passed and
    failed, so words can't tell them apart).
    """
    if len({e[0] for e in examples}) < 2 or len(examples) < 3:
        return None
    negatives = negatives or []
    confirmed = confirmed or []

    def matching(pool: list, pred: list, label: str) -> int:
        return sum(1 for e in pool if e[0] == label and eval_pred(pred, e[1], e[2], e[3]))

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
            neg = 0.0
            if negatives:
                bad = matching(negatives, pred, label)
                neg = excess_negatives(bad, matching(confirmed, pred, label), purity) if bad else 0.0
            p = c / (len(hit) + neg)
            if p < purity:
                continue
            leak = neg + sum(1 for e in examples if e[0] != label and eval_pred(pred, e[1], e[2], e[3]))
            score = (c * p - leak, -_complexity(pred), pred_label(pred))
            if best is None or score > best[0]:
                best = (score, pred, label, p, len(hit), hit, neg)
        if best is None:
            break
        _, pred, label, p, n, hit, neg = best
        # support: every example (not only those left for this rule) where the predicate holds and the model chose `label`
        rule = {"pred": pred, "edge": label, "purity": round(p, 4), "n": n, "support": matching(examples, pred, label)}
        if neg:
            rule["neg"] = round(neg, 2)
        if class_sets and pred[0] == "task":
            _class_sets(rule, examples, negatives)
        rules.append(rule)
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
