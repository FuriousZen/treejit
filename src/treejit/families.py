"""Tree keying: family = tool schemas + the *stable lines* of the system prompt.

Harnesses like Claude Code embed cwd, date, platform and git status in the system
prompt, sometimes near the top, so neither hashing it whole nor a common prefix
works. Instead each prompt is cut into lines, each line is normalized (volatile
tokens such as dates, times, absolute paths, hashes, uuids and numbers are masked;
git status and commit-log lines collapse to one placeholder each) and hashed. A
family keeps, per masked line, how many of its members contained it, and each
line weighs its length in characters.

  stable line   present in >= STABLE of the family's members (a 1-member family:
                every line of it)
  join          the prompt contains >= COVER of the family's stable chars AND the
                family's stable lines make up >= OWN of the prompt's own chars

When several families of the same tool set qualify, the highest coverage score
wins, then the oldest family, then the smallest id. A member is a distinct raw
system prompt (a session repeating its prompt on every request counts once);
`family_members` remembers which family each one went to, so a prompt never moves.
Family ids never change. Families written by the old longest-common-prefix code
(only `prefix` set) are seeded from that prefix as a 1-member family and keep the
old `startswith(prefix)` rule as a fallback, so they resolve to the same id.

`families.prefix` holds the family's stable lines (masked) for `treejit show`.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from .store import Store
from .util import canon, h, now

STABLE = 0.8     # share of members a line must appear in to be stable
COVER = 0.9      # share of the family's stable chars the prompt must contain
OWN = 0.8        # share of the prompt's chars the family's stable lines must cover
HALVE_AT = 64    # member count at which counts are halved (sliding window; drops one-off lines)
LEGACY_MIN = 64  # old prefix-based families: shorter prefixes never matched by startswith
MEMO_MAX = 4096  # in-memory (tools, system) -> family memo

# ------------------------------------------------------------------ line normalization

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_MASKS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bhttps?://\S+"), "<url>"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<email>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"), "<date>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "<date>"),
    (re.compile(rf"\b{_MONTH} \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}\b|\b\d{{1,2}} {_MONTH},? \d{{4}}\b"), "<date>"),
    (re.compile(r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b"), "<day>"),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s?[AaPp][Mm])?\b"), "<time>"),
    # absolute paths (unix, ~, windows); a slash inside a word ("and/or") is not a path
    (re.compile(r"(?:(?<![\w./\\-])(?:~|[A-Za-z]:)?/|(?<![\w])[A-Za-z]:\\)[^\s\"'`()<>\[\]{},;]*"), "<path>"),
    # hex ids (commit hashes, digests): at least one digit and one letter, 7+ chars
    (re.compile(r"\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{7,}\b"), "<hex>"),
    (re.compile(r"\d+(?:\.\d+)*"), "<n>"),
]
# whole lines of git output (Claude Code's gitStatus block): collapse to one placeholder each
_GIT_STATUS = re.compile(r"^[MADRCUT?!]{1,2}\s+\S+(?:\s+->\s+\S+)?$")
_GIT_COMMIT = re.compile(r"^[*|\\/ ]*[0-9a-f]{7,40}\s")
_BRANCH = re.compile(r"^((?:Current branch|On branch|Branch)\s*:?)\s+\S+$")
_PLACEHOLDER = re.compile(r"<(?:url|email|uuid|date|day|time|path|hex|n|git-status|git-commit)>")
_LETTERS = re.compile(r"[A-Za-z]")
MIN_LETTERS = 4  # a masked line with fewer letters is a data line (file list, number): counted once


def normalize(line: str) -> str:
    s = " ".join(line.split())
    if not s:
        return ""
    if _GIT_STATUS.match(s):
        return "<git-status>"
    if _GIT_COMMIT.match(s):
        return "<git-commit>"
    m = _BRANCH.match(s)
    if m:
        return m.group(1) + " <branch>"
    for rx, rep in _MASKS:
        s = rx.sub(rep, s)
    return s


def lines_of(text: str) -> list[tuple[str, str, int]]:
    """[(key, masked line, weight)] in prompt order, one entry per key. The k-th
    repeat of an instruction line is its own key (repeats keep their weight); data
    lines (git output, bare paths/numbers) are counted once however often they occur."""
    out: list[tuple[str, str, int]] = []
    seen: dict[str, int] = {}
    for raw in text.splitlines():
        m = normalize(raw)
        if not m:
            continue
        k = seen.get(m, 0)
        data = len(_LETTERS.findall(_PLACEHOLDER.sub("", m))) < MIN_LETTERS
        if k and data:
            continue
        seen[m] = k + 1
        out.append((h(m, k), m, len(m)))
    return out


# ------------------------------------------------------------------ in-memory family state


@dataclass
class _Fam:
    id: str
    created: float
    n: int = 0
    cnt: dict[str, int] = field(default_factory=dict)
    w: dict[str, int] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)
    pos: dict[str, int] = field(default_factory=dict)
    legacy: str | None = None
    stable: dict[str, int] = field(default_factory=dict)  # key -> weight
    stable_chars: int = 0

    def restable(self) -> None:
        need = STABLE * self.n - 1e-9
        self.stable = {k: self.w[k] for k, c in self.cnt.items() if c >= need}
        self.stable_chars = sum(self.stable.values())

    def score(self, keys: dict[str, int], total: int) -> tuple[float, float]:
        """(share of the family's stable chars in the prompt, share of the prompt covered by stable lines)"""
        common = sum(w for k, w in self.stable.items() if k in keys)
        a = common / self.stable_chars if self.stable_chars else (1.0 if total == 0 else 0.0)
        b = common / total if total else (1.0 if self.stable_chars == 0 else 0.0)
        return a, b

    def display(self) -> str:
        ks = sorted(self.stable, key=lambda k: self.pos.get(k, 0))
        return "".join(self.text[k] + "\n" for k in ks)


class _Cache:
    def __init__(self) -> None:
        self.fams: dict[str, tuple[tuple, list[_Fam]]] = {}  # tools_hash -> (stamp, families)
        self.memo: OrderedDict[str, str] = OrderedDict()     # h(tools_hash, system) -> family id


def _cache(store: Store) -> _Cache:
    c = getattr(store, "_families_cache", None)
    if c is None:
        c = _Cache()
        store._families_cache = c  # type: ignore[attr-defined]
    return c


def _stamp(store: Store, tools_hash: str) -> tuple:
    return tuple((r["id"], r["n_members"], r["updated"]) for r in
                 store.q("SELECT id, n_members, updated FROM families WHERE tools_hash=? ORDER BY created, id", (tools_hash,)))


def _load(store: Store, tools_hash: str) -> list[_Fam]:
    c = _cache(store)
    stamp = _stamp(store, tools_hash)
    hit = c.fams.get(tools_hash)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    fams: list[_Fam] = []
    rows = store.q("SELECT id, prefix, created, n_members, legacy_prefix FROM families WHERE tools_hash=? ORDER BY created, id",
                   (tools_hash,))
    for r in rows:
        f = _Fam(r["id"], r["created"] or 0.0, n=r["n_members"] or 0, legacy=r["legacy_prefix"])
        if f.n == 0:  # written by the prefix-based keying: seed from the stored prefix
            _seed_legacy(store, f, r["prefix"] or "")
        else:
            for lr in store.q("SELECT line, n, chars, text, pos FROM family_lines WHERE family=?", (f.id,)):
                f.cnt[lr["line"]], f.w[lr["line"]] = lr["n"], lr["chars"]
                f.text[lr["line"]], f.pos[lr["line"]] = lr["text"], lr["pos"]
        f.restable()
        fams.append(f)
    c.fams[tools_hash] = (_stamp(store, tools_hash), fams)
    return fams


def _seed_legacy(store: Store, f: _Fam, prefix: str) -> None:
    f.n, f.legacy = 1, prefix
    for i, (k, m, w) in enumerate(lines_of(prefix)):
        f.cnt[k], f.w[k], f.text[k], f.pos[k] = 1, w, m, i
    f.restable()
    _save(store, f, legacy=True)


def _save(store: Store, f: _Fam, legacy: bool = False) -> None:
    with store.transaction():
        db = store.db
        db.execute("DELETE FROM family_lines WHERE family=?", (f.id,))
        db.executemany("INSERT INTO family_lines(family, line, n, chars, text, pos) VALUES(?,?,?,?,?,?)",
                       [(f.id, k, c, f.w[k], f.text[k], f.pos[k]) for k, c in f.cnt.items()])
        if legacy:
            db.execute("UPDATE families SET n_members=?, legacy_prefix=?, prefix=? WHERE id=?",
                       (f.n, f.legacy, f.display(), f.id))
        else:
            db.execute("UPDATE families SET n_members=?, prefix=?, updated=? WHERE id=?", (f.n, f.display(), now(), f.id))


def _add_member(f: _Fam, lines: list[tuple[str, str, int]]) -> None:
    f.n += 1
    base = len(f.pos)
    for i, (k, m, w) in enumerate(lines):
        f.cnt[k] = f.cnt.get(k, 0) + 1
        f.w[k], f.text[k] = w, m
        f.pos.setdefault(k, base + i)
    if f.n >= HALVE_AT:  # sliding window: keeps counts bounded and forgets one-off lines
        f.n //= 2
        for k in list(f.cnt):
            f.cnt[k] //= 2
            if f.cnt[k] == 0:
                del f.cnt[k], f.w[k], f.text[k], f.pos[k]
    f.restable()


# ------------------------------------------------------------------ resolve


def choose(fams: list[_Fam], lines: list[tuple[str, str, int]]) -> _Fam | None:
    """The qualifying family with the best coverage (ties: oldest, then smallest id)."""
    keys = {k: w for k, _, w in lines}
    total = sum(keys.values())
    best, best_key = None, None
    for f in fams:
        a, b = f.score(keys, total)
        if a >= COVER - 1e-9 and b >= OWN - 1e-9:
            key = (-(a + b), f.created, f.id)
            if best_key is None or key < best_key:
                best, best_key = f, key
    return best


def resolve(store: Store, system: str, tools: list[dict], dialect: str) -> str:
    tools_hash = h(canon(tools))
    sys_key = h(tools_hash, system)
    c = _cache(store)
    fid = c.memo.get(sys_key)
    if fid is not None:
        c.memo.move_to_end(sys_key)
        return fid
    with store.lock:
        fid = _resolve_new(store, system, tools_hash, sys_key, dialect)
    c.memo[sys_key] = fid
    if len(c.memo) > MEMO_MAX:
        c.memo.popitem(last=False)
    return fid


def _resolve_new(store: Store, system: str, tools_hash: str, sys_key: str, dialect: str) -> str:
    r = store.q1("SELECT family FROM family_members WHERE sys=?", (sys_key,))
    if r is not None:
        return r["family"]
    fams = _load(store, tools_hash)
    lines = lines_of(system)
    f = choose(fams, lines)
    if f is None:  # families from the prefix-based keying keep their old rule
        f = next((x for x in fams if x.legacy and len(x.legacy) >= LEGACY_MIN and system.startswith(x.legacy)), None)
    t = now()
    if f is None:
        fid = h(tools_hash, system, n=12)
        f = next((x for x in fams if x.id == fid), None)
        if f is None:
            store.x("INSERT OR IGNORE INTO families(id, tools_hash, prefix, dialect, created, updated, n_members) "
                    "VALUES(?,?,?,?,?,?,0)", (fid, tools_hash, "", dialect, t, t))
            f = _Fam(fid, t)
            fams.append(f)
    _add_member(f, lines)
    _save(store, f)
    store.x("INSERT OR IGNORE INTO family_members(sys, family, ts) VALUES(?,?,?)", (sys_key, f.id, t))
    _cache(store).fams[tools_hash] = (_stamp(store, tools_hash), fams)
    return f.id
