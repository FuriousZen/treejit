"""Tree keying: family = tool schemas + the *stable* prefix of the system prompt.

Harnesses like Claude Code embed cwd, date and git status in the system prompt,
so hashing it whole would give every session its own tree. Instead each family
remembers the longest common prefix of the system prompts it has seen (trimmed
to a line boundary); a new prompt joins a family if it starts with that prefix,
or if sharing a long enough prefix shrinks it. The family id never changes.
"""

from __future__ import annotations

import os

from .store import Store
from .util import canon, h, now

MIN_PREFIX = 64        # shorter common prefixes don't merge families
MIN_SHARE = 0.5        # the shrunk prefix must keep this share of the old one


def _trim(lcp: str) -> str:
    cut = lcp.rfind("\n")
    if cut < 0:
        cut = lcp.rfind(" ")
    return lcp[: cut + 1] if cut >= 0 else lcp


def resolve(store: Store, system: str, tools: list[dict], dialect: str) -> str:
    tools_hash = h(canon(tools))
    rows = store.q("SELECT id, prefix FROM families WHERE tools_hash=? ORDER BY created", (tools_hash,))
    for r in rows:
        if system.startswith(r["prefix"]):
            return r["id"]
    best, best_lcp = None, ""
    for r in rows:
        lcp = _trim(os.path.commonprefix([r["prefix"], system]))
        if len(lcp) >= MIN_PREFIX and len(lcp) >= MIN_SHARE * len(r["prefix"]) and len(lcp) > len(best_lcp):
            best, best_lcp = r, lcp
    t = now()
    if best is not None:
        store.x("UPDATE families SET prefix=?, updated=? WHERE id=?", (best_lcp, t, best["id"]))
        return best["id"]
    fid = h(tools_hash, system, n=12)
    store.x("INSERT OR IGNORE INTO families(id, tools_hash, prefix, dialect, created, updated) VALUES(?,?,?,?,?,?)",
            (fid, tools_hash, system, dialect, t, t))
    return fid
