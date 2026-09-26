"""Prompt-cache cost model for forwarded request bodies (PLAN C1).

`CacheModel` bills a sequence of Anthropic-shaped request bodies the way the Messages API prompt cache
does, to the first order:

  - the prompt is `tools`, then `system`, then the `thinking` setting (toggling it invalidates the
    messages cache on every model), then every content block of every message, in order. It is split
    into *positions* (one per block); a cache entry covers the exact bytes up to a position;
  - breakpoints: one explicit at the end of `system`, plus one at the end of the conversation. With
    `breakpoints="auto"` (top-level automatic caching) that is the last block of the request, a
    treejit hint included. With `breakpoints="harness"` it is the last block the *harness* sent
    (a hint appended by treejit lies after it, the way Claude Code places its marker);
  - a read is the longest live entry that is a byte prefix of this request. Reads only land where an
    earlier request wrote a breakpoint. Entries live `ttl` seconds after their last read or write;
    the 20-position lookback is not modelled (it doesn't bind in these scenarios);
  - billed = READ x read + WRITE x (last breakpoint - read) + 1.0 x (tail after the last breakpoint),
    in tokens = chars / 4 of canonical JSON.

`scenario()` drives the real engine (training runs, then a probe conversation) under a compaction mode
and returns the forwarded bodies with their timestamps; `bill()` totals them. Used by
tests/test_compaction.py (acceptance: first_sight never bills more than compaction off) and by
repro/C1_cache.py (the full table).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from conftest import Model, run_agent

from treejit import TreeJIT, compaction, dialects, families

READ, WRITE = 0.1, 1.25
HINT_TAG = "<treejit-hints>"


def _dump(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _strip_cc(x: Any) -> Any:
    """cache_control markers are not part of the cached bytes."""
    if isinstance(x, dict):
        return {k: _strip_cc(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_strip_cc(v) for v in x]
    return x


def positions(body: dict) -> tuple[list[str], int, int]:
    """(parts, system_end_index, last_harness_part_index): one part per cacheable position."""
    parts = [_dump(_strip_cc(body.get("tools"))), _dump(_strip_cc(body.get("system")))]
    sys_end = 1
    parts.append(_dump(body.get("thinking")))
    last_harness = len(parts) - 1
    for m in body.get("messages") or []:
        c = m.get("content")
        blocks = [c] if isinstance(c, str) or c is None else c
        for b in blocks:
            p = _dump([m.get("role"), _strip_cc(b)])
            parts.append(p)
            text = b.get("text", "") if isinstance(b, dict) else ""
            if not (isinstance(text, str) and text.startswith(HINT_TAG)):
                last_harness = len(parts) - 1
    return parts, sys_end, last_harness


@dataclass
class Bill:
    requests: int = 0
    tokens: float = 0.0      # raw input tokens sent
    read: float = 0.0        # served from cache
    billed: float = 0.0      # input-token equivalents
    rows: list = field(default_factory=list)

    def vs(self, other: "Bill") -> float:
        return (self.billed - other.billed) / other.billed if other.billed else 0.0


class CacheModel:
    def __init__(self, ttl: float = 300.0, breakpoints: str = "harness") -> None:
        assert breakpoints in ("auto", "harness")
        self.ttl, self.breakpoints = ttl, breakpoints
        self.entries: dict[tuple[int, str], float] = {}   # (prefix chars, sha of prefix) -> expiry

    def request(self, body: dict, t: float = 0.0) -> tuple[int, int, float]:
        parts, sys_end, last_harness = positions(body)
        s = "".join(parts)
        ends, acc = [], 0
        for p in parts:
            acc += len(p)
            ends.append(acc)
        read = 0
        for (length, digest), exp in list(self.entries.items()):
            if exp < t:
                del self.entries[(length, digest)]
            elif read < length <= len(s) and hashlib.sha256(s[:length].encode()).hexdigest() == digest:
                read = length
        if read:
            self.entries[(read, hashlib.sha256(s[:read].encode()).hexdigest())] = t + self.ttl
        bp = [ends[sys_end], ends[-1] if self.breakpoints == "auto" else ends[last_harness]]
        for p in bp:
            self.entries[(p, hashlib.sha256(s[:p].encode()).hexdigest())] = t + self.ttl
        last = max(bp)
        billed = READ * read + WRITE * max(0, last - read) + (len(s) - max(last, read))
        return len(s), read, billed


def bill(bodies: list[tuple[float, dict]], ttl: float = 300.0, breakpoints: str = "harness") -> Bill:
    cm, out = CacheModel(ttl, breakpoints), Bill()
    for t, b in bodies:
        total, read, billed = cm.request(b, t)
        out.requests += 1
        out.tokens += total / 4
        out.read += read / 4
        out.billed += billed / 4
        out.rows.append((round(total / 4), round(read / 4), round(billed / 4)))
    return out


# ---------------------------------------------------------------- scenarios on the real engine

N = 15
SYSTEM = "You are a careful agent working in a repository.\n" * 8
TOOLS = [
    {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
]
PATTERNS = ("dense", "interleaved", "bursty")   # "lead" (the model makes steps 0 and 8) is used for the thinking check


def _content(k: int, task: str) -> str:
    return "\n".join(f"{task}: file{k} line {j} " + "x" * 40 for j in range(50))  # ~3.3k chars


def _policy(pattern: str):
    def policy(task, hist, body):
        i = len(hist)
        if i >= N:
            return None
        if ((pattern == "interleaved" and i % 2 == 1) or (pattern == "bursty" and i % 5 == 4)
                or (pattern == "lead" and i in (0, 8))):
            return ("Bash", {"command": f"touch mark_{i}"})   # a write the model makes (not approved -> T4)
        return ("Bash", {"command": f"cat file_{i}.txt"})     # a big read-only observation that replays
    return policy


def _execute(task: str):
    def ex(name, args):
        cmd = args["command"]
        if cmd.startswith("cat"):
            return _content(int(cmd.split("_")[1].split(".")[0]), task), False
        return "", False
    return ex


class _SnapModel(Model):
    def __call__(self, body):
        return super().__call__(copy.deepcopy(body))


class Clock:
    """Fake time for the epoch mode: `step` seconds per forwarded request, plus `gaps` {request index: s}."""

    def __init__(self, step: float = 10.0, gaps: dict[int, float] | None = None) -> None:
        self.t, self.step, self.gaps, self.i = 1000.0, step, gaps or {}, 0

    def tick(self) -> float:
        self.t += self.step + self.gaps.get(self.i, 0.0)
        self.i += 1
        return self.t


@contextmanager
def _patched_clock(clock: Clock, times: list[float]) -> Iterator[None]:
    orig_apply = compaction.apply

    def apply(store, view, cfg, req, body, clock_=None):
        t = clock.tick()
        times.append(t)
        return orig_apply(store, view, cfg, req, body, clock=t)

    compaction.apply = apply
    try:
        yield
    finally:
        compaction.apply = orig_apply


def scenario(pattern: str, mode: str | None, *, system: str = SYSTEM, hints: str = "off", thinking: bool = False,
             gaps: dict[int, float] | None = None, db_dir: str | None = None) -> list[tuple[float, dict]]:
    """Forwarded (t, body) pairs of one probe conversation. mode None = compaction off.

    dense: a fully replayed 15-step conversation is forwarded after every step (an upper bound on how often
    the keep-last boundary moves under a warm cache). interleaved: every odd step is a model-made write.
    bursty: every fifth step is. The probe of interleaved/bursty runs through `jit.wrap` (real engine)."""
    key = (pattern, mode, system, hints, thinking, tuple(sorted((gaps or {}).items())))
    if db_dir is None and key in _MEMO:
        return list(_MEMO[key])
    if db_dir is not None:
        return _scenario(pattern, mode, system, hints, thinking, gaps, db_dir)
    with tempfile.TemporaryDirectory() as tmp:
        out = _MEMO[key] = _scenario(pattern, mode, system, hints, thinking, gaps, tmp)
    return list(out)


_MEMO: dict[tuple, list[tuple[float, dict]]] = {}


def _scenario(pattern: str, mode: str | None, system: str, hints: str, thinking: bool, gaps: dict[int, float] | None,
              tmp: str) -> list[tuple[float, dict]]:
    kw: dict[str, Any] = dict(compact=mode is not None, theta=0.0, hard_cap=100, max_depth=20, batch=False, t2=False, t3=False,
                              hints=hints)
    if mode is not None:
        kw["compact_mode"] = mode
    db = os.path.join(tmp, f"{pattern}_{mode}_{hints}_{thinking}_{len(system)}_{sorted((gaps or {}).items())}.db")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db + suffix):
            os.remove(db + suffix)
    jit = TreeJIT(db, **kw)
    extra = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if thinking else {}
    try:
        client = jit.wrap(Model(_policy(pattern)), dialect="anthropic")
        last = None
        for i in range(3):
            rid = f"train{i}"
            last = run_agent(lambda b: client(dict(b, **extra), extra_headers={"X-TreeJIT-Run": rid}), f"task {i}",
                             _execute(f"task {i}"), max_steps=N + 2, tools=TOOLS, system=system)
            jit.outcome(rid, "pass")
        clock, times = Clock(gaps=gaps), []
        if pattern == "dense":
            assert last is not None and all("_tj_" in b["id"] for m in last if m["role"] == "assistant"
                                            for b in m["content"] if b["type"] == "tool_use")
            # a fresh conversation with the same replayed steps (new call ids: the training run's own final
            # forward already made its decisions)
            text = json.dumps(last)
            for b in [b for m in last if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]:
                text = text.replace(b["id"], b["id"][:-10] + hashlib.sha256(b["id"].encode()).hexdigest()[:10])
            last = json.loads(text)
            d = dialects.get("anthropic")
            bodies = []
            for n in range(1, N + 1):
                body = dict({"model": "m", "max_tokens": 100, "system": system, "tools": TOOLS,
                             "messages": copy.deepcopy(last[: 1 + 2 * n])}, **extra)
                req = d.parse_request(body)
                view = jit.view(families.resolve(jit.store, req.system, req.tools, "anthropic"))
                fwd = d.prepare_forward(req)
                t = clock.tick()
                if mode is not None:
                    fwd = compaction.apply(jit.store, view, jit.cfg, req, fwd, clock=t).body
                bodies.append((t, fwd))
            return bodies
        model = _SnapModel(_policy(pattern))
        client = jit.wrap(model, dialect="anthropic")
        with _patched_clock(clock, times):
            run_agent(lambda b: client(dict(b, **extra), extra_headers={"X-TreeJIT-Run": "probe"}), "task 9",
                      _execute("task 9"), max_steps=N + 2, tools=TOOLS, system=system)
        if mode is None:  # compaction.apply isn't called: space requests out the same way
            times = [clock.tick() for _ in model.bodies]
        return list(zip(times, model.bodies))
    finally:
        jit.close()


def append_only(bodies: list[dict]) -> list[int]:
    """Indices i where forward i does not extend forward i-1 (all but its last message byte-identical)."""
    bad = []
    for i in range(1, len(bodies)):
        a, b = bodies[i - 1]["messages"], bodies[i]["messages"]
        if b[: len(a) - 1] != a[:-1]:
            bad.append(i)
    return bad
