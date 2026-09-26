"""Small shared helpers: stable hashing, canonical JSON, time."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any


def canon(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace. Used for hashing and storage."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def h(*parts: Any, n: int = 16) -> str:
    """Short stable hex hash of the given parts."""
    m = hashlib.sha256()
    for p in parts:
        m.update(p if isinstance(p, bytes) else (p if isinstance(p, str) else canon(p)).encode())
        m.update(b"\x1f")
    return m.hexdigest()[:n]


def now() -> float:
    return time.time()


def rand_id(n: int = 12) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    raw = os.urandom(n)
    return "".join(alphabet[b % len(alphabet)] for b in raw)


_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def strip_reminders(text: str) -> str:
    """Remove harness-injected <system-reminder> blocks (Claude Code) from task text."""
    return _REMINDER_RE.sub("", text).strip()


def short(s: str, n: int = 60) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def decay(age_s: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(age_s, 0.0) / (half_life_days * 86400.0))
