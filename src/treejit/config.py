"""Configuration. Every knob has a default; a `treejit.toml` ([treejit] table) or
TREEJIT_* environment variables override them."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any

# Tools that are safe to replay without operator approval: read-only or idempotent.
DEFAULT_REPLAY_TOOLS = [
    "Read", "Glob", "Grep", "LS", "NotebookRead", "WebFetch", "WebSearch",
    "read_file", "list_files", "list_dir", "search_files", "grep", "glob",
    "get_*", "find_*", "list_*", "search_*", "read_*", "lookup_*", "calculate*", "think",
]

# Tools whose `command`/`cmd` string argument is a shell command.
DEFAULT_SHELL_TOOLS = ["Bash", "bash", "shell", "run_shell_command", "execute_command", "terminal", "run_command"]

# Irreversible or externally visible actions. Never crossed by replay unless the
# operator approved the edge AND the edge is past its promotion threshold.
DEFAULT_COMMIT_TOOLS = [
    "send_*", "cancel_*", "return_*", "exchange_*", "modify_*", "transfer_*", "book_*",
    "charge*", "pay*", "refund*", "delete_*", "update_*", "create_*", "post_*", "publish*",
]
DEFAULT_COMMIT_COMMANDS = [
    "git push", "npm publish", "pnpm publish", "yarn publish", "cargo publish", "twine upload",
    "gh pr", "gh release", "docker push", "kubectl apply", "kubectl delete", "terraform apply",
    "curl", "wget", "scp", "rsync", "ssh",
]


@dataclass
class Config:
    db: str = "treejit.db"
    # promotion / recognition
    promote_runs: int = 2          # N: distinct passing runs before an edge goes live
    max_depth: int = 12            # D: root-anchored path depth; deeper steps use n-gram anchors only
    ngram: tuple = (3, 2, 1)       # last-k-edge macro contexts, most specific first
    purity: float = 0.8            # min share of evidence a child needs to be chosen
    # replay bounds
    theta: float = 0.5             # confidence budget: product of edge confidences must stay above
    hard_cap: int = 8              # K: max consecutive replayed steps
    batch: bool = True             # collapse independent proven edges into one assistant message
    max_batch: int = 4
    # escalation tiers between replay and a full model call
    t2: bool = True                # choose among known children / checkpoint when the budget runs out
    t3: bool = True                # fill holes of a known edge with one small constrained call
    small_model: str = ""          # model for T2/T3 subcalls (default: the request's model)
    subcall_max_tokens: int = 512
    # failure handling
    tomb_k: float = 2.0            # decayed failures across distinct inputs before tombstoning
    tomb_prob: float = 0.5
    half_life_days: float = 14.0
    max_runs: int = 2000           # most recent runs per family a rebuild reads (bounds build time)
    # eviction
    evict_days: float = 30.0
    evict_min_hits: int = 3
    # frontier hints
    hints: str = "failures"        # off | failures | always
    hint_max: int = 5
    # frontier prefix compaction (opt-in): digest verified replayed observations in forwarded requests
    compact: bool = False
    compact_keep_last: int = 3     # the last N observations always go upstream in full
    compact_min_chars: int = 400   # smaller observations are left alone
    # tool policy
    replay_tools: list = field(default_factory=lambda: list(DEFAULT_REPLAY_TOOLS))
    shell_tools: list = field(default_factory=lambda: list(DEFAULT_SHELL_TOOLS))
    commit_tools: list = field(default_factory=lambda: list(DEFAULT_COMMIT_TOOLS))
    commit_commands: list = field(default_factory=lambda: list(DEFAULT_COMMIT_COMMANDS))
    extra_readonly_commands: list = field(default_factory=list)
    # proxy
    host: str = "127.0.0.1"
    port: int = 8787
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"

    @classmethod
    def load(cls, path: str | None = None, **overrides: Any) -> "Config":
        values: dict[str, Any] = {}
        path = path or os.environ.get("TREEJIT_CONFIG") or ("treejit.toml" if os.path.exists("treejit.toml") else None)
        if path:
            try:
                import tomllib  # py3.11+
            except ImportError:  # pragma: no cover
                tomllib = None
            if tomllib is not None:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                values.update(data.get("treejit", data))
        names = {f.name: f for f in dataclasses.fields(cls)}
        for name, f in names.items():
            env = os.environ.get("TREEJIT_" + name.upper())
            if env is not None:
                values[name] = _coerce(env, f.default if f.default is not dataclasses.MISSING else [])
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - set(names)
        if unknown:
            raise ValueError(f"unknown treejit config keys: {sorted(unknown)}")
        if "ngram" in values:
            values["ngram"] = tuple(values["ngram"])
        return cls(**values)


def _coerce(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (list, tuple)):
        items = [x.strip() for x in raw.split(",") if x.strip()]
        return tuple(int(x) for x in items) if isinstance(default, tuple) else items
    return raw
