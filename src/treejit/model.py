"""Dialect-neutral view of an agent conversation.

An *episode* is everything after the last user message that isn't a tool result:
the task text plus the sequence of (tool call, observation) steps taken so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REPLAY_MARK = "tj"  # replayed tool-call ids: toolu_tj_<node12>_<conf:02x><rand> (call_tj_... for OpenAI)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Observation:
    text: str
    is_error: bool = False


@dataclass
class Step:
    call: ToolCall
    obs: Observation | None = None

    @property
    def replayed_node(self) -> str | None:
        """Node id encoded in a replayed call id, or None if the model chose this step."""
        parts = self.call.id.split("_")
        if len(parts) >= 4 and parts[1] == REPLAY_MARK:
            return parts[2]
        return None

    @property
    def replayed_conf(self) -> float | None:
        """Confidence the replayed step was chosen with (encoded in its id)."""
        parts = self.call.id.split("_")
        if len(parts) >= 4 and parts[1] == REPLAY_MARK:
            try:
                return int(parts[3][:2], 16) / 255
            except ValueError:
                return None
        return None


@dataclass
class Episode:
    task: str
    steps: list[Step] = field(default_factory=list)
    ready: bool = True  # last message is from the user side and every call has a result


@dataclass
class NormRequest:
    dialect: str
    model: str
    system: str
    tools: list[dict]      # [{"name":..., "schema":...}] normalized
    stream: bool
    episode: Episode
    raw: dict
    thinking: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read + self.cache_write


@dataclass
class ResponseInfo:
    calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
