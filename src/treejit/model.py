"""Dialect-neutral view of an agent conversation.

An *episode* is one task: everything since the last user turn that started a new task
(a task boundary, see dialects.episode_of). It is the task text plus the sequence of
(tool call, observation) steps taken so far. Later user turns inside the episode (a "yes"
to a confirmation question, an answer, a steering message after an interrupt) are
*user steps*: pseudo-calls named `$user:<kind>` whose observation is the user's text.
Bindings and features see them like any observation; replay never produces one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

REPLAY_MARK = "tj"  # replayed tool-call ids: toolu_tj_<node12>_<conf:02x><rand>[_<via>] (call_tj_... for OpenAI)
# <via>: how a subcall-assisted step was produced. t3: holes filled by the model; t2: the model
# chose among known children; ck: the model confirmed a step at a budget checkpoint.


USER = "$user"  # user-step pseudo-tool prefix; never a real tool name ('$' is not allowed in tool names)
# kinds: yes / no = a short confirmation / refusal of something the agent asked; text = any other reply
# after the agent ended its turn; steer = the user cut in while the agent was working (an interrupt,
# a message queued between tool calls). Only the first three mean "the model stopped here".
USER_KINDS = ("yes", "no", "text", "steer")


def is_user(name: str) -> bool:
    return name.startswith(USER)


def user_ended(name: str) -> bool:
    """A user step that answers a finished agent turn (the model chose to stop and talk)."""
    return is_user(name) and name != f"{USER}:steer"


_WEAK_ID = re.compile(r"[A-Za-z_.:\-]*\d{0,6}")


def weak_call_id(cid: str) -> bool:
    """Call ids that don't identify a conversation: empty, short, or a counter (`call_0`,
    `toolu_01`, `functions.Bash:0`) as some local servers and proxies produce."""
    return len(cid) < 12 or bool(_WEAK_ID.fullmatch(cid))


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

    @property
    def is_user(self) -> bool:
        return is_user(self.call.name)

    @property
    def replayed_via(self) -> str:
        """'t2' | 'ck' | 't3' for steps produced with a subcall, '' otherwise."""
        parts = self.call.id.split("_")
        if len(parts) >= 5 and parts[1] == REPLAY_MARK:
            return parts[4]
        return ""


@dataclass
class Episode:
    task: str
    steps: list[Step] = field(default_factory=list)
    ready: bool = True  # last message is from the user side and every call has a result
    # identity (see engine.TreeJIT._run_id): which conversation this is and which task in it
    index: int = 0                    # task boundaries before this episode in the conversation
    origin: str = ""                  # first user text of the whole conversation
    anchor_ids: list[str] | None = None  # tool-call ids of the conversation's first assistant turn (None: none yet)
    anchor_text: str = ""             # ...and its text
    anchor_salt: str | None = None    # observation that followed it (for weak ids), None if not seen yet
    session: str = ""                 # harness session id (Claude Code metadata.user_id, OpenAI prompt_cache_key)
    user: str = ""                    # weaker per-user id (OpenAI `user`): mixed into the key, never enough alone


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
    passthrough: str = ""  # why the request is forwarded untouched and unrecorded (e.g. "stateful")


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
