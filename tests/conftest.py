"""Shared helpers: a scripted Anthropic-shaped model and a minimal agent loop."""

from __future__ import annotations

import itertools
from typing import Callable

import pytest

from treejit import TreeJIT

TOOLS = [
    {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
    {"name": "send_email", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}}}},
]
SYSTEM = "You are a careful agent working in a repository. " * 8

_ids = itertools.count()


def tool_msg(name: str, args: dict) -> dict:
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "tool_use", "id": f"toolu_m{next(_ids):08d}", "name": name, "input": args}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 10}}


def text_msg(text: str = "done") -> dict:
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "m", "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 100, "output_tokens": 5}}


def history(body: dict) -> list[tuple[str, dict, str, bool]]:
    calls, results = [], {}
    for m in body["messages"][1:]:
        if isinstance(m["content"], list):
            for b in m["content"]:
                if b["type"] == "tool_use":
                    calls.append(b)
                elif b["type"] == "tool_result":
                    results[b["tool_use_id"]] = (b["content"], b.get("is_error", False))
    return [(c["name"], c["input"], *results.get(c["id"], ("", False))) for c in calls]


class Model:
    """policy(task, history, body) -> (name, args) or None; counts calls."""

    def __init__(self, policy: Callable) -> None:
        self.policy = policy
        self.calls = 0
        self.bodies: list[dict] = []

    def __call__(self, body: dict) -> dict:
        self.calls += 1
        self.bodies.append(body)
        c = body["messages"][0]["content"]
        task = c if isinstance(c, str) else next(b["text"] for b in c if b["type"] == "text")
        act = self.policy(task, history(body), body)
        return text_msg() if act is None else tool_msg(*act)


def run_agent(call: Callable[[dict], dict], task: str, tools_exec: Callable[[str, dict], tuple[str, bool]],
              max_steps: int = 20, tools: list | None = None, system: str = SYSTEM) -> list[dict]:
    msgs = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        resp = call({"model": "m", "max_tokens": 100, "system": system, "tools": tools or TOOLS, "messages": msgs})
        msgs.append({"role": "assistant", "content": resp["content"]})
        uses = [b for b in resp["content"] if b["type"] == "tool_use"]
        if not uses:
            break
        results = []
        for u in uses:
            out, err = tools_exec(u["name"], u["input"])
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": out, "is_error": err})
        msgs.append({"role": "user", "content": results})
    return msgs


def calls_of(msgs: list[dict]) -> list[tuple[str, dict]]:
    return [(b["name"], b["input"]) for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]


def replayed_ids(msgs: list[dict]) -> list[str]:
    return [b["id"] for m in msgs if m["role"] == "assistant" for b in m["content"]
            if b["type"] == "tool_use" and "_tj_" in b["id"]]


@pytest.fixture
def jit(tmp_path):
    j = TreeJIT(str(tmp_path / "t.db"))
    yield j
    j.close()
