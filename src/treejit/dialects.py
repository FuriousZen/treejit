"""Wire dialects: Anthropic Messages and OpenAI-compatible Chat Completions.

Each dialect parses a request body into a NormRequest, builds replay responses
(JSON or SSE) from tool calls, extracts calls/usage from upstream responses
(JSON or SSE, incrementally), and injects frontier hints.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from .model import REPLAY_MARK, Episode, NormRequest, Observation, ResponseInfo, Step, ToolCall, Usage
from .util import rand_id, strip_reminders


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict):
            t = block.get("type")
            if t in ("text", "input_text", "output_text") or (t is None and "text" in block):
                out.append(block.get("text", ""))
            elif t in ("image", "image_url", "input_image"):
                out.append("[image]")
            elif t == "tool_result":
                out.append(_text_of(block.get("content")))
            elif t == "document":
                out.append("[document]")
    return "\n".join(x for x in out if x)


def _sse(event: str | None, data: Any) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


class SSEParser:
    """Incremental text/event-stream parser yielding (event, data) pairs."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[tuple[str | None, str]]:
        self._buf += chunk.replace(b"\r\n", b"\n")
        events = []
        while b"\n\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n\n", 1)
            event, data = None, []
            for line in raw.decode("utf-8", "replace").split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])
            if data or event:
                events.append((event, "\n".join(data)))
        return events


class Dialect:
    name = ""
    call_prefix = ""

    def new_call_id(self, node: str, conf: float = 1.0, via: str = "") -> str:
        c = max(0, min(255, round(conf * 255)))
        return f"{self.call_prefix}_{REPLAY_MARK}_{node[:12]}_{c:02x}{rand_id(10)}" + (f"_{via}" if via else "")

    # implemented by subclasses
    def parse_request(self, body: dict) -> NormRequest: ...
    def build_response(self, model: str, calls: list[ToolCall]) -> dict: ...
    def build_sse(self, model: str, calls: list[ToolCall], body: dict) -> list[bytes]: ...
    def parse_response(self, body: dict) -> ResponseInfo: ...
    def stream_accumulator(self) -> "StreamAccumulator": ...
    def inject_hint(self, body: dict, text: str) -> dict: ...

    def prepare_forward(self, req: NormRequest) -> dict:
        return req.raw


class StreamAccumulator:
    def __init__(self) -> None:
        self.parser = SSEParser()
        self.info = ResponseInfo()

    def feed(self, chunk: bytes) -> None:
        for event, data in self.parser.feed(chunk):
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            self.on_event(event, obj)

    def on_event(self, event: str | None, obj: dict) -> None: ...

    def result(self) -> ResponseInfo:
        return self.info


# --------------------------------------------------------------------------- Anthropic


class Anthropic(Dialect):
    name = "anthropic"
    call_prefix = "toolu"

    def parse_request(self, body: dict) -> NormRequest:
        system = body.get("system") or ""
        system_text = system if isinstance(system, str) else _text_of(system)
        tools = [
            {"name": t.get("name"), "schema": t.get("input_schema"), "description": t.get("description", ""), "type": t.get("type")}
            for t in body.get("tools") or []
            if isinstance(t, dict) and t.get("name")
        ]
        msgs = body.get("messages") or []
        start = -1
        for i, m in enumerate(msgs):
            if m.get("role") == "user" and not _has_block(m, "tool_result"):
                start = i
        task = strip_reminders(_text_of(msgs[start].get("content"))) if start >= 0 else ""
        steps: list[Step] = []
        by_id: dict[str, Step] = {}
        for m in msgs[start + 1 :]:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            if m.get("role") == "assistant":
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        st = Step(ToolCall(b.get("id", ""), b.get("name", ""), b.get("input") or {}))
                        steps.append(st)
                        by_id[st.call.id] = st
            else:
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        st = by_id.get(b.get("tool_use_id", ""))
                        if st is not None:
                            st.obs = Observation(_text_of(b.get("content")), bool(b.get("is_error")))
        ready = bool(msgs) and msgs[-1].get("role") == "user" and all(s.obs is not None for s in steps)
        thinking = isinstance(body.get("thinking"), dict) and body["thinking"].get("type") in ("enabled", "adaptive")
        return NormRequest(
            "anthropic", body.get("model", ""), system_text, tools, bool(body.get("stream")),
            Episode(task, steps, ready), body, thinking,
        )

    def build_response(self, model: str, calls: list[ToolCall]) -> dict:
        return {
            "id": "msg_" + REPLAY_MARK + rand_id(20),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in calls],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    def build_sse(self, model: str, calls: list[ToolCall], body: dict) -> list[bytes]:
        msg = self.build_response(model, calls)
        start = dict(msg, content=[], stop_reason=None)
        out = [_sse("message_start", {"type": "message_start", "message": start})]
        for i, c in enumerate(calls):
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": i,
                                                    "content_block": {"type": "tool_use", "id": c.id, "name": c.name, "input": {}}}))
            out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(c.args)}}))
            out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": i}))
        out.append(_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                                          "usage": {"output_tokens": 0}}))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out

    def parse_response(self, body: dict) -> ResponseInfo:
        info = ResponseInfo(stop_reason=body.get("stop_reason"))
        for b in body.get("content") or []:
            if b.get("type") == "tool_use":
                info.calls.append(ToolCall(b.get("id", ""), b.get("name", ""), b.get("input") or {}))
            elif b.get("type") == "text":
                info.text += b.get("text", "")
        info.usage = _anthropic_usage(body.get("usage") or {})
        return info

    def stream_accumulator(self) -> StreamAccumulator:
        return _AnthropicStream()

    def inject_hint(self, body: dict, text: str) -> dict:
        body = copy.copy(body)
        msgs = list(body.get("messages") or [])
        if not msgs or msgs[-1].get("role") != "user":
            return body
        last = dict(msgs[-1])
        content = last.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        blocks.append({"type": "text", "text": text})
        last["content"] = blocks
        msgs[-1] = last
        body["messages"] = msgs
        return body

    def prepare_forward(self, req: NormRequest) -> dict:
        body = req.raw
        # With extended thinking on, the API expects assistant turns in a tool loop to
        # carry signed thinking blocks. Replayed turns have none, so fall back to a
        # non-thinking frontier call for this episode.
        if req.thinking and any(s.replayed_node for s in req.episode.steps):
            body = {k: v for k, v in body.items() if k != "thinking"}
        return body


def _has_block(msg: dict, kind: str) -> bool:
    c = msg.get("content")
    return isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == kind for b in c)


def _anthropic_usage(u: dict) -> Usage:
    return Usage(
        int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
        int(u.get("cache_read_input_tokens") or 0), int(u.get("cache_creation_input_tokens") or 0),
    )


class _AnthropicStream(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self._blocks: dict[int, dict] = {}

    def on_event(self, event: str | None, obj: dict) -> None:
        t = obj.get("type")
        if t == "message_start":
            self.info.usage = _anthropic_usage((obj.get("message") or {}).get("usage") or {})
        elif t == "content_block_start":
            self._blocks[obj.get("index", 0)] = dict(obj.get("content_block") or {}, _json="")
        elif t == "content_block_delta":
            b = self._blocks.get(obj.get("index", 0))
            d = obj.get("delta") or {}
            if b is not None:
                if d.get("type") == "input_json_delta":
                    b["_json"] += d.get("partial_json", "")
                elif d.get("type") == "text_delta":
                    self.info.text += d.get("text", "")
        elif t == "content_block_stop":
            b = self._blocks.get(obj.get("index", 0))
            if b is not None and b.get("type") == "tool_use":
                try:
                    args = json.loads(b["_json"]) if b["_json"] else (b.get("input") or {})
                except json.JSONDecodeError:
                    args = {}
                self.info.calls.append(ToolCall(b.get("id", ""), b.get("name", ""), args))
        elif t == "message_delta":
            self.info.stop_reason = (obj.get("delta") or {}).get("stop_reason") or self.info.stop_reason
            u = obj.get("usage") or {}
            if "output_tokens" in u:
                self.info.usage.output_tokens = int(u["output_tokens"] or 0)
            if u.get("input_tokens"):
                self.info.usage.input_tokens = int(u["input_tokens"])


# --------------------------------------------------------------------------- OpenAI


class OpenAI(Dialect):
    name = "openai"
    call_prefix = "call"

    def parse_request(self, body: dict) -> NormRequest:
        msgs = body.get("messages") or []
        system_parts = []
        for m in msgs:
            if m.get("role") in ("system", "developer"):
                system_parts.append(_text_of(m.get("content")))
            else:
                break
        tools = []
        for t in body.get("tools") or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if fn and fn.get("name"):
                tools.append({"name": fn["name"], "schema": fn.get("parameters"), "description": fn.get("description", "")})
        start = -1
        for i, m in enumerate(msgs):
            if m.get("role") == "user":
                start = i
        task = strip_reminders(_text_of(msgs[start].get("content"))) if start >= 0 else ""
        steps: list[Step] = []
        by_id: dict[str, Step] = {}
        for m in msgs[start + 1 :]:
            role = m.get("role")
            if role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    except json.JSONDecodeError:
                        args = {"_raw": raw}
                    if not isinstance(args, dict):
                        args = {"_value": args}
                    st = Step(ToolCall(tc.get("id", ""), fn.get("name", ""), args))
                    steps.append(st)
                    by_id[st.call.id] = st
            elif role == "tool":
                st = by_id.get(m.get("tool_call_id", ""))
                if st is not None:
                    st.obs = Observation(_text_of(m.get("content")), False)
        ready = bool(msgs) and msgs[-1].get("role") in ("tool", "user") and all(s.obs is not None for s in steps)
        return NormRequest("openai", body.get("model", ""), "\n".join(system_parts), tools, bool(body.get("stream")),
                           Episode(task, steps, ready), body)

    def build_response(self, model: str, calls: list[ToolCall]) -> dict:
        return {
            "id": "chatcmpl-" + REPLAY_MARK + rand_id(20),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}} for c in calls
                ]},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def build_sse(self, model: str, calls: list[ToolCall], body: dict) -> list[bytes]:
        cid, created = "chatcmpl-" + REPLAY_MARK + rand_id(20), int(time.time())

        def chunk(delta: dict, finish: str | None = None) -> bytes:
            return _sse(None, {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        out = [chunk({"role": "assistant", "content": None})]
        for i, c in enumerate(calls):
            out.append(chunk({"tool_calls": [{"index": i, "id": c.id, "type": "function", "function": {"name": c.name, "arguments": ""}}]}))
            out.append(chunk({"tool_calls": [{"index": i, "function": {"arguments": json.dumps(c.args)}}]}))
        out.append(chunk({}, "tool_calls"))
        if (body.get("stream_options") or {}).get("include_usage"):
            out.append(_sse(None, {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
                                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}))
        out.append(b"data: [DONE]\n\n")
        return out

    def parse_response(self, body: dict) -> ResponseInfo:
        info = ResponseInfo()
        choices = body.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        info.stop_reason = choices[0].get("finish_reason")
        info.text = msg.get("content") or ""
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            info.calls.append(ToolCall(tc.get("id", ""), fn.get("name", ""), args if isinstance(args, dict) else {}))
        info.usage = _openai_usage(body.get("usage") or {})
        return info

    def stream_accumulator(self) -> StreamAccumulator:
        return _OpenAIStream()

    def inject_hint(self, body: dict, text: str) -> dict:
        body = copy.copy(body)
        msgs = list(body.get("messages") or [])
        if not msgs:
            return body
        last = dict(msgs[-1])
        c = last.get("content")
        if isinstance(c, list):
            last["content"] = list(c) + [{"type": "text", "text": text}]
        else:
            last["content"] = (c or "") + "\n\n" + text
        msgs[-1] = last
        body["messages"] = msgs
        return body


def _openai_usage(u: dict) -> Usage:
    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return Usage(int(u.get("prompt_tokens") or 0) - cached, int(u.get("completion_tokens") or 0), cached, 0)


class _OpenAIStream(StreamAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self._calls: dict[int, dict] = {}

    def on_event(self, event: str | None, obj: dict) -> None:
        if obj.get("usage"):
            self.info.usage = _openai_usage(obj["usage"])
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                self.info.text += d["content"]
            for tc in d.get("tool_calls") or []:
                slot = self._calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] = fn.get("name") or slot["name"]
                slot["args"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                self.info.stop_reason = ch["finish_reason"]

    def result(self) -> ResponseInfo:
        self.info.calls = []
        for _, c in sorted(self._calls.items()):
            try:
                args = json.loads(c["args"]) if c["args"] else {}
            except json.JSONDecodeError:
                args = {}
            self.info.calls.append(ToolCall(c["id"], c["name"], args if isinstance(args, dict) else {}))
        return self.info


DIALECTS: dict[str, Dialect] = {"anthropic": Anthropic(), "openai": OpenAI()}


def get(name: str) -> Dialect:
    return DIALECTS[name]
