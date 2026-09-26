"""Inline mode: `treejit.wrap(client)` for harnesses you own, and for tests.

Works with the Anthropic SDK (`client.messages.create`), the OpenAI SDK
(`client.chat.completions.create`), or any plain callable `fn(body) -> dict`
(pass dialect="anthropic"|"openai").

Streaming (`stream=True`) goes through the engine too. A replay returns a
`ReplayStream` of events built from the replay SSE (SDK event models when the
SDK is installed, plain dicts otherwise). A forward returns a `TeeStream` that
yields the upstream events unchanged and records usage, the run and END once
the stream is exhausted. `client.messages.stream(...)` (Anthropic) is routed
through the same path. Every other attribute of the client and its namespaces
(`messages.count_tokens`, `messages.batches`, `beta`, ...) is the real client's,
untouched and unrecorded. Async clients are not supported inline: use the proxy.

Run ids: `wrap(client, run_id=...)`, or per call via
`extra_headers={"X-TreeJIT-Run": ...}`. The header never goes upstream.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable, Iterable, Iterator

from . import dialects

RUN_HEADER = "x-treejit-run"


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(mode="json") if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    raise TypeError(f"cannot convert {type(obj).__name__} to dict")


def _as_sdk(proto: Any, data: dict) -> Any:
    """Build an SDK response object of the same family as the client's return type, if the SDK is installed."""
    try:
        if proto == "anthropic":
            from anthropic.types import Message  # type: ignore

            return Message.model_validate(data)
        if proto == "openai":
            from openai.types.chat import ChatCompletion  # type: ignore

            return ChatCompletion.model_validate(data)
    except Exception:  # SDK missing or schema drift: fall back to the plain dict
        pass
    return data


_EVENT_ADAPTERS: dict[str, Any] = {}


def _event_obj(dialect: str, data: dict) -> Any:
    """One stream event as the SDK's model (RawMessageStreamEvent / ChatCompletionChunk), else the dict."""
    try:
        if dialect == "anthropic":
            ta = _EVENT_ADAPTERS.get(dialect)
            if ta is None:
                from anthropic.types import RawMessageStreamEvent  # type: ignore
                from pydantic import TypeAdapter

                ta = _EVENT_ADAPTERS[dialect] = TypeAdapter(RawMessageStreamEvent)
            return ta.validate_python(data)
        if dialect == "openai":
            from openai.types.chat import ChatCompletionChunk  # type: ignore

            return ChatCompletionChunk.model_validate(data)
    except Exception:  # SDK missing or schema drift: plain dict
        pass
    return data


def _sse_events(sse: Iterable[bytes]) -> list[dict]:
    parser, out = dialects.SSEParser(), []
    for chunk in sse:
        for _, data in parser.feed(chunk):
            if not data or data == "[DONE]":
                continue
            try:
                out.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    return out


class ReplayStream:
    """A replayed streaming response: the replay SSE as events, served locally (no upstream call).
    Shaped like the SDKs' `Stream`: iterable once, a context manager, `close()`, `.response` (None)."""

    response = None

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self._it = iter(events)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        return next(self._it)

    def close(self) -> None:
        self._it = iter(())

    def __enter__(self) -> "ReplayStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class TeeStream:
    """A forwarded streaming response. Yields the upstream events unchanged and feeds each to the
    dialect's stream accumulator. Exhaustion completes the request (usage, run id, END); closing it
    early completes it with status 499 (never END). Other attributes are the upstream stream's."""

    def __init__(self, upstream: Any, jit: Any, res: Any, acc: Any, t0: float) -> None:
        self._upstream, self._jit, self._res, self._acc, self._t0 = upstream, jit, res, acc, t0
        self._it: Iterator[Any] | None = None
        self._done = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._done:
            raise StopIteration
        try:
            if self._it is None:
                self._it = iter(self._upstream)
            ev = next(self._it)
        except StopIteration:
            self._finish(200)
            raise
        except Exception as e:
            self._finish(int(getattr(e, "status_code", 0) or 599))
            raise
        except BaseException:  # KeyboardInterrupt, GeneratorExit: the client went away
            self._finish(499)
            raise
        self._feed(ev)
        return ev

    def _feed(self, ev: Any) -> None:
        try:  # accounting never breaks the caller's stream
            if isinstance(ev, (bytes, bytearray)):
                self._acc.feed(bytes(ev))
            elif isinstance(ev, str):
                self._acc.feed(ev.encode())
            else:
                self._acc.on_event(None, _to_dict(ev))
        except Exception:
            pass

    def _finish(self, status: int) -> None:
        if self._done:
            return
        self._done = True
        try:
            info = self._acc.result()
        except Exception:
            info = None
        self._jit.complete(self._res, info, status, (time.perf_counter() - self._t0) * 1000)

    @property
    def run_id(self) -> str | None:
        return self._res.run_id

    def close(self) -> None:
        # Closed before the end. If the stop reason already arrived the response is complete (only
        # trailing events such as message_stop or a usage chunk are skipped): record it as such.
        # Otherwise it was cut short: 499, which never derives a run or marks END.
        try:
            stopped = bool(self._acc.info.stop_reason)
        except Exception:
            stopped = False
        self._finish(200 if stopped else 499)
        fn = getattr(self._upstream, "close", None)
        if callable(fn):
            fn()

    def __enter__(self) -> "TeeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if not self._done:
                self._finish(499)
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._upstream, name)


class _SdkEvents:
    """Feeds `anthropic.lib.streaming.MessageStream`, which needs event models: dict events are
    converted (plain-callable or fake upstreams); SDK events pass as they are."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[Any]:
        for ev in self._stream:
            yield _event_obj("anthropic", ev) if isinstance(ev, dict) else ev

    @property
    def response(self) -> Any:
        return getattr(self._stream, "response", None)

    def close(self) -> None:
        self._stream.close()


class _MessageStream:
    """Fallback for `messages.stream()` without the anthropic SDK installed: the raw events, plus
    `text_stream`, `until_done()`, `get_final_message()` (a dict) and `get_final_text()`."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._snap: dict | None = None
        self._json: dict[int, str] = {}
        self._iterator = self.__stream__()
        self.text_stream = self.__text__()

    @property
    def response(self) -> Any:
        return getattr(self._stream, "response", None)

    def __iter__(self) -> Iterator[Any]:
        yield from self._iterator

    def __next__(self) -> Any:
        return next(self._iterator)

    def __stream__(self) -> Iterator[Any]:
        for ev in self._stream:
            self._accumulate(_to_dict(ev))
            yield ev

    def __text__(self) -> Iterator[str]:
        for ev in self:
            e = _to_dict(ev)
            if e.get("type") == "content_block_delta" and (e.get("delta") or {}).get("type") == "text_delta":
                yield e["delta"].get("text", "")

    def _accumulate(self, e: dict) -> None:
        t = e.get("type")
        if t == "message_start":
            self._snap = dict(e.get("message") or {}, content=[])
            return
        snap = self._snap
        if snap is None:
            return
        if t == "content_block_start":
            snap["content"].append(dict(e.get("content_block") or {}))
        elif t == "content_block_delta":
            i, d = e.get("index", 0), e.get("delta") or {}
            if i >= len(snap["content"]):
                return
            b = snap["content"][i]
            if d.get("type") == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif d.get("type") == "input_json_delta":
                self._json[i] = self._json.get(i, "") + d.get("partial_json", "")
                try:
                    b["input"] = json.loads(self._json[i])
                except json.JSONDecodeError:
                    pass
        elif t == "message_delta":
            d = e.get("delta") or {}
            snap["stop_reason"] = d.get("stop_reason", snap.get("stop_reason"))
            snap["stop_sequence"] = d.get("stop_sequence", snap.get("stop_sequence"))
            usage = dict(snap.get("usage") or {})
            usage.update({k: v for k, v in (e.get("usage") or {}).items() if v is not None})
            snap["usage"] = usage

    def until_done(self) -> None:
        for _ in self:
            pass

    @property
    def current_message_snapshot(self) -> dict | None:
        return self._snap

    def get_final_message(self) -> dict:
        self.until_done()
        if self._snap is None:
            raise RuntimeError("the stream ended without a message_start event")
        return self._snap

    def get_final_text(self) -> str:
        return "".join(b.get("text", "") for b in self.get_final_message()["content"] if b.get("type") == "text")

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "_MessageStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _StreamManager:
    def __init__(self, api_request: Callable[[], Any]) -> None:
        self._request, self._stream = api_request, None

    def __enter__(self) -> _MessageStream:
        self._stream = _MessageStream(self._request())
        return self._stream

    def __exit__(self, *exc: Any) -> None:
        if self._stream is not None:
            self._stream.close()


class _Create:
    def __init__(self, jit: Any, create: Callable, dialect: str, run_id: str | None, sdk: bool) -> None:
        self.jit, self.create, self.dialect, self.run_id, self.sdk = jit, create, dialect, run_id, sdk

    def __call__(self, body: dict | None = None, **kwargs: Any) -> Any:
        if body is not None:
            kwargs = {**body, **kwargs}
        extra = dict(kwargs.pop("extra_headers", None) or {})
        headers = {k: v for k, v in extra.items() if v is not None}
        if self.run_id and not any(k.lower() == RUN_HEADER for k in headers):
            headers["X-TreeJIT-Run"] = self.run_id
        fwd_headers = {k: v for k, v in extra.items() if k.lower() != RUN_HEADER}
        stream = bool(kwargs.get("stream"))
        res = self.jit.handle(self.dialect, kwargs, headers)
        if res.kind == "subcall":
            res = self._subcall(res, fwd_headers)
        if res.kind == "replay":
            if res.stream:
                events = _sse_events(res.sse or [])
                return ReplayStream([_event_obj(self.dialect, e) for e in events] if self.sdk else events)
            return _as_sdk(self.dialect, res.body) if self.sdk else res.body
        t0 = time.perf_counter()
        try:
            out = self._call(res.body, fwd_headers)
        except Exception as e:
            self.jit.complete(res, None, status=int(getattr(e, "status_code", 0) or 599),
                              latency_ms=(time.perf_counter() - t0) * 1000)
            raise
        if stream:
            return TeeStream(out, self.jit, res, dialects.get(self.dialect).stream_accumulator(), t0)
        info = dialects.get(self.dialect).parse_response(_to_dict(out))
        self.jit.complete(res, info, 200, (time.perf_counter() - t0) * 1000)
        return out

    def stream(self, **kwargs: Any) -> Any:
        """`messages.stream(...)`: a context manager over create(stream=True). With the anthropic SDK
        installed it is the SDK's own MessageStreamManager (text_stream, get_final_message, derived
        text/input_json events), fed by ReplayStream / TeeStream."""
        kwargs.pop("stream", None)
        real = getattr(getattr(self.create, "__self__", None), "stream", None)
        if kwargs.get("output_format") is not None and callable(real):
            # structured-output parsing lives in the SDK's stream(): pass through, unrecorded
            extra = {k: v for k, v in (kwargs.pop("extra_headers", None) or {}).items() if k.lower() != RUN_HEADER}
            return real(**kwargs, extra_headers=extra) if extra else real(**kwargs)

        def request() -> Any:
            return self(**kwargs, stream=True)

        try:
            from anthropic import NOT_GIVEN  # type: ignore
            from anthropic.lib.streaming import MessageStreamManager  # type: ignore
        except ImportError:
            return _StreamManager(request)
        try:
            return MessageStreamManager(lambda: _SdkEvents(request()), output_format=NOT_GIVEN)
        except TypeError:  # older SDKs: no output_format
            return MessageStreamManager(lambda: _SdkEvents(request()))

    def _subcall(self, res: Any, headers: dict) -> Any:
        """T2/T3: one small non-streaming call; any failure falls back to the T4 forward."""
        t0 = time.perf_counter()
        try:
            out, status = _to_dict(self._call(res.body, headers)), 200
        except Exception as e:
            out, status = None, int(getattr(e, "status_code", 0) or 599)
        return self.jit.resume(res, out, status, (time.perf_counter() - t0) * 1000)

    def _call(self, body: dict, headers: dict) -> Any:
        if self.sdk and headers:
            out = self.create(**body, extra_headers=headers)
        else:
            out = self.create(**body) if self.sdk else self.create(body)
        if inspect.isawaitable(out):
            close = getattr(out, "close", None)
            if callable(close):
                close()
            raise TypeError("inline mode wraps synchronous clients only; use the proxy for async clients")
        return out


class _Ns:
    """A namespace with some attributes overridden; everything else is the real namespace's."""

    def __init__(self, real: Any = None, **kw: Any) -> None:
        self.__dict__["_real"] = real
        self.__dict__.update(kw)

    def __getattr__(self, name: str) -> Any:
        real = self.__dict__.get("_real")
        if real is None:
            raise AttributeError(name)
        return getattr(real, name)


class Wrapped:
    """Proxy object exposing the wrapped create() at the usual SDK path."""

    def __init__(self, client: Any, create: _Create) -> None:
        self._client = client
        self._create = create
        if create.dialect == "anthropic" and create.sdk:
            self.messages = _Ns(client.messages, create=create, stream=create.stream)
        elif create.dialect == "openai" and create.sdk:
            chat = client.chat
            self.chat = _Ns(chat, completions=_Ns(chat.completions, create=create))

    def __call__(self, body: dict | None = None, **kwargs: Any) -> Any:
        return self._create(body, **kwargs)

    def create(self, body: dict | None = None, **kwargs: Any) -> Any:
        return self._create(body, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_client(jit: Any, client: Any, run_id: str | None = None, dialect: str | None = None) -> Wrapped:
    if type(client).__name__.startswith("Async"):
        raise TypeError(f"inline mode wraps synchronous clients only ({type(client).__name__}); use the proxy")
    messages = getattr(client, "messages", None)
    if dialect in (None, "anthropic") and messages is not None and callable(getattr(messages, "create", None)):
        return Wrapped(client, _Create(jit, messages.create, "anthropic", run_id, True))
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if dialect in (None, "openai") and completions is not None and callable(getattr(completions, "create", None)):
        return Wrapped(client, _Create(jit, completions.create, "openai", run_id, True))
    if callable(client):
        if dialect not in dialects.DIALECTS:
            raise ValueError("wrapping a plain callable needs dialect='anthropic' or 'openai'")
        return Wrapped(client, _Create(jit, client, dialect, run_id, False))
    raise TypeError("wrap() expects an Anthropic/OpenAI client or a callable body -> response dict")


def wrap(client: Any, db: str = "treejit.db", run_id: str | None = None, dialect: str | None = None, **config: Any) -> Wrapped:
    """Convenience: wrap a client with a TreeJIT backed by `db`. The engine is at `.jit`."""
    from .engine import TreeJIT

    jit = TreeJIT(db, **config)
    w = wrap_client(jit, client, run_id=run_id, dialect=dialect)
    w.jit = jit  # type: ignore[attr-defined]
    return w
