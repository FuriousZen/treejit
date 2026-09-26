"""Inline mode: `treejit.wrap(client)` for harnesses you own, and for tests.

Works with the Anthropic SDK (`client.messages.create`), the OpenAI SDK
(`client.chat.completions.create`), or any plain callable `fn(body) -> dict`
(pass dialect="anthropic"|"openai"). Streaming calls pass straight through.

Run ids: `wrap(client, run_id=...)`, or per call via
`extra_headers={"X-TreeJIT-Run": ...}`.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import dialects


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


class _Create:
    def __init__(self, jit: Any, create: Callable, dialect: str, run_id: str | None, sdk: bool) -> None:
        self.jit, self.create, self.dialect, self.run_id, self.sdk = jit, create, dialect, run_id, sdk

    def __call__(self, body: dict | None = None, **kwargs: Any) -> Any:
        if body is not None:
            kwargs = {**body, **kwargs}
        extra = dict(kwargs.pop("extra_headers", None) or {})
        headers = {k: v for k, v in extra.items() if v is not None}
        if self.run_id and not any(k.lower() == "x-treejit-run" for k in headers):
            headers["X-TreeJIT-Run"] = self.run_id
        if kwargs.get("stream"):
            return self._call(kwargs, extra)
        res = self.jit.handle(self.dialect, kwargs, headers)
        fwd_headers = {k: v for k, v in extra.items() if k.lower() != "x-treejit-run"}
        if res.kind == "subcall":
            res = self._subcall(res, fwd_headers)
        if res.kind == "replay":
            return _as_sdk(self.dialect, res.body) if self.sdk else res.body
        t0 = time.perf_counter()
        try:
            out = self._call(res.body, fwd_headers)
        except Exception:
            self.jit.complete(res, None, status=599, latency_ms=(time.perf_counter() - t0) * 1000)
            raise
        info = dialects.get(self.dialect).parse_response(_to_dict(out))
        self.jit.complete(res, info, 200, (time.perf_counter() - t0) * 1000)
        return out

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
            return self.create(**body, extra_headers=headers)
        return self.create(**body) if self.sdk else self.create(body)


class _Ns:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class Wrapped:
    """Proxy object exposing the wrapped create() at the usual SDK path."""

    def __init__(self, client: Any, create: _Create) -> None:
        self._client = client
        self._create = create
        if create.dialect == "anthropic" and create.sdk:
            self.messages = _Ns(create=create)
        elif create.dialect == "openai" and create.sdk:
            self.chat = _Ns(completions=_Ns(create=create))

    def __call__(self, body: dict | None = None, **kwargs: Any) -> Any:
        return self._create(body, **kwargs)

    def create(self, body: dict | None = None, **kwargs: Any) -> Any:
        return self._create(body, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_client(jit: Any, client: Any, run_id: str | None = None, dialect: str | None = None) -> Wrapped:
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
