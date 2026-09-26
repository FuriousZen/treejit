"""Proxy mode: a raw ASGI app in front of the model API.

    treejit serve --port 8787
    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
    OPENAI_BASE_URL=http://127.0.0.1:8787/v1 <harness>

Routes:
    POST /v1/messages                 Anthropic Messages (JSON or SSE)
    POST /v1/chat/completions         OpenAI-compatible chat completions (JSON or SSE)
    POST /v1/responses                OpenAI Responses API (JSON or SSE; stateless requests are learned and
                                      replayed, `previous_response_id` / `conversation` pass through)
    POST /outcome                     {"run_id": "...|latest", "outcome": "pass|fail|error", "reason": "...", "wait": true}
                                      rebuilds off the event loop (the proxy always rebuilds in the
                                      background, `rebuild = "sync"` included); answers once the tree is
                                      rebuilt ("wait": false: once the outcome is recorded)
    GET  /health, GET /stats
    anything else                     passed through untouched

Needs `httpx` (HTTP client) and an ASGI server such as `uvicorn`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from . import dialects
from .engine import Result, TreeJIT

HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate", "accept-encoding"}
RESP_DROP = {"content-length", "connection", "transfer-encoding", "content-encoding", "keep-alive"}

ROUTES = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai", "/chat/completions": "openai",
          "/v1/responses": "responses", "/responses": "responses"}


class ProxyApp:
    def __init__(self, jit: TreeJIT, client: Any = None) -> None:
        self.jit = jit
        self.cfg = jit.cfg
        self._client = client
        # Rebuilds always run on the engine's worker thread, whatever `rebuild` says: requests are handled
        # on the event loop, and a sync build there (or waiting there for the build lock while /outcome
        # builds on a thread) would stall every other request. Requests keep the previous tree until the
        # new one is swapped in; /outcome (wait=true, the default) still answers once it is.
        jit.set_rebuild_mode("background")

    @property
    def client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))
        return self._client

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    if self._client is not None:
                        await self._client.aclose()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        path, method = scope["path"], scope["method"]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        started = False
        raw_send = send

        async def send(msg: dict) -> None:  # noqa: F811 - track whether a response has begun
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await raw_send(msg)

        try:
            if method == "POST" and path in ROUTES:
                await self._model(ROUTES[path], scope, headers, body, send)
            elif method == "POST" and path == "/outcome":
                await self._outcome(body, send)
            elif method == "GET" and path == "/health":
                await _json(send, 200, {"ok": True})
            elif method == "GET" and path == "/stats":
                await _json(send, 200, self.jit.stats())
            else:
                await self._passthrough(scope, headers, body, send)
        except Exception as e:  # never take the harness down with us
            if started:
                raise
            await _json(send, 502, {"type": "error", "error": {"type": "treejit_error", "message": f"{type(e).__name__}: {e}"}})

    # ------------------------------------------------------------------ routes
    def _upstream(self, dialect: str, headers: dict) -> str:
        if dialect == "anthropic":
            return self.cfg.anthropic_upstream.rstrip("/")
        if dialect in ("openai", "responses"):
            return self.cfg.openai_upstream.rstrip("/")
        return (self.cfg.anthropic_upstream if ("x-api-key" in headers or "anthropic-version" in headers)
                else self.cfg.openai_upstream).rstrip("/")

    async def _model(self, dialect: str, scope: dict, headers: dict, body: bytes, send: Any) -> None:
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            await self._passthrough(scope, headers, body, send)
            return
        res = self.jit.handle(dialect, req, headers)
        if res.kind == "subcall":
            res = await self._subcall(dialect, scope, headers, res)
        if res.kind == "replay":
            await self._send_replay(res, send)
            return
        await self._forward(dialect, scope, headers, res, send)

    async def _subcall(self, dialect: str, scope: dict, headers: dict, res: Result) -> Result:
        """T2/T3: one small non-streaming call upstream with the client's auth; any failure -> T4."""
        url = self._upstream(dialect, headers) + _path_qs(scope)
        fwd_headers = {k: v for k, v in headers.items() if k not in HOP and not k.startswith("x-treejit")}
        fwd_headers["accept-encoding"] = "identity"
        t0 = time.perf_counter()
        try:
            r = await self.client.post(url, headers=fwd_headers, content=json.dumps(res.body).encode())
            status = r.status_code
            out = r.json() if status < 400 else None
        except Exception:  # network error, bad JSON: the subcall is best-effort
            status, out = 599, None
        return self.jit.resume(res, out if isinstance(out, dict) else None, status, (time.perf_counter() - t0) * 1000)

    async def _send_replay(self, res: Result, send: Any) -> None:
        extra = [(k.encode(), v.encode()) for k, v in res.headers.items()]
        if res.stream:
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"), *extra]})
            for chunk in res.sse or []:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            await _json(send, 200, res.body, extra)

    async def _forward(self, dialect: str, scope: dict, headers: dict, res: Result, send: Any) -> None:
        url = self._upstream(dialect, headers) + _path_qs(scope)
        fwd_headers = {k: v for k, v in headers.items() if k not in HOP and not k.startswith("x-treejit")}
        fwd_headers["accept-encoding"] = "identity"
        payload = json.dumps(res.body).encode()
        t0 = time.perf_counter()
        request = self.client.build_request("POST", url, headers=fwd_headers, content=payload)
        upstream = await self.client.send(request, stream=True)
        status = upstream.status_code
        out_headers = [(k.encode(), v.encode()) for k, v in upstream.headers.items() if k.lower() not in RESP_DROP]
        out_headers += [(k.encode(), v.encode()) for k, v in res.headers.items()]
        await send({"type": "http.response.start", "status": status, "headers": out_headers})
        d = dialects.get(dialect)
        is_sse = "text/event-stream" in upstream.headers.get("content-type", "")
        acc = d.stream_accumulator() if is_sse else None
        buf = b""
        try:
            async for chunk in upstream.aiter_raw():
                if acc is not None:
                    acc.feed(chunk)
                else:
                    buf += chunk
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            await upstream.aclose()
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        info = None
        if status < 400:
            if acc is not None:
                info = acc.result()
            else:
                try:
                    info = d.parse_response(json.loads(buf or b"{}"))
                except json.JSONDecodeError:
                    info = None
        self.jit.complete(res, info, status, (time.perf_counter() - t0) * 1000)

    async def _passthrough(self, scope: dict, headers: dict, body: bytes, send: Any) -> None:
        url = self._upstream("", headers) + _path_qs(scope)
        fwd_headers = {k: v for k, v in headers.items() if k not in HOP and not k.startswith("x-treejit")}
        fwd_headers["accept-encoding"] = "identity"
        request = self.client.build_request(scope["method"], url, headers=fwd_headers, content=body)
        upstream = await self.client.send(request, stream=True)
        await send({"type": "http.response.start", "status": upstream.status_code, "headers": [
            (k.encode(), v.encode()) for k, v in upstream.headers.items() if k.lower() not in RESP_DROP]})
        try:
            async for chunk in upstream.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            await upstream.aclose()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _outcome(self, body: bytes, send: Any) -> None:
        try:
            data = json.loads(body or b"{}")
            run_id = data.get("run_id") or "latest"
            result = data.get("outcome", data.get("result", data.get("pass")))
            wait = data.get("wait", True) is not False
            # off the event loop: recording the outcome touches the db, and a sync-mode rebuild is CPU-bound
            ids = await asyncio.to_thread(self.jit.outcome, run_id, result, data.get("reason"), wait)
        except (ValueError, json.JSONDecodeError) as e:
            await _json(send, 400, {"error": str(e)})
            return
        await _json(send, 200, {"updated": ids})


def _path_qs(scope: dict) -> str:
    qs = scope.get("query_string", b"")
    return scope["path"] + (("?" + qs.decode()) if qs else "")


async def _json(send: Any, status: int, obj: Any, extra: list | None = None) -> None:
    data = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()), *(extra or [])]})
    await send({"type": "http.response.body", "body": data, "more_body": False})


def serve(jit: TreeJIT, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(ProxyApp(jit), host=host or jit.cfg.host, port=port or jit.cfg.port, log_level="warning")
