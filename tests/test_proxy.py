"""Proxy mode end to end: harness -> ProxyApp -> fake upstream, all in-process over ASGI."""

from __future__ import annotations

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from conftest import SYSTEM, TOOLS, Model  # noqa: E402

from treejit import TreeJIT  # noqa: E402
from treejit.proxy import ProxyApp  # noqa: E402
from treejit_bench.runner import _sse_events, parse_anthropic_sse  # noqa: E402


def upstream_app(model, seen: list):
    async def app(scope, receive, send):
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        seen.append((scope["path"], dict((k.decode(), v.decode()) for k, v in scope["headers"]), body))
        if scope["path"] == "/v1/messages/count_tokens":
            data = json.dumps({"input_tokens": 42}).encode()
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": data})
            return
        req = json.loads(body)
        if req.get("model") == "overloaded":
            await send({"type": "http.response.start", "status": 529, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"type":"error","error":{"type":"overloaded_error"}}'})
            return
        resp = model(req)
        if req.get("stream"):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            for c in _sse_events(resp):
                await send({"type": "http.response.body", "body": c, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        else:
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": json.dumps(resp).encode()})
    return app


def policy(task, hist, body):
    path = task.split()[-1]
    plan = [("Bash", {"command": "git status --short"}), ("Read", {"file_path": path})]
    return plan[len(hist)] if len(hist) < len(plan) else None


async def run_task(client, task, rid, stream=True):
    msgs = [{"role": "user", "content": task}]
    tiers = []
    for _ in range(10):
        body = {"model": "m", "max_tokens": 64, "system": SYSTEM, "tools": TOOLS, "messages": msgs, "stream": stream}
        r = await client.post("/v1/messages", json=body, headers={"x-api-key": "k", "anthropic-version": "2023-06-01", "X-TreeJIT-Run": rid})
        assert r.status_code == 200, r.text
        tiers.append(r.headers.get("x-treejit-tier"))
        if stream:
            assert r.headers["content-type"].startswith("text/event-stream")
            resp = parse_anthropic_sse(r.content)
        else:
            resp = r.json()
        msgs.append({"role": "assistant", "content": resp["content"]})
        uses = [b for b in resp["content"] if b["type"] == "tool_use"]
        if not uses:
            break
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "ok"} for u in uses]})
    o = await client.post("/outcome", json={"run_id": rid, "outcome": "pass"})
    assert o.status_code == 200 and o.json()["updated"] == [rid]
    return msgs, tiers


@pytest.mark.parametrize("stream", [True, False])
def test_proxy_records_learns_and_replays(tmp_path, stream):
    async def main():
        jit = TreeJIT(str(tmp_path / "p.db"))
        jit.cfg.anthropic_upstream = "http://up"
        model, seen = Model(policy), []
        up = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream_app(model, seen)), base_url="http://up")
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://tj")
        for i in range(2):
            _, tiers = await run_task(client, f"look at src/f{i}.py", f"r{i}", stream)
            assert set(tiers) == {"T4"}
        calls_before = model.calls
        msgs, tiers = await run_task(client, "look at src/new.py", "r9", stream)
        assert tiers[0] == "T0" and tiers[-1] == "T4"
        assert model.calls - calls_before == 1
        read = [b for m in msgs if m["role"] == "assistant" for b in m["content"] if b.get("name") == "Read"][0]
        assert read["input"] == {"file_path": "src/new.py"} and "_tj_" in read["id"]
        # upstream never saw treejit headers; auth headers pass through
        assert all("x-treejit-run" not in h for _, h, _ in seen) and all(h.get("x-api-key") == "k" for _, h, _ in seen)
        usage = jit.store.q1("SELECT SUM(input_tokens) t FROM requests WHERE tier='T4'")["t"]
        assert usage and usage > 0
        await client.aclose()
        await up.aclose()
        jit.close()

    asyncio.run(main())


def test_proxy_passthrough_errors_health_and_bad_outcome(tmp_path):
    async def main():
        jit = TreeJIT(str(tmp_path / "p.db"))
        jit.cfg.anthropic_upstream = "http://up"
        model, seen = Model(policy), []
        up = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream_app(model, seen)), base_url="http://up")
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://tj")
        r = await client.post("/v1/messages/count_tokens", json={"messages": []}, headers={"x-api-key": "k"})
        assert r.json() == {"input_tokens": 42}
        r = await client.post("/v1/messages", json={"model": "overloaded", "tools": TOOLS, "system": SYSTEM,
                                                    "messages": [{"role": "user", "content": "x"}]}, headers={"x-api-key": "k"})
        assert r.status_code == 529
        assert jit.store.q1("SELECT status FROM requests ORDER BY id DESC LIMIT 1")["status"] == 529
        assert (await client.get("/health")).json() == {"ok": True}
        assert (await client.post("/outcome", json={"run_id": "x", "outcome": "maybe"})).status_code == 400
        # a request without tools is forwarded untouched and never recorded as a run
        r = await client.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "title?"}]}, headers={"x-api-key": "k"})
        assert r.status_code == 200 and jit.store.q1("SELECT COUNT(*) n FROM runs")["n"] == 0
        await client.aclose()
        await up.aclose()
        jit.close()

    asyncio.run(main())
