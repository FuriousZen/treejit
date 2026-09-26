"""Drive the simulated agent through the suite with treejit off and on.

Modes
  baseline      the agent talks to the simulated model directly
  treejit       inline mode: treejit.wrap(model); read-only tools replay freely,
                everything else goes to the model (allowlist only)
  treejit+ok    same, with the operator having approved all edges (`treejit approve '*'`),
                so proven write steps and commit points can replay too
  proxy modes   `--via-proxy` runs the treejit modes through the real ASGI proxy with
                streamed (SSE) responses instead of inline mode
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from treejit import TreeJIT
from treejit.util import now

from .sim import SimModel, make_task

MODEL_BASE_MS = 600.0   # simulated time-to-first-token
MODEL_MS_PER_TOKEN = 15.0


@dataclass
class TaskResult:
    mode: str
    index: int
    family: str
    kind: str
    success: bool
    reason: str
    model_calls: int = 0      # full model calls (T4, or every call for the plain agent)
    small_calls: int = 0      # treejit T2/T3 subcalls (short prompt, forced tool)
    tool_calls: int = 0
    replayed_calls: int = 0
    side_exits: int = 0       # replayed steps whose result broke the learned postcondition
    input_tokens: int = 0
    output_tokens: int = 0    # both include subcall tokens
    small_tokens: int = 0
    model_ms: float = 0.0
    engine_ms: float = 0.0
    tiers: list = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def wall_ms(self) -> float:
        return self.model_ms + self.engine_ms


def _kind(env: Any) -> str:
    return getattr(env, "kind", None) or "retail"


def _execute(env: Any, content: list[dict]) -> list[dict]:
    out = []
    for b in content:
        if b.get("type") == "tool_use":
            text, err = env.run(b["name"], b.get("input") or {})
            out.append({"type": "tool_result", "tool_use_id": b["id"], "content": text, "is_error": err})
    return out


def run_suite(n_tasks: int, seed: int = 0, family: str = "mixed", mode: str = "treejit", noise: float = 0.06,
              db: str | None = None, max_steps: int = 30, **overrides: Any) -> list[TaskResult]:
    rng = random.Random(seed)
    tasks = [make_task(rng, i, family) for i in range(n_tasks)]
    model = SimModel(seed + 1, noise)
    jit = None
    call: Callable[[dict, str], dict]
    if mode == "baseline":
        call = lambda body, run_id: model(body)  # noqa: E731
    else:
        jit = _fresh_jit(db, mode, **overrides)
        wrapped = jit.wrap(model, dialect="anthropic")
        call = lambda body, run_id: wrapped(body, extra_headers={"X-TreeJIT-Run": run_id})  # noqa: E731
    results = []
    for i, (fam, text, env, tools, system) in enumerate(tasks):
        r = TaskResult(mode, i, fam, _kind(env), False, "")
        msgs: list[dict] = [{"role": "user", "content": text}]
        for _ in range(max_steps):
            body = {"model": "sim-1", "max_tokens": 1024, "system": system, "tools": tools, "messages": msgs}
            before = _snap(model)
            t0 = time.perf_counter()
            resp = call(body, f"task-{seed}-{i}")
            ms = (time.perf_counter() - t0) * 1000
            _account(r, resp, before, _snap(model), ms)
            msgs.append({"role": "assistant", "content": resp["content"]})
            results_blocks = _execute(env, resp["content"])
            if not results_blocks:
                break
            msgs.append({"role": "user", "content": results_blocks})
        r.success, r.reason = env.verify()
        if jit is not None:
            r.side_exits = _side_exits(jit, f"task-{seed}-{i}")
            jit.outcome(f"task-{seed}-{i}", r.success, None if r.success else r.reason)
        results.append(r)
    if jit is not None:
        jit.close()
    return results


def _fresh_jit(db: str | None, mode: str, **overrides: Any) -> TreeJIT:
    path = db or ":memory:"
    if path != ":memory:":
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
    jit = TreeJIT(path, **overrides)
    if mode.endswith("+ok"):
        jit.store.x("INSERT OR REPLACE INTO approvals(edge, node, ts) VALUES('*', '', ?)", (now(),))
    return jit


def _side_exits(jit: TreeJIT, run_id: str) -> int:
    row = jit.store.q1("SELECT COUNT(*) n FROM requests WHERE run_id=? AND note LIKE 'side_exit%'", (run_id,))
    return int(row["n"]) if row else 0


def _snap(model: SimModel) -> tuple[int, int, int, int]:
    return model.calls, model.small_calls, model.small_tokens[0], model.small_tokens[1]


def _account(r: TaskResult, resp: dict, before: tuple, after: tuple, ms: float) -> None:
    uses = [b for b in resp.get("content", []) if b.get("type") == "tool_use"]
    r.tool_calls += len(uses)
    from_model = after[0] > before[0]
    small = after[1] - before[1]
    if small:
        s_in, s_out = after[2] - before[2], after[3] - before[3]
        r.small_calls += small
        r.small_tokens += s_in + s_out
        r.input_tokens += s_in
        r.output_tokens += s_out
        r.model_ms += small * MODEL_BASE_MS + MODEL_MS_PER_TOKEN * s_out
    if from_model:
        u = resp.get("usage") or {}
        r.model_calls += 1
        r.input_tokens += int(u.get("input_tokens") or 0)
        r.output_tokens += int(u.get("output_tokens") or 0)
        r.model_ms += MODEL_BASE_MS + MODEL_MS_PER_TOKEN * int(u.get("output_tokens") or 0)
        r.engine_ms += ms  # proxy/engine overhead on a forwarded call (the simulated model itself is instant)
        r.tiers.append("T4")
    else:
        r.replayed_calls += len(uses)
        r.engine_ms += ms
        r.tiers.append("S" if small else "R")


# ============================================================================ via the real proxy


def _upstream_app(model: SimModel) -> Any:
    """A fake Anthropic API (ASGI) serving the simulated model, JSON or SSE."""

    async def app(scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        req = json.loads(body)
        resp = model(req)
        if not req.get("stream"):
            data = json.dumps(resp).encode()
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": data})
            return
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        for chunk in _sse_events(resp):
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})

    return app


def _sse_events(resp: dict) -> list[bytes]:
    def ev(name: str, data: dict) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    start = dict(resp, content=[], stop_reason=None, usage={"input_tokens": resp["usage"]["input_tokens"], "output_tokens": 1})
    out = [ev("message_start", {"type": "message_start", "message": start})]
    for i, b in enumerate(resp["content"]):
        if b["type"] == "text":
            out.append(ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
            out.append(ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}))
        else:
            out.append(ev("content_block_start", {"type": "content_block_start", "index": i,
                                                  "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}))
            js = json.dumps(b["input"])
            for k in range(0, len(js), 16):  # split JSON across deltas like the real API
                out.append(ev("content_block_delta", {"type": "content_block_delta", "index": i,
                                                      "delta": {"type": "input_json_delta", "partial_json": js[k:k + 16]}}))
        out.append(ev("content_block_stop", {"type": "content_block_stop", "index": i}))
    out.append(ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": resp["stop_reason"]},
                                    "usage": {"output_tokens": resp["usage"]["output_tokens"]}}))
    out.append(ev("message_stop", {"type": "message_stop"}))
    return out


def parse_anthropic_sse(raw: bytes) -> dict:
    """Independent (harness-side) reconstruction of a streamed Anthropic message."""
    msg: dict = {}
    blocks: dict[int, dict] = {}
    for part in raw.decode().split("\n\n"):
        data = [line[6:] for line in part.split("\n") if line.startswith("data: ")]
        if not data:
            continue
        e = json.loads(data[0])
        t = e["type"]
        if t == "message_start":
            msg = e["message"]
        elif t == "content_block_start":
            blocks[e["index"]] = dict(e["content_block"], _j="")
        elif t == "content_block_delta":
            d = e["delta"]
            if d["type"] == "text_delta":
                blocks[e["index"]]["text"] += d["text"]
            else:
                blocks[e["index"]]["_j"] += d["partial_json"]
        elif t == "message_delta":
            msg["stop_reason"] = e["delta"].get("stop_reason")
            msg.setdefault("usage", {})["output_tokens"] = e.get("usage", {}).get("output_tokens", 0)
    content = []
    for i in sorted(blocks):
        b = blocks[i]
        j = b.pop("_j")
        if b["type"] == "tool_use":
            b["input"] = json.loads(j) if j else {}
        content.append(b)
    msg["content"] = content
    return msg


async def _run_proxy_async(n_tasks: int, seed: int, family: str, mode: str, noise: float, db: str | None,
                           max_steps: int, stream: bool, **overrides: Any) -> list[TaskResult]:
    import httpx

    from treejit.proxy import ProxyApp

    rng = random.Random(seed)
    tasks = [make_task(rng, i, family) for i in range(n_tasks)]
    model = SimModel(seed + 1, noise)
    jit = _fresh_jit(db, mode, **overrides)
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=_upstream_app(model)), base_url="http://upstream")
    jit.cfg.anthropic_upstream = "http://upstream"
    proxy = ProxyApp(jit, client=upstream)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy), base_url="http://treejit")
    results = []
    try:
        for i, (fam, text, env, tools, system) in enumerate(tasks):
            r = TaskResult(mode + "@proxy", i, fam, _kind(env), False, "")
            run_id = f"task-{seed}-{i}"
            msgs: list[dict] = [{"role": "user", "content": text}]
            for _ in range(max_steps):
                body = {"model": "sim-1", "max_tokens": 1024, "system": system, "tools": tools, "messages": msgs, "stream": stream}
                before = _snap(model)
                t0 = time.perf_counter()
                http = await client.post("/v1/messages", json=body, headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01",
                                                                             "X-TreeJIT-Run": run_id})
                ms = (time.perf_counter() - t0) * 1000
                http.raise_for_status()
                resp = parse_anthropic_sse(http.content) if stream else http.json()
                _account(r, resp, before, _snap(model), ms)
                msgs.append({"role": "assistant", "content": resp["content"]})
                blocks = _execute(env, resp["content"])
                if not blocks:
                    break
                msgs.append({"role": "user", "content": blocks})
            r.success, r.reason = env.verify()
            r.side_exits = _side_exits(jit, run_id)
            out = await client.post("/outcome", json={"run_id": run_id, "outcome": "pass" if r.success else "fail",
                                                      "reason": None if r.success else r.reason})
            out.raise_for_status()
            results.append(r)
    finally:
        await client.aclose()
        await upstream.aclose()
        jit.close()
    return results


def run_suite_proxy(n_tasks: int, seed: int = 0, family: str = "mixed", mode: str = "treejit", noise: float = 0.06,
                    db: str | None = None, max_steps: int = 30, stream: bool = True, **overrides: Any) -> list[TaskResult]:
    return asyncio.run(_run_proxy_async(n_tasks, seed, family, mode, noise, db, max_steps, stream, **overrides))


def to_rows(results: list[TaskResult]) -> list[dict]:
    rows = []
    for r in results:
        d = asdict(r)
        d["tiers"] = "".join(t if t in ("R", "S") else "M" for t in r.tiers)
        d["tokens"] = r.tokens
        d["wall_ms"] = round(r.wall_ms, 1)
        d["model_ms"] = round(r.model_ms, 1)
        d["engine_ms"] = round(r.engine_ms, 2)
        rows.append(d)
    return rows
