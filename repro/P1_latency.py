"""/outcome latency in the proxy, and how long it blocks the event loop (every other in-flight request).

usage: P1_latency.py <db> [n_runs_cap ...]
For each cap: POST /outcome for an existing run through ProxyApp; concurrently GET /health every 5 ms and record
the worst /health latency while the outcome is being processed.
"""
import asyncio, json, os, shutil, sys, tempfile, time
from treejit import TreeJIT
from treejit.proxy import ProxyApp


async def call(app, method, path, body=b""):
    msgs = [{"type": "http.request", "body": body, "more_body": False}]
    out = {}

    async def receive():
        return msgs.pop(0) if msgs else {"type": "http.disconnect"}

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
        elif m["type"] == "http.response.body":
            out["body"] = out.get("body", b"") + m.get("body", b"")
    await app({"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}, receive, send)
    return out


async def measure(app, run_id):
    worst, done = 0.0, False

    async def health_loop():
        nonlocal worst
        last = time.perf_counter()
        while not done:
            await asyncio.sleep(0.005)          # a /health request issued every 5 ms: its latency is the loop stall
            t = time.perf_counter()
            await call(app, "GET", "/health")
            worst = max(worst, time.perf_counter() - last - 0.005)
            last = time.perf_counter()

    hl = asyncio.create_task(health_loop())
    await asyncio.sleep(0.02)
    t = time.perf_counter()
    r = await call(app, "POST", "/outcome", json.dumps({"run_id": run_id, "outcome": "pass"}).encode())
    ms = (time.perf_counter() - t) * 1000
    done = True
    await hl
    return ms, worst * 1000, r["status"]


src = sys.argv[1]
caps = [int(x) for x in sys.argv[2:]] or [100, 300, 1000, 2000]
for cap in caps:
    tmp = os.path.join(tempfile.mkdtemp(), "p.db"); shutil.copy(src, tmp)
    jit = TreeJIT(tmp, max_runs=cap)
    app = ProxyApp(jit)
    rid = jit.store.q1("SELECT id FROM runs ORDER BY created DESC LIMIT 1")["id"]
    jit.rebuild()
    ms, worst, st = asyncio.run(measure(app, rid))
    print(f"max_runs={cap:5d}: POST /outcome {ms:7.0f} ms (status {st}); event loop blocked (worst /health delay) {worst:7.0f} ms")
    jit.close()
