"""X2 live check: preserved thinking vs treejit's replayed turns and sticky hints, against a real model.

Question (PLAN X2 d): on models whose thinking can't be turned off (Claude Opus 5.5, Fable 5.1), does the
API accept assistant turns that carry no thinking block (treejit's replayed tool calls) inside a tool loop
whose other turns do carry signed thinking blocks, with the history-editing check enforced? And does
anything treejit does to the forwarded copy (sticky hints, compaction) trip the check?

Two phases, both with `thinking: {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}`
and `anthropic-beta: thinking-binding-controls-2026-08-01` (setting the field opts any account into
enforcement; the header adds `input_transformations` to every response):

  A. direct (no treejit): one real turn (thinking + tool_use), then a synthetic replay-shaped assistant turn
     (tool_use only, id `toolu_tj_...`), then a real turn after it. Answers (d) on its own.
  B. through the treejit proxy (in-process ASGI, `hints="always"`): the tree is pre-trained offline with a
     scripted model so the first steps replay (T0), then the real model continues the loop.

Every response's status, treejit tier, `input_transformations` and any 400 text is printed. Without
credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile) it prints SKIP and
exits 0. Cost: roughly 6-10 short requests at effort "low".

usage (from the repo root):
  PYTHONPATH=$PWD/src python3 repro/X2_live_check.py [--model claude-opus-5-5] [--phase A|B|AB] [--turns 6]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

BETA = "thinking-binding-controls-2026-08-01"
API = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
SYSTEM = ("You are a careful assistant with tools for a small read-only file store. Use the tools to answer; "
          "look before you answer. Keep the final answer to one sentence.")
TOOLS = [
    {"name": "list_dir", "description": "List the files in a directory.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "read_file", "description": "Read a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]
FILES = {"notes/a.txt": "alpha beta gamma", "notes/b.txt": "delta epsilon", "notes/c.txt": "zeta eta theta iota"}


def credentials() -> dict[str, str] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {"x-api-key": key}
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not token and shutil.which("ant"):
        try:
            token = subprocess.run(["ant", "auth", "print-credentials", "--access-token"], capture_output=True,
                                   text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            token = ""
    if token:
        return {"authorization": f"Bearer {token}", "x-oauth": "1"}
    return None


def headers(auth: dict[str, str]) -> dict[str, str]:
    betas = [BETA] + (["oauth-2025-04-20"] if auth.get("x-oauth") else [])
    out = {k: v for k, v in auth.items() if k != "x-oauth"}
    out.update({"anthropic-version": "2023-06-01", "anthropic-beta": ",".join(betas), "content-type": "application/json"})
    return out


def run_tool(name: str, args: dict) -> str:
    path = str(args.get("path", "")).strip("/")
    if name == "list_dir":
        return "\n".join(sorted(p for p in FILES if p.startswith(path + "/"))) or "(empty)"
    return FILES.get(path, f"no such file: {path}")


def body(model: str, messages: list, **kw) -> dict:
    return {"model": model, "max_tokens": 4096, "system": SYSTEM, "tools": TOOLS, "messages": messages,
            "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}},
            "output_config": {"effort": "low"}, **kw}


def show(label: str, status: int, resp: dict, tier: str = "") -> None:
    kinds = [b.get("type") for b in resp.get("content") or []]
    err = (resp.get("error") or {}).get("message", "") if status >= 400 else ""
    print(f"  {label:28s} status={status} {('tier=' + tier) if tier else '':9s} content={kinds} "
          f"input_transformations={json.dumps(resp.get('input_transformations'))}" + (f"\n      400: {err}" if err else ""))


def results_for(resp: dict) -> list[dict]:
    return [{"type": "tool_result", "tool_use_id": b["id"], "content": run_tool(b["name"], b.get("input") or {})}
            for b in resp.get("content") or [] if b.get("type") == "tool_use"]


# ------------------------------------------------------------------ phase A: direct


def phase_a(model: str, auth: dict) -> list[int]:
    import httpx

    print(f"\n== A: direct, {model}: real turn, replay-shaped turn without thinking, real turn ==")
    statuses = []
    with httpx.Client(timeout=300) as c:
        msgs = [{"role": "user", "content": "What files are in notes/? Then tell me what notes/a.txt and notes/b.txt say."}]
        r = c.post(API + "/v1/messages", headers=headers(auth), json=body(model, msgs))
        resp = r.json()
        show("1 real", r.status_code, resp)
        statuses.append(r.status_code)
        if r.status_code >= 400:
            return statuses
        msgs += [{"role": "assistant", "content": resp["content"]}, {"role": "user", "content": results_for(resp)}]
        # what treejit sends for a replayed step: a tool_use-only assistant turn, no thinking block
        rid = "toolu_tj_000000000000_ff00000000000000"
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": rid, "name": "read_file",
                                                    "input": {"path": "notes/a.txt"}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": rid, "content": FILES["notes/a.txt"]}]}]
        for k in range(2, 5):
            r = c.post(API + "/v1/messages", headers=headers(auth), json=body(model, msgs))
            resp = r.json()
            show(f"{k} after replay-shaped turn", r.status_code, resp)
            statuses.append(r.status_code)
            if r.status_code >= 400 or resp.get("stop_reason") != "tool_use":
                break
            msgs += [{"role": "assistant", "content": resp["content"]}, {"role": "user", "content": results_for(resp)}]
    return statuses


# ------------------------------------------------------------------ phase B: through the proxy


def pretrain(jit, model: str) -> None:
    """Three passing offline runs with a scripted model: list notes/, read the file the task names."""
    k = [0]

    def fake(b: dict) -> dict:
        task = b["messages"][0]["content"]
        task = task if isinstance(task, str) else task[0]["text"]   # a hint may follow as a second block
        n = sum(1 for m in b["messages"] if m["role"] == "assistant")
        plan = [("list_dir", {"path": "notes"}), ("read_file", {"path": task.split()[-1].rstrip("?.")})]
        k[0] += 1
        if n < len(plan):
            content = [{"type": "tool_use", "id": f"toolu_train{k[0]:012d}", "name": plan[n][0], "input": plan[n][1]}]
            return {"id": "m", "type": "message", "role": "assistant", "model": model, "content": content,
                    "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1}}
        return {"id": "m", "type": "message", "role": "assistant", "model": model, "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}], "usage": {"input_tokens": 1, "output_tokens": 1}}

    client = jit.wrap(fake, dialect="anthropic")
    for i, f in enumerate(["notes/a.txt", "notes/b.txt", "notes/c.txt"]):
        msgs = [{"role": "user", "content": f"Summarize {f}"}]
        for _ in range(4):
            resp = client({"model": model, "max_tokens": 64, "system": SYSTEM, "tools": TOOLS, "messages": msgs},
                          extra_headers={"X-TreeJIT-Run": f"train{i}"})
            msgs.append({"role": "assistant", "content": resp["content"]})
            res = results_for(resp)
            if not res:
                break
            msgs.append({"role": "user", "content": res})
        jit.outcome(f"train{i}", "pass")


def offline_upstream():
    """--offline: a scripted stand-in for the API (thinking block + the next call), to check the plumbing.
    It also enforces the two request-shape rules of the always-thinking models: no forced tool_choice and
    no `thinking` other than adaptive."""
    import httpx

    def handler(b: dict) -> tuple[int, dict]:
        if (b.get("tool_choice") or {}).get("type") in ("tool", "any") or (b.get("thinking") or {}).get("type") not in (None, "adaptive"):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "shape"}}
        done = {tuple(sorted(x["input"].items())) for m in b["messages"] if m["role"] == "assistant"
                for x in m["content"] if x.get("type") == "tool_use"}
        plan = [("list_dir", {"path": "notes"}), ("read_file", {"path": "notes/b.txt"}), ("read_file", {"path": "notes/c.txt"})]
        nxt = next((p for p in plan if tuple(sorted(p[1].items())) not in done), None)
        content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
        content += ([{"type": "tool_use", "id": f"toolu_live{len(b['messages']):010d}", "name": nxt[0], "input": nxt[1]}]
                    if nxt else [{"type": "text", "text": "b says delta epsilon; c mentions eta."}])
        return 200, {"id": "msg", "type": "message", "role": "assistant", "model": b["model"], "content": content,
                     "stop_reason": "tool_use" if nxt else "end_turn", "input_transformations": [],
                     "usage": {"input_tokens": 10, "output_tokens": 5}}

    async def app(scope, receive, send):
        raw = b""
        while True:
            m = await receive()
            raw += m.get("body", b"")
            if not m.get("more_body"):
                break
        status, out = handler(json.loads(raw))
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(out).encode()})

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=API)


async def phase_b(model: str, auth: dict, turns: int, offline: bool = False) -> list[int]:
    import httpx

    from treejit import TreeJIT
    from treejit.proxy import ProxyApp

    print(f"\n== B: through the treejit proxy (hints=always), {model}{' (offline stand-in)' if offline else ''} ==")
    tmp = tempfile.mkdtemp()
    jit = TreeJIT(os.path.join(tmp, "x2.db"), hints="always")
    jit.cfg.anthropic_upstream = API
    pretrain(jit, model)
    statuses: list[int] = []
    up = offline_upstream() if offline else httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
    proxy = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://treejit")
    try:
        msgs = [{"role": "user", "content": "Summarize notes/b.txt, then also check whether notes/c.txt mentions eta."}]
        for k in range(turns):
            r = await proxy.post("/v1/messages", json=body(model, msgs),
                                 headers={**headers(auth), "X-TreeJIT-Run": "x2-live"})
            resp = r.json()
            tier = r.headers.get("x-treejit-tier", "")
            show(f"{k + 1}", r.status_code, resp, tier)
            statuses.append(r.status_code)
            if r.status_code >= 400:
                break
            msgs.append({"role": "assistant", "content": resp["content"]})
            res = results_for(resp)
            if not res:
                break
            msgs.append({"role": "user", "content": res})
        notes = [r["tier"] + " " + (r["note"] or "") for r in jit.store.q("SELECT tier, note FROM requests WHERE run_id='x2-live'")]
        print("  treejit requests:", *notes, sep="\n    ")
    finally:
        await proxy.aclose()
        await up.aclose()
        jit.close()
    return statuses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="claude-opus-5-5", help="e.g. claude-opus-5-5, claude-fable-5-1")
    ap.add_argument("--phase", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--offline", action="store_true", help="phase B against a scripted stand-in (plumbing check)")
    args = ap.parse_args()
    if args.offline:
        statuses = asyncio.run(phase_b(args.model, {"x-api-key": "offline"}, args.turns, offline=True))
        print(f"\noffline plumbing check: {statuses}")
        return 1 if any(s >= 400 for s in statuses) else 0
    auth = credentials()
    if auth is None:
        print("SKIP: no credentials (set ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or run `ant auth login`)")
        return 0
    if importlib.util.find_spec("httpx") is None:
        print("SKIP: needs httpx")
        return 0
    statuses = []
    if "A" in args.phase:
        statuses += phase_a(args.model, auth)
    if "B" in args.phase:
        statuses += asyncio.run(phase_b(args.model, auth, args.turns))
    bad = [s for s in statuses if s >= 400]
    print(f"\nresult: {len(statuses)} requests, {len(bad)} errors {bad}")
    print("replayed turns without thinking blocks were " + ("REJECTED (see the 400s above)" if 400 in bad else
                                                           "accepted" if not bad else "inconclusive (non-400 errors)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
