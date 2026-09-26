"""T1: inline-mode streaming passes straight through: no replay, no recording, no END, and the
X-TreeJIT-Run header leaks upstream."""
import sys
import tempfile

sys.path.insert(0, "/home/user/treejit/tests")
from conftest import SYSTEM, TOOLS, Model  # noqa: E402

from treejit import TreeJIT  # noqa: E402


def policy(task, hist, body):
    plan = [("Bash", {"command": "ls"}), ("Bash", {"command": "cat README.md"})]
    return plan[len(hist)] if len(hist) < len(plan) else None


def execute(name, args):
    return ("README.md\nsrc" if args["command"] == "ls" else "# readme\n" * 5), False


def to_events(msg):
    """Anthropic raw stream events (as dicts; the SDK yields RawMessageStreamEvent objects with .type)."""
    yield {"type": "message_start", "message": dict(msg, content=[], stop_reason=None)}
    for i, b in enumerate(msg["content"]):
        if b["type"] == "tool_use":
            import json
            yield {"type": "content_block_start", "index": i, "content_block": dict(b, input={})}
            yield {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}}
        else:
            yield {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}
        yield {"type": "content_block_stop", "index": i}
    yield {"type": "message_delta", "delta": {"stop_reason": msg["stop_reason"]}, "usage": {"output_tokens": 7}}
    yield {"type": "message_stop"}


def accumulate(events):
    import json
    blocks, stop = {}, None
    for e in events:
        if e["type"] == "content_block_start":
            blocks[e["index"]] = dict(e["content_block"], _j="")
        elif e["type"] == "content_block_delta":
            d = e["delta"]
            if d["type"] == "input_json_delta":
                blocks[e["index"]]["_j"] += d["partial_json"]
            else:
                blocks[e["index"]]["text"] += d["text"]
        elif e["type"] == "message_delta":
            stop = e["delta"]["stop_reason"]
    out = []
    for _, b in sorted(blocks.items()):
        j = b.pop("_j")
        if b["type"] == "tool_use":
            b["input"] = json.loads(j or "{}")
        out.append(b)
    return out, stop


class FakeMessages:
    def __init__(self):
        self.model = Model(policy)
        self.upstream = 0
        self.headers_seen = []

    def create(self, extra_headers=None, **body):
        self.upstream += 1
        self.headers_seen.append(dict(extra_headers or {}))
        msg = self.model(body)
        return to_events(msg) if body.get("stream") else msg


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def run(client, task, rid, stream):
    msgs = [{"role": "user", "content": task}]
    for _ in range(6):
        out = client.messages.create(model="m", max_tokens=100, system=SYSTEM, tools=TOOLS, messages=msgs, stream=stream,
                                     extra_headers={"X-TreeJIT-Run": rid})
        if stream:
            content, _ = accumulate(out)
        else:
            content = out["content"]
        msgs.append({"role": "assistant", "content": content})
        uses = [b for b in content if b["type"] == "tool_use"]
        if not uses:
            break
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": execute(u["name"], u["input"])[0]} for u in uses]})
    return msgs


with tempfile.TemporaryDirectory() as tmp:
    jit = TreeJIT(tmp + "/t.db")
    fake = FakeClient()
    client = jit.wrap(fake)
    for i in range(3):
        run(client, f"inspect repo {i}", f"train{i}", stream=False)
        jit.outcome(f"train{i}", "pass")
    n0 = fake.messages.upstream
    run(client, "inspect repo 9", "probe-json", stream=False)
    print(f"non-streaming run after training: upstream calls = {fake.messages.upstream - n0} (replayed steps served locally)")
    n1 = fake.messages.upstream
    msgs = run(client, "inspect repo 10", "probe-sse", stream=True)
    print(f"streaming run after training:     upstream calls = {fake.messages.upstream - n1} (expected 1 if replayed)")
    ids = [b["id"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    print(f"  streamed tool ids: {ids} (no '_tj_' marker -> nothing replayed)")
    for rid in ("probe-json", "probe-sse"):
        r = jit.store.run(rid)
        nreq = jit.store.q1("SELECT COUNT(*) n FROM requests WHERE run_id=?", (rid,))["n"]
        print(f"  run {rid}: runs row={'yes' if r else 'NO'}, steps={jit.store.n_steps(rid)}, requests logged={nreq}, "
              f"ended_after={r['ended_after'] if r else None}")
    print(f"  headers the upstream saw on the last streaming call: {fake.messages.headers_seen[-1]}")
    print(f"  headers the upstream saw on the last non-streaming call: {fake.messages.headers_seen[n1 - 1]}")
    jit.close()
