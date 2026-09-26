"""C1: frontier compaction vs the Anthropic prompt cache.

Two traffic patterns over a 15-step episode (all steps big read-only `cat`s):
  dense       : every request is forwarded (T4) while all earlier steps were replayed
                (upper bound on how often the keep-last boundary flips under a warm cache)
  interleaved : real engine run; reads replay, every odd step is a write the model makes
                (`touch`, not approved -> T4). Forwarded bodies are captured from the upstream stub.

Policies compared (compaction.apply is the real code; only cfg.compact_keep_last is varied per call):
  off          compaction disabled
  current      keep_last = 3 (window moves by one step each step)
  chunk-K      boundary advances only every K steps: keep_eff = 3 + ((n-3) % K)
  first-sight  append-only: a step is compacted only if it is outside keep-last the FIRST time it is
               forwarded; a step that was ever sent in full stays full (emulated here by remembering ids)

Cache model (assumptions, stated in the report):
  - breakpoints: one explicit at end of system (static), plus automatic (top-level cache_control)
    at the end of every request -> reads land only on positions where an earlier request wrote one
  - read = longest earlier breakpoint position p such that bytes[:p] are identical
  - billed input = 0.1*read + 1.25*(T - read) (5-min TTL, all within TTL, lookback not binding)
  - tokens = chars / 4 of canonical JSON (tools, system, then each message)
"""
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, "/home/user/treejit/tests")
from conftest import Model, run_agent  # noqa: E402

from treejit import TreeJIT, compaction, dialects, families  # noqa: E402

N = 15
BIG_SYSTEM = int(os.environ.get("BIG_SYSTEM", "0"))  # extra system chars (Claude Code ~ 60k chars)
SYSTEM = "You are a careful agent working in a repository.\n" * 8 + ("static instructions line\n" * (BIG_SYSTEM // 24))
TOOLS = [
    {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
]


def content(k, task):
    return "\n".join(f"{task}: file{k} line {j} " + "x" * 40 for j in range(50))  # ~3.3k chars


def make_policy(pattern):
    def policy(task, hist, body):
        i = len(hist)
        if i >= N:
            return None
        if (pattern == "interleaved" and i % 2 == 1) or (pattern == "bursty" and i % 5 == 4):
            return ("Bash", {"command": f"touch mark_{i}"})
        return ("Bash", {"command": f"cat file_{i}.txt"})
    return policy


def execute(task):
    def ex(name, args):
        cmd = args["command"]
        if cmd.startswith("cat"):
            return content(int(cmd.split("_")[1].split(".")[0]), task), False
        return "", False
    return ex


def canon_parts(body):
    parts = [json.dumps(body.get("tools"), sort_keys=True, separators=(",", ":")),
             json.dumps(body.get("system"), sort_keys=True, separators=(",", ":"))]
    parts += [json.dumps(m, sort_keys=True, separators=(",", ":")) for m in body["messages"]]
    return parts


def positions(parts):
    out, s = [], 0
    for p in parts:
        s += len(p)
        out.append(s)
    return out


def analyze(bodies, label, frontier_bp=False):
    """returns per-request rows and totals"""
    written = []   # (serialized string, set of breakpoint positions)
    rows = []
    prev = None
    tot = {"T": 0, "billed": 0.0, "read": 0}
    for b in bodies:
        parts = canon_parts(b)
        s = "".join(parts)
        pos = positions(parts)
        common = len(os.path.commonprefix([prev, s])) if prev is not None else 0
        read = 0
        for ws, wpos in written:
            for p in wpos:
                if p <= len(s) and p > read and ws[:p] == s[:p]:
                    read = p
        T = len(s)
        billed = 0.1 * read + 1.25 * (T - read)
        rows.append((T // 4, common // 4, read // 4, round(billed / 4)))
        tot["T"] += T // 4
        tot["read"] += read // 4
        tot["billed"] += billed / 4
        bps = {pos[1], pos[-1]}   # end of system, end of request
        if frontier_bp:  # treejit adds a breakpoint on the last message that carries a digest
            idx = [i for i, p in enumerate(parts) if "[treejit:" in p]
            if idx:
                bps.add(pos[idx[-1]])
        written.append((s, bps))
        prev = s
    return rows, tot


def train(jit, pattern, n_runs=3):
    model = Model(make_policy(pattern))
    client = jit.wrap(model, dialect="anthropic")
    last = None
    for i in range(n_runs):
        rid = f"train{i}"
        last = run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": rid}), f"task {i}", execute(f"task {i}"),
                         max_steps=N + 2, tools=TOOLS, system=SYSTEM)
        jit.outcome(rid, "pass")
    return model, last


class SnapModel(Model):
    def __call__(self, body):
        return super().__call__(copy.deepcopy(body))


class Variant:
    def __init__(self, name, K=1, first_sight=False, bp=False):
        self.name, self.K, self.first_sight, self.bp = name, K, first_sight, bp
        self.sent_full: set[str] = set()


def patched_apply(variant, orig):
    def apply(store, view, cfg, req, body):
        n = len(req.episode.steps)
        keep0 = 3
        cfg.compact_keep_last = keep0 + (max(0, n - keep0) % variant.K)
        steps = req.episode.steps
        if variant.first_sight:
            # a step already sent full once stays full: hide it from compaction by pretending it's model-chosen
            saved = [(st, st.call.id) for st in steps if st.call.id in variant.sent_full]
            for st, _ in saved:
                st.call = copy.copy(st.call)
                st.call.id = "toolu_model_" + st.call.id  # no replay marker -> never compacted
            res = orig(store, view, cfg, req, body)
            for st, cid in saved:
                st.call.id = cid
            # everything forwarded in full this time is remembered
            got = {}
            for m in res.body["messages"]:
                for bl in m["content"] if isinstance(m.get("content"), list) else []:
                    if bl.get("type") == "tool_result":
                        got[bl["tool_use_id"]] = bl["content"]
            for st in steps:
                if st.call.id in got and not str(got[st.call.id]).startswith("[treejit"):
                    variant.sent_full.add(st.call.id)
            cfg.compact_keep_last = keep0
            return res
        res = orig(store, view, cfg, req, body)
        cfg.compact_keep_last = keep0
        return res
    return apply


def dense_bodies(variant, tmp):
    jit = TreeJIT(os.path.join(tmp, f"dense_{variant.name}.db"), compact=variant.name != "off", theta=0.0,
                  hard_cap=100, max_depth=20, batch=False, t2=False, t3=False)
    _, last = train(jit, "dense")
    ids = [b["id"] for m in last if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    assert all("_tj_" in i for i in ids), ids
    d = dialects.get("anthropic")
    orig = compaction.apply
    ap = patched_apply(variant, orig) if variant.name != "off" else None
    bodies = []
    # request n = conversation after n steps (n=1..15): the model is asked for step n+1 / the answer
    for n in range(1, N + 1):
        body = {"model": "m", "max_tokens": 100, "system": SYSTEM, "tools": TOOLS, "messages": copy.deepcopy(last[: 1 + 2 * n])}
        req = d.parse_request(body)
        fam = families.resolve(jit.store, req.system, req.tools, "anthropic")
        view = jit.view(fam)
        fwd = d.prepare_forward(req)
        if ap is not None:
            fwd = ap(jit.store, view, jit.cfg, req, fwd).body
        bodies.append(fwd)
    jit.close()
    return bodies


def interleaved_bodies(variant, tmp, pattern="interleaved"):
    jit = TreeJIT(os.path.join(tmp, f"{pattern}_{variant.name}.db"), compact=variant.name != "off", theta=0.0,
                  hard_cap=100, max_depth=20, batch=False, t2=False, t3=False)
    train(jit, pattern)
    orig = compaction.apply
    if variant.name != "off":
        compaction.apply = patched_apply(variant, orig)
    try:
        model = SnapModel(make_policy(pattern))
        client = jit.wrap(model, dialect="anthropic")
        msgs = run_agent(lambda b: client(b, extra_headers={"X-TreeJIT-Run": "probe"}), "task 9", execute("task 9"),
                         max_steps=N + 2, tools=TOOLS, system=SYSTEM)
    finally:
        compaction.apply = orig
    nrep = sum(1 for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use" and "_tj_" in b["id"])
    jit.close()
    return model.bodies, nrep


def main():
    variants = lambda: [Variant("off"), Variant("current"), Variant("chunk-3", K=3), Variant("chunk-5", K=5), Variant("current+bp", bp=True), Variant("chunk-3+bp", K=3, bp=True),
                        Variant("first-sight", first_sight=True)]
    with tempfile.TemporaryDirectory() as tmp:
        for pattern in ("dense", "interleaved", "bursty"):
            print(f"\n=== pattern: {pattern}  (system chars={len(SYSTEM)}) ===")
            summary = []
            for v in variants():
                if pattern == "dense":
                    bodies, extra = dense_bodies(v, tmp), ""
                else:
                    bodies, nrep = interleaved_bodies(v, tmp, pattern)
                    extra = f" replayed_steps={nrep}"
                rows, tot = analyze(bodies, v.name, frontier_bp=v.bp)
                if v.name in ("off", "current", "first-sight") and os.environ.get("ROWS"):
                    print(f"-- {v.name}{extra}: per forwarded request (tokens): total / common-prefix-with-prev / cache-read / billed-equiv")
                    for i, r in enumerate(rows):
                        print(f"   req{i+1:2d}  T={r[0]:6d}  common={r[1]:6d}  read={r[2]:6d}  billed={r[3]:6d}")
                summary.append((v.name, len(bodies), tot["T"], tot["read"], round(tot["billed"])))
            base_T = summary[0][2]
            base_b = summary[0][4]
            print(f"\n   {'policy':12s} {'reqs':>4s} {'sum input tok':>14s} {'cache read':>11s} {'billed-equiv':>13s} {'vs off (raw)':>13s} {'vs off (billed)':>16s}")
            for name, nreq, T, rd, b in summary:
                print(f"   {name:12s} {nreq:4d} {T:14,d} {rd:11,d} {b:13,d} {100*(T-base_T)/base_T:+12.1f}% {100*(b-base_b)/base_b:+15.1f}%")


if __name__ == "__main__":
    main()
