"""E4: proxy + SSE traffic WITHOUT X-TreeJIT-Run (derived run ids), outcome via run_id="latest"
(like a Claude Code Stop hook), then check `treejit explain` against the requests table.

usage: python E4_explain_proxy.py [--tasks 80] [--seed 0] [--family mixed] [--mode treejit+ok]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys

sys.path[:0] = ["/home/user/treejit/src", "/home/user/treejit/bench/src"]
import httpx  # noqa: E402

from treejit import operate  # noqa: E402
from treejit.proxy import ProxyApp  # noqa: E402
from treejit_bench.runner import _execute, _fresh_jit, _upstream_app, parse_anthropic_sse  # noqa: E402
from treejit_bench.sim import SimModel, make_task  # noqa: E402
import random  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


async def main(n, seed, family, mode, db):
    rng = random.Random(seed)
    tasks = [make_task(rng, i, family) for i in range(n)]
    model = SimModel(seed + 1)
    jit = _fresh_jit(db, mode)
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=_upstream_app(model)), base_url="http://upstream")
    jit.cfg.anthropic_upstream = "http://upstream"
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ProxyApp(jit, client=up)), base_url="http://treejit")
    truth = []  # per task: (tiers seen in x-treejit-tier headers, run ids seen, sim tokens, success)
    for i, (fam, text, env, tools, system) in enumerate(tasks):
        msgs = [{"role": "user", "content": text}]
        tiers, rids = [], set()
        for _ in range(30):
            body = {"model": "sim-1", "max_tokens": 1024, "system": system, "tools": tools, "messages": msgs, "stream": True}
            r = await client.post("/v1/messages", json=body, headers={"x-api-key": "k", "anthropic-version": "2023-06-01"})
            r.raise_for_status()
            tiers.append(r.headers.get("x-treejit-tier"))
            if r.headers.get("x-treejit-run"):
                rids.add(r.headers["x-treejit-run"])
            resp = parse_anthropic_sse(r.content)
            msgs.append({"role": "assistant", "content": resp["content"]})
            blocks = _execute(env, resp["content"])
            if not blocks:
                break
            msgs.append({"role": "user", "content": blocks})
        ok, why = env.verify()
        o = await client.post("/outcome", json={"run_id": "latest", "outcome": "pass" if ok else "fail", "reason": None if ok else why})
        truth.append({"i": i, "tiers": tiers, "rids": sorted(rids), "outcome_updated": o.json()["updated"], "ok": ok})
    await client.aclose()
    await up.aclose()
    return jit, truth


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--family", default="mixed")
    p.add_argument("--mode", default="treejit+ok")
    a = p.parse_args()
    db = os.path.join(HERE, f"E4_{a.mode}_s{a.seed}.db")
    jit, truth = asyncio.run(main(a.tasks, a.seed, a.family, a.mode, db))
    st = jit.store
    # 1. every request row with a family should belong to a run
    orphans = st.q("SELECT id, tier, note, input_tokens+output_tokens tok FROM requests WHERE family IS NOT NULL AND run_id IS NULL")
    print(f"orphan request rows (family set, run_id NULL): {len(orphans)}")
    for o in orphans[:8]:
        print("   ", dict(o))
    # 2. outcome bookkeeping: did 'latest' hit the right run, one run per task?
    multi = [t for t in truth if len(t["rids"]) != 1]
    wrong = [t for t in truth if t["outcome_updated"] != t["rids"][-1:]]
    print(f"tasks with !=1 run id in headers: {len(multi)}; tasks whose 'latest' outcome hit a different run: {len(wrong)}")
    for t in (multi + wrong)[:5]:
        print("   ", t)
    runs_no_outcome = st.q("SELECT id FROM runs WHERE outcome IS NULL")
    print(f"runs left without outcome: {len(runs_no_outcome)}")
    # 3. explain vs requests table, per run
    tok_mismatch, tier_mismatch = 0, 0
    tier_tot: dict[str, int] = {}
    for t in truth:
        for rid in t["rids"]:
            d = operate.explain(st, rid)
            q = st.q1("SELECT COALESCE(SUM(input_tokens+output_tokens+cache_read+cache_write),0) n FROM requests WHERE run_id=?", (rid,))
            if d["tokens"] != q["n"]:
                tok_mismatch += 1
            hdr = {}
            for x in t["tiers"]:
                hdr[x] = hdr.get(x, 0) + 1
            exp = {k: v for k, v in d["requests_by_tier"].items()}
            # a T2/T3 request that failed is logged under T2/T3 AND the forward under T4; headers show only T4
            for k, v in exp.items():
                tier_tot[k] = tier_tot.get(k, 0) + v
            got_hdr = {k: v for k, v in hdr.items() if k not in ("T2", "T3")}
            exp_nosub = {k: v for k, v in exp.items() if k not in ("T2", "T3")}
            if got_hdr != exp_nosub and not (hdr.get("T2") or hdr.get("T3")):
                tier_mismatch += 1
    print(f"runs whose explain tokens != requests-table tokens: {tok_mismatch}; tier-count mismatches vs response headers: {tier_mismatch}")
    print("requests by tier over all runs (explain):", tier_tot)
    print("requests by tier (table, family not null):", {r['tier']: r['n'] for r in st.q("SELECT tier, COUNT(*) n FROM requests WHERE family IS NOT NULL GROUP BY tier")})
    jit.close()
    # 4. print a few CLI explains that contain T2/T3 steps
    import sqlite3
    con = sqlite3.connect(db)
    picks = [r[0] for r in con.execute("SELECT DISTINCT run_id FROM requests WHERE tier IN ('T2','T3') AND run_id IS NOT NULL ORDER BY id LIMIT 40")]
    con.close()
    shown = 0
    for rid in picks[::8][:3]:
        out = subprocess.run([sys.executable, "-m", "treejit", "--db", db, "explain", rid], capture_output=True, text=True,
                             env=dict(os.environ, PYTHONPATH="/home/user/treejit/src"))
        print(f"\n$ treejit explain {rid} --db {os.path.basename(db)}\n{out.stdout}{out.stderr}")
