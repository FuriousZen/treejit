"""Operator views: short-id resolution, the approval queue (`treejit pending`,
`approve --review`) and per-run timelines (`treejit explain`)."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, TextIO

from .config import Config
from .store import Store
from .util import now, short

SHORT = 8          # id prefix length printed by the CLI
MIN_PREFIX = 4     # shortest prefix accepted on input

# where ids of each kind live; operator tables count so revoke/unpin work after eviction
ID_SOURCES = {
    "edge": ("SELECT id FROM edges", "SELECT edge FROM approvals", "SELECT edge FROM pins WHERE edge<>''"),
    "node": ("SELECT id FROM nodes", "SELECT node FROM pins", "SELECT node FROM approvals WHERE node<>''"),
    "family": ("SELECT id FROM families",),
    "run": ("SELECT id FROM runs",),
}


class IdError(ValueError):
    pass


def resolve_id(store: Store, kind: str, text: str) -> str:
    """An exact id, or a unique prefix of at least MIN_PREFIX characters."""
    ids: set[str] = set()
    for sql in ID_SOURCES[kind]:
        ids.update(r[0] for r in store.q(sql) if r[0])
    if text in ids:
        return text
    if len(text) < MIN_PREFIX:
        raise IdError(f"unknown {kind} {text!r} (prefixes need at least {MIN_PREFIX} characters)")
    hits = sorted(i for i in ids if i.startswith(text))
    if not hits:
        raise IdError(f"unknown {kind} {text!r}")
    if len(hits) > 1:
        shown = ", ".join(hits[:5]) + (", ..." if len(hits) > 5 else "")
        raise IdError(f"ambiguous {kind} prefix {text!r}: matches {len(hits)} ({shown})")
    return hits[0]


def compact_call(tool: str, args: Any, n: int = 100) -> str:
    """Bash(git push origin main) / send_email(to="a@b.c")."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return short(f"{tool}({args})", n)
    if isinstance(args, dict) and len(args) == 1 and isinstance(next(iter(args.values())), str):
        return short(f"{tool}({next(iter(args.values()))})", n)
    if isinstance(args, dict):
        inner = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
        return short(f"{tool}({inner})", n)
    return short(f"{tool}({args!r})", n)


# ------------------------------------------------------------------ pending approvals

POLICY_BLOCKS = ("needs_approval", "commit_point_needs_evidence")


def pending(store: Store, cfg: Config, family: str | None = None) -> list[dict]:
    """Promoted edges held back only by policy, grouped by edge (one entry per edge,
    with every node where approving it would let replay cross it)."""
    where, args = ("AND ne.family=?", (family,)) if family else ("", ())
    rows = store.q(
        "SELECT ne.*, e.tool, e.label, n.kind, n.depth FROM node_edges ne JOIN edges e ON e.id=ne.edge "
        f"LEFT JOIN nodes n ON n.id=ne.node WHERE ne.blocked IN (?,?) {where} "
        "ORDER BY ne.pass_runs DESC, ne.node", (*POLICY_BLOCKS, *args))
    approvals = store.approvals()
    by_edge: dict[str, dict] = {}
    for r in rows:
        eid = r["edge"]
        it = by_edge.get(eid)
        if it is None:
            it = by_edge[eid] = {
                "edge": eid, "family": r["family"], "tool": r["tool"], "label": r["label"],
                "commit_point": False, "approved_everywhere": (eid, "") in approvals or ("*", "") in approvals,
                "pass_runs": 0, "fail_runs": 0, "example": "", "nodes": [],
            }
        commit = bool(r["commit_point"])
        need = max(0, cfg.promote_runs + 1 - r["pass_runs"]) if commit else 0
        it["nodes"].append({
            "node": r["node"], "kind": r["kind"] or "?", "depth": r["depth"], "pass_runs": r["pass_runs"],
            "fail_runs": r["fail_runs"], "blocked": r["blocked"], "approved": (eid, r["node"]) in approvals,
            "needs_more_passing_runs": need,
        })
        it["commit_point"] = it["commit_point"] or commit
        if r["pass_runs"] > it["pass_runs"] or not it["example"]:
            it["pass_runs"], it["fail_runs"] = r["pass_runs"], r["fail_runs"]
            it["example"] = compact_call(r["tool"], r["ref"] or "{}")
    out = list(by_edge.values())
    for it in out:
        it["needs_approval"] = any(n["blocked"] == "needs_approval" for n in it["nodes"])
        it["needs_more_passing_runs"] = min(n["needs_more_passing_runs"] for n in it["nodes"])
        it["nodes"].sort(key=lambda n: (n["kind"] != "r", n["depth"] or 0, -n["pass_runs"], n["node"]))
    # tree order: shallowest root-path position first (macro-only edges last)
    first = lambda it: it["nodes"][0]  # noqa: E731
    out.sort(key=lambda it: (it["family"], first(it)["kind"] != "r", first(it)["depth"] or 0, it["edge"]))
    return out


def _node_where(n: dict) -> str:
    kind = {"r": f"root path, depth {n['depth']}", "g": f"macro, last {n['depth']}"}.get(n["kind"], n["kind"])
    return f"node {n['node'][:SHORT]} ({kind})"


def pending_text(items: list[dict]) -> str:
    if not items:
        return "nothing pending: no promoted edge is waiting on approval"
    out = []
    for it in items:
        flags = ["COMMIT POINT"] if it["commit_point"] else ["write"]
        if not it["needs_approval"]:
            flags.append("approved")
        out.append(f"edge {it['edge'][:SHORT]}  {it['tool']}  [{', '.join(flags)}]  pass={it['pass_runs']} fail={it['fail_runs']}"
                   f"  family={it['family'][:SHORT]}")
        out.append(f"  e.g. {it['example']}")
        if it["needs_more_passing_runs"]:
            k = it["needs_more_passing_runs"]
            out.append(f"  commit point: needs {k} more passing run{'s' if k > 1 else ''} before replay"
                       + (" (even after approval)" if it["needs_approval"] else ""))
        for n in it["nodes"]:
            state = "approved here" if n["approved"] else n["blocked"]
            out.append(f"  at {_node_where(n)}  pass={n['pass_runs']} fail={n['fail_runs']}  {state}")
        if it["needs_approval"]:
            first = next(n for n in it["nodes"] if n["blocked"] == "needs_approval")
            out.append(f"  approve here:       treejit approve {it['edge'][:SHORT]} --node {first['node'][:SHORT]}")
            out.append(f"  approve everywhere: treejit approve {it['edge'][:SHORT]}")
        out.append("")
    return "\n".join(out).rstrip()


def approve(store: Store, edge: str, node: str = "") -> None:
    store.x("INSERT OR REPLACE INTO approvals(edge, node, ts) VALUES(?,?,?)", (edge, node, now()))


def review(store: Store, cfg: Config, family: str | None, inp: TextIO, out: TextIO) -> list[tuple[str, str]]:
    """Walk the approval queue: y = approve at the listed nodes, e = everywhere,
    n/s = leave pending, q = stop. Returns the (edge, node) approvals made."""
    items = [it for it in pending(store, cfg, family) if it["needs_approval"]]
    done: list[tuple[str, str]] = []
    if not items:
        print("nothing pending: no promoted edge is waiting on approval", file=out)
        return done
    for k, it in enumerate(items, 1):
        print(f"[{k}/{len(items)}] " + pending_text([it]), file=out)
        nodes = [n["node"] for n in it["nodes"] if n["blocked"] == "needs_approval"]
        while True:
            print(f"approve {it['tool']}? [y]es at {len(nodes)} node(s) / [e]verywhere / [n]o / [s]kip / [q]uit: ",
                  end="", file=out, flush=True)
            line = inp.readline()
            ans = line.strip().lower()[:1] if line else "q"
            if ans in ("y", "e", "n", "s", "q"):
                break
            print("please answer y, e, n, s or q", file=out)
        if ans == "q":
            print("stopped", file=out)
            break
        if ans == "y":
            for nid in nodes:
                approve(store, it["edge"], nid)
                done.append((it["edge"], nid))
            print(f"-> approved {it['edge'][:SHORT]} at " + ", ".join(n[:SHORT] for n in nodes) + "\n", file=out)
        elif ans == "e":
            approve(store, it["edge"], "")
            done.append((it["edge"], ""))
            print(f"-> approved {it['edge'][:SHORT]} everywhere\n", file=out)
        else:
            print("-> left pending\n", file=out)
    print(f"{len({e for e, _ in done})} edge(s) approved", file=out)
    return done


# ------------------------------------------------------------------ explain


def latest_run(store: Store) -> str | None:
    r = store.q1("SELECT id FROM runs ORDER BY updated DESC LIMIT 1")
    return r["id"] if r else None


def explain(store: Store, run_id: str) -> dict:
    """Per-step timeline of one run, joined from its steps and its logged requests."""
    run = store.run(run_id)
    if run is None:
        raise IdError(f"unknown run {run_id!r}")
    steps = {s["call_id"]: s for s in store.steps(run_id)}
    reqs = store.q("SELECT * FROM requests WHERE run_id=? ORDER BY id", (run_id,))
    rows: list[dict] = []
    seen: set[str] = set()
    tiers: dict[str, int] = defaultdict(int)
    tokens = 0
    for q in reqs:
        tier = q["tier"] or "?"
        tiers[tier] += 1
        tok = sum(q[c] or 0 for c in ("input_tokens", "output_tokens", "cache_read", "cache_write"))
        tokens += tok
        ids = json.loads(q["call_ids"] or "[]")
        note = q["note"] or ""
        model = tier in ("T4", "pass")
        who = ("model (after side exit)" if note.startswith("side_exit") else "model") if model else tier
        parts = note.split("; ")
        if not ids:
            rows.append({"step": None, "request": q["id"], "who": who, "call": "(no tool call: final answer)" if q["status"] in (None, 200)
                         else f"(status {q['status']})", "note": note, "tokens": tok, "obs": ""})
            continue
        for j, cid in enumerate(ids):
            st = steps.get(cid)
            seen.add(cid)
            rows.append({
                "step": st["idx"] if st else None, "request": q["id"], "who": who,
                "call": compact_call(st["tool"], st["args"]) if st else f"(call {cid} not recorded yet)",
                "note": parts[j] if not model and len(parts) == len(ids) else note,
                "tokens": tok if j == 0 else 0, "obs": _first_line(st), "error": bool(st and st["is_error"]),
            })
    for cid, st in steps.items():  # steps whose request isn't in the log (e.g. an older client)
        if cid not in seen:
            rows.append({"step": st["idx"], "request": None, "who": "replay" if st["replayed"] else "?",
                         "call": compact_call(st["tool"], st["args"]), "note": "", "tokens": 0, "obs": _first_line(st),
                         "error": bool(st["is_error"])})
    rows.sort(key=lambda r: (r["request"] is None, r["request"] or 0, r["step"] if r["step"] is not None else 1 << 30))
    n_steps = len(steps)
    replayed = sum(1 for s in steps.values() if s["replayed"])
    return {
        "run": run_id, "family": run["family"], "task": run["task"] or "", "outcome": run["outcome"],
        "reason": run["reason"], "steps": n_steps, "replayed_calls": replayed, "model_calls": tiers.get("T4", 0) + tiers.get("pass", 0),
        "tokens": tokens, "requests_by_tier": dict(tiers), "timeline": rows,
    }


def _first_line(st: Any) -> str:
    if st is None or st["obs"] is None:
        return ""
    text = st["obs"].strip()
    return short(text.splitlines()[0] if text else "", 70)


def explain_text(d: dict) -> str:
    out = [f"run {d['run']}  family={d['family'][:SHORT]}  outcome={d['outcome'] or '-'}"
           + (f" ({d['reason']})" if d["reason"] else ""),
           f"task: {short(d['task'], 100)!r}",
           f"{d['steps']} tool calls, {d['replayed_calls']} replayed; {d['model_calls']} model call(s), {d['tokens']} tokens; "
           "requests by tier: " + " ".join(f"{t}={n}" for t, n in sorted(d["requests_by_tier"].items())),
           ""]
    for r in d["timeline"]:
        idx = f"#{r['step']}" if r["step"] is not None else "  -"
        tok = f" tok={r['tokens']}" if r["tokens"] else ""
        out.append(f"{idx:>4}  {r['who']:<8} {r['call']}{tok}")
        if r["note"]:
            out.append(f"        why: {short(r['note'], 110)}")
        if r["obs"]:
            out.append(f"        {'error' if r.get('error') else 'obs'}: {r['obs']}")
    return "\n".join(out)
