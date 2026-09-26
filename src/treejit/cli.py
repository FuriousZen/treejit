"""treejit CLI: serve, show, runs, outcome, pin, approve, pending, explain, prune, export, build, stats."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import Config
from .engine import TreeJIT
from .operate import IdError, resolve_id
from .util import now, short


def _jit(args: argparse.Namespace) -> TreeJIT:
    cfg = Config.load(args.config)
    if args.db:
        cfg.db = args.db
    jit = TreeJIT(config=cfg)
    if getattr(args, "family", None):
        args.family = _resolve(jit, "family", args.family)
    return jit


def _resolve(jit: TreeJIT, kind: str, text: str) -> str:
    try:
        return resolve_id(jit.store, kind, text)
    except IdError as e:
        raise SystemExit(f"treejit: {e}")


def cmd_serve(a: argparse.Namespace) -> None:
    from .proxy import serve

    jit = _jit(a)
    print(f"treejit proxy on http://{a.host or jit.cfg.host}:{a.port or jit.cfg.port}  db={jit.cfg.db}", file=sys.stderr)
    serve(jit, a.host, a.port)


def cmd_show(a: argparse.Namespace) -> None:
    from .export import show_text

    jit = _jit(a)
    jit.rebuild_dirty()
    print(show_text(jit.store, a.family, ids=a.ids, max_depth=a.depth, macros=not a.no_macros))


def cmd_runs(a: argparse.Namespace) -> None:
    jit = _jit(a)
    rows = jit.store.q("SELECT r.*, (SELECT COUNT(*) FROM requests q WHERE q.run_id=r.id AND q.tier='T4') fc, "
                       "(SELECT COUNT(*) FROM steps s WHERE s.run_id=r.id AND s.replayed=1) rp "
                       "FROM runs r ORDER BY updated DESC LIMIT ?", (a.limit,))
    for r in rows:
        print(f"{r['id']:<28} {r['outcome'] or '-':<5} steps={r['n_steps']:<3} replayed={r['rp']:<3} model_calls={r['fc']:<3} "
              f"family={r['family']}  {short(r['task'] or '', 60)!r}")


def cmd_outcome(a: argparse.Namespace) -> None:
    jit = _jit(a)
    ids = jit.outcome(a.run_id, a.result, a.reason)
    print(json.dumps({"updated": ids}))


def cmd_pin(a: argparse.Namespace) -> None:
    jit = _jit(a)
    a.node = _resolve(jit, "node", a.node)
    a.edge = _resolve(jit, "edge", a.edge) if a.edge else ""
    if a.unpin:
        jit.store.x("DELETE FROM pins WHERE node=? AND edge=?", (a.node, a.edge))
    else:
        jit.store.x("INSERT OR REPLACE INTO pins(node, edge, ts) VALUES(?,?,?)", (a.node, a.edge, now()))
    _mark_dirty(jit)
    print("unpinned" if a.unpin else "pinned", a.node, a.edge or "(whole node)")


def cmd_approve(a: argparse.Namespace) -> None:
    from . import operate

    jit = _jit(a)
    if a.review:
        jit.rebuild_dirty()
        if operate.review(jit.store, jit.cfg, a.family, sys.stdin, sys.stdout):
            _mark_dirty(jit)
        return
    if not a.edge:
        raise SystemExit("treejit: approve needs an edge id (or '*', or --review)")
    not_commit = getattr(a, "not_commit", False)
    if a.edge == "*" and not_commit:
        raise SystemExit("treejit: --not-commit is per edge: name the edge (approve '*' never implies it)")
    if a.edge != "*":
        a.edge = _resolve(jit, "edge", a.edge)
    if a.node:
        a.node = _resolve(jit, "node", a.node)
    if a.revoke:
        if not not_commit:
            jit.store.x("DELETE FROM approvals WHERE edge=? AND node=?", (a.edge, a.node or ""))
        if not_commit or not a.node:
            jit.store.x("DELETE FROM not_commit WHERE edge=?", (a.edge,))
    else:
        operate.approve(jit.store, a.edge, a.node or "", not_commit=not_commit)
    _mark_dirty(jit)
    what = " (not a commit point)" if not_commit else ""
    print("revoked" if a.revoke else "approved", a.edge + what, "at", a.node or "every node")


def cmd_revoke(a: argparse.Namespace) -> None:
    a.revoke, a.review = True, False
    cmd_approve(a)


def cmd_pending(a: argparse.Namespace) -> None:
    from . import operate

    jit = _jit(a)
    jit.rebuild_dirty()
    items = operate.pending(jit.store, jit.cfg, a.family)
    print(json.dumps(items, indent=2) if a.json else operate.pending_text(items))


def cmd_explain(a: argparse.Namespace) -> None:
    from . import operate

    jit = _jit(a)
    run_id = operate.latest_run(jit.store) if a.run_id == "latest" else _resolve(jit, "run", a.run_id)
    if run_id is None:
        raise SystemExit("treejit: no runs recorded yet")
    d = operate.explain(jit.store, run_id)
    print(json.dumps(d, indent=2) if a.json else operate.explain_text(d))


def cmd_prune(a: argparse.Namespace) -> None:
    jit = _jit(a)
    days = a.days if a.days is not None else jit.cfg.evict_days
    min_hits = a.min_hits if a.min_hits is not None else jit.cfg.evict_min_hits
    cutoff = now() - days * 86400
    pins = {n for n, _ in jit.store.pins()}
    rows = jit.store.q("SELECT n.id, n.family, n.last_seen, COALESCE(h.hits,0) hits, COALESCE(h.last_hit,0) last_hit "
                       "FROM nodes n LEFT JOIN hits h ON h.node=n.id")
    victims = [r for r in rows if r["id"] not in pins and r["hits"] < min_hits
               and max(r["last_seen"] or 0, r["last_hit"]) < cutoff and r["id"] != _root(r["family"])]
    if not a.dry_run:
        t = now()
        jit.store.xmany("INSERT OR REPLACE INTO evictions(node, ts) VALUES(?,?)", [(r["id"], t) for r in victims])
        for fam in {r["family"] for r in victims}:
            jit.rebuild(fam)
    print(f"{'would evict' if a.dry_run else 'evicted'} {len(victims)} node(s) (idle > {days}d, hits < {min_hits})")


def _root(family: str) -> str:
    from .tree import node_id

    return node_id(family, "r", ())


def _mark_dirty(jit: TreeJIT) -> None:
    jit.store.x("UPDATE families SET dirty=1")
    jit.rebuild()


def cmd_export(a: argparse.Namespace) -> None:
    from . import export

    jit = _jit(a)
    if a.format == "skills":
        out = a.out or "skills"
        files = export.export_skills(jit.store, out, a.family)
        print("\n".join(files) if files else "no hot paths yet")
        return
    text = export.export_html(jit.store, a.family) if a.format == "html" else export.mermaid(jit.store, a.family)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(a.out)
    else:
        sys.stdout.write(text)


def cmd_build(a: argparse.Namespace) -> None:
    jit = _jit(a)
    for r in jit.rebuild(a.family):
        print(json.dumps(r))


def cmd_stats(a: argparse.Namespace) -> None:
    jit = _jit(a)
    s = jit.stats(a.family)
    total = sum(v["tool_calls"] for v in s.values()) or 1
    # every tier but the model (T4) and pass-through requests is served by treejit (T0, T1, T2, T3)
    replayed = sum(v["tool_calls"] for k, v in s.items() if k not in ("T4", "pass"))
    small = sum(v["requests"] for k, v in s.items() if k in ("T2", "T3"))
    print(json.dumps(s, indent=2))
    print(f"tool calls served by replay: {replayed}/{total} ({100 * replayed / total:.1f}%)")
    print(f"T2/T3 subcalls: {small} (T2 {s.get('T2', {}).get('requests', 0)}, T3 {s.get('T3', {}).get('requests', 0)}); "
          f"full model calls (T4): {s.get('T4', {}).get('requests', 0)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="treejit", description="Inference proxy with memory.")
    p.add_argument("--db", help="SQLite file (default: treejit.db or $TREEJIT_DB)")
    p.add_argument("--config", help="treejit.toml")
    # --db / --config are accepted after the subcommand too (`treejit explain latest --db x.db`)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, **kw: Any) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], **kw)

    s = add("serve", help="run the proxy")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)

    s = add("show", help="print the tree")
    s.add_argument("--family")
    s.add_argument("--ids", action="store_true", help="show short node/edge ids (for pin/approve)")
    s.add_argument("--depth", type=int, default=40)
    s.add_argument("--no-macros", action="store_true")
    s.set_defaults(fn=cmd_show)

    s = add("runs", help="list recent runs")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_runs)

    s = add("outcome", help="report a run's verifier result")
    s.add_argument("run_id", help="run id, or 'latest'")
    s.add_argument("result", choices=["pass", "fail", "error"])
    s.add_argument("--reason")
    s.set_defaults(fn=cmd_outcome)

    s = add("pin", help="pin a node (or one edge at it): promote and protect from eviction")
    s.add_argument("node")
    s.add_argument("edge", nargs="?")
    s.add_argument("--unpin", action="store_true")
    s.set_defaults(fn=cmd_pin)

    s = add("approve", help="allow replay to cross a non-read-only edge / commit point")
    s.add_argument("edge", nargs="?", help="edge id or unique prefix (>= 4 chars), or '*' for every edge")
    s.add_argument("--node", help="only at this node (default: everywhere)")
    s.add_argument("--revoke", action="store_true")
    s.add_argument("--review", action="store_true", help="walk pending edges interactively (y/e/n/s/q on stdin)")
    s.add_argument("--family", help="with --review: only this family")
    s.add_argument("--not-commit", action="store_true",
                   help="also declare this edge not a commit point (its effects are local: a test script, a local "
                        "make target); per edge only, never implied by approve '*'")
    s.set_defaults(fn=cmd_approve)

    s = add("revoke", help="undo an approval (same as approve --revoke)")
    s.add_argument("edge")
    s.add_argument("--node")
    s.add_argument("--not-commit", action="store_true", help="only withdraw the not-a-commit-point declaration")
    s.set_defaults(fn=cmd_revoke)

    s = add("pending", help="list promoted edges waiting on operator approval")
    s.add_argument("--family")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_pending)

    s = add("explain", help="per-step timeline of one run: who decided each step and why")
    s.add_argument("run_id", help="run id, unique prefix, or 'latest'")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_explain)

    s = add("prune", help="evict cold nodes")
    s.add_argument("--days", type=float)
    s.add_argument("--min-hits", type=int)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_prune)

    s = add("export", help="export the tree")
    s.add_argument("--format", choices=["html", "mermaid", "skills"], default="html")
    s.add_argument("--out")
    s.add_argument("--family")
    s.set_defaults(fn=cmd_export)

    s = add("build", help="rebuild trees from the trace log")
    s.add_argument("--family")
    s.set_defaults(fn=cmd_build)

    s = add("stats", help="replay vs frontier counts")
    s.add_argument("--family")
    s.set_defaults(fn=cmd_stats)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
