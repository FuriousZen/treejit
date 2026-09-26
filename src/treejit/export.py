"""Views of the tree: text (`treejit show`), self-contained HTML, Mermaid, and SKILL.md per hot path."""

from __future__ import annotations

import html
import json
import os
import re
from collections import defaultdict
from typing import Any

from .bindings import rule_label
from .features import pred_label
from .store import Store
from .tree import END, node_id
from .util import short

TIER_COLORS = {"hot": "#d9480f", "live": "#e8a33d", "warm": "#9fb83a", "cold": "#8a94a6", "tomb": "#343a40"}
TIER_ORDER = {"hot": 0, "live": 1, "warm": 2, "cold": 3, "tomb": 4}


def family_ids(store: Store, family: str | None = None) -> list[str]:
    if family:
        rows = store.q("SELECT id FROM families WHERE id LIKE ?", (family + "%",))
    else:
        rows = store.q("SELECT f.id FROM families f WHERE EXISTS(SELECT 1 FROM runs r WHERE r.family=f.id) ORDER BY f.created")
    return [r["id"] for r in rows]


def tree_data(store: Store, family: str, max_depth: int = 40) -> dict[str, Any]:
    edges = {r["id"]: r for r in store.q("SELECT * FROM edges WHERE family=?", (family,))}
    nodes = {r["id"]: r for r in store.q("SELECT * FROM nodes WHERE family=?", (family,))}
    kids: dict[str, list] = defaultdict(list)
    for r in store.q("SELECT * FROM node_edges WHERE family=?", (family,)):
        kids[r["node"]].append(r)
    hits = {r["node"]: r["hits"] for r in store.q("SELECT node, hits FROM hits")}

    def item(ne: Any, child: str | None, ctx: tuple, depth: int) -> dict:
        bindings = json.loads(ne["bindings"] or "{}")
        return {
            "node": ne["node"], "edge": ne["edge"], "label": edges[ne["edge"]]["label"] if ne["edge"] in edges else ne["edge"],
            "tool": edges[ne["edge"]]["tool"] if ne["edge"] in edges else "?",
            "tier": ne["tier"], "blocked": ne["blocked"] or "", "pass": ne["pass_runs"], "fail": ne["fail_runs"], "n": ne["n"],
            "conf": ne["conf"], "purity": ne["purity"], "blamed": ne["blamed"],
            "bindings": {k: rule_label(v) for k, v in bindings.items()},
            "guard": json.loads(ne["guard"] or "{}"), "post": json.loads(ne["post"] or "{}"),
            "reasons": json.loads(ne["reasons"] or "[]"), "commit_point": bool(ne["commit_point"]),
            "savings": ne["savings"], "score": ne["score"], "ref": json.loads(ne["ref"] or "{}"),
            "hits": hits.get(ne["node"], 0),
            "children": walk(child, ctx, depth + 1) if child and child in nodes and depth < max_depth else [],
        }

    def walk(nid: str, ctx: tuple, depth: int) -> list[dict]:
        out = []
        for ne in sorted(kids.get(nid, []), key=lambda r: (TIER_ORDER.get(r["tier"], 9), -r["pass_runs"], -r["n"])):
            cctx = ctx + (ne["edge"],)
            out.append(item(ne, node_id(family, "r", cctx), cctx, depth))
        return out

    root = node_id(family, "r", ())
    macros = []
    for nid, n in nodes.items():
        if n["kind"] != "g":
            continue
        good = [ne for ne in kids.get(nid, []) if ne["tier"] in ("hot", "live", "tomb")]
        if not good or n["n_runs"] < 2:
            continue
        ctx = json.loads(n["ctx"])
        macros.append({
            "node": nid, "ctx": [edges[e]["label"] if e in edges else e for e in ctx], "runs": n["n_runs"],
            "children": [item(ne, None, (), 0) for ne in sorted(good, key=lambda r: -r["pass_runs"])],
        })
    macros.sort(key=lambda m: (-len(m["ctx"]), -m["runs"]))
    stumps = {nid: json.loads(n["stump"]) for nid, n in nodes.items() if n["stump"]}
    fam = store.q1("SELECT * FROM families WHERE id=?", (family,))
    runs = store.q1("SELECT COUNT(*) n, SUM(outcome='pass') p, SUM(outcome='fail') f FROM runs WHERE family=?", (family,))
    return {
        "family": family, "prefix": (fam["prefix"] if fam else "")[:200], "runs": runs["n"], "pass": runs["p"] or 0,
        "fail": runs["f"] or 0, "root": walk(root, (), 0), "macros": macros, "stumps": stumps,
        "root_stump": stumps.get(root),
    }


def _stats(it: dict) -> str:
    s = f"pass={it['pass']} fail={it['fail']} conf={it['conf']:.2f}"
    if it["bindings"]:
        s += " " + " ".join(f"{k}←{v}" for k, v in it["bindings"].items())
    if it["guard"]:
        s += " guard{" + ",".join(f"{k}={v}" for k, v in it["guard"].items()) + "}"
    if it["reasons"]:
        s += f" reason={it['reasons'][0]!r}"
    return s


def show_text(store: Store, family: str | None = None, ids: bool = False, max_depth: int = 40, macros: bool = True) -> str:
    out = []
    for fam in family_ids(store, family):
        d = tree_data(store, fam, max_depth)
        out.append(f"family {fam}  runs={d['runs']} pass={d['pass']} fail={d['fail']}  prefix={short(d['prefix'], 60)!r}")
        if d["root_stump"]:
            out.append("  root branch: " + _stump_text(d["root_stump"], d["root"]))
        _text_walk(d["root"], "  ", out, ids)
        if macros and d["macros"]:
            out.append("  macros (last-k-edge contexts):")
            for m in d["macros"][:20]:
                out.append(f"    [{' → '.join(short(c, 30) for c in m['ctx'])}]")
                for it in m["children"]:
                    out.append(f"      ↳ [{_tier_tag(it)}] {it['label']}  {_stats(it)}" + (f"  node={it['node'][:8]}" if ids else ""))
        out.append("")
    return "\n".join(out) if out else "(empty: no recorded runs yet)"


def _stump_text(dl: dict, items: list[dict]) -> str:
    labels = {it["edge"]: short(it["label"], 30) for it in items} | {END: "END (final answer)"}
    parts = [f"if {pred_label(r['pred'])} → {labels.get(r['edge'], r['edge'])}" for r in dl["rules"]]
    parts.append("else → model")
    return "; ".join(parts)


def _text_walk(items: list[dict], indent: str, out: list[str], ids: bool) -> None:
    for i, it in enumerate(items):
        last = i == len(items) - 1
        branch = "└─ " if last else "├─ "
        extra = f"  node={it['node'][:8]} edge={it['edge'][:8]}" if ids else ""
        out.append(f"{indent}{branch}[{_tier_tag(it)}] {it['label']}  {_stats(it)}{extra}")
        _text_walk(it["children"], indent + ("   " if last else "│  "), out, ids)


def _tier_tag(it: dict) -> str:
    """HOT / LIVE:holes / LIVE:needs_approval ...; the reason only where it isn't implied by the tier."""
    tag = f"{it['tier'].upper():4}"
    return f"{tag.strip()}:{it['blocked']}" if it["tier"] == "live" and it["blocked"] else tag


# ------------------------------------------------------------------ mermaid


def mermaid(store: Store, family: str | None = None) -> str:
    lines = ["flowchart TD"]
    for t, c in TIER_COLORS.items():
        lines.append(f"  classDef {t} fill:{c},color:#fff,stroke:#222")
    counter = [0]

    def nid() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def esc(s: str) -> str:
        return s.replace('"', "'").replace("\n", " ")

    for fam in family_ids(store, family):
        d = tree_data(store, fam)
        root = nid()
        lines.append(f'  {root}(["{fam} · {d["runs"]} runs"])')

        def rec(parent: str, items: list[dict]) -> None:
            for it in items:
                me = nid()
                lines.append(f'  {me}["{esc(short(it["label"], 48))}<br/>pass {it["pass"]} · fail {it["fail"]}"]:::{it["tier"]}')
                lines.append(f"  {parent} --> {me}")
                rec(me, it["children"])

        rec(root, d["root"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ html


def export_html(store: Store, family: str | None = None) -> str:
    fams = [tree_data(store, f) for f in family_ids(store, family)]

    def li(it: dict) -> str:
        c = TIER_COLORS.get(it["tier"], "#888")
        meta = html.escape(_stats(it))
        body = f'<span class="tier" style="background:{c}">{it["tier"]}</span> <code>{html.escape(it["label"])}</code> <span class="meta">{meta}</span>'
        if it["children"]:
            inner = "".join(li(x) for x in it["children"])
            return f'<li><details open><summary>{body}</summary><ul>{inner}</ul></details></li>'
        return f"<li>{body}</li>"

    sections = []
    for d in fams:
        tree = "".join(li(x) for x in d["root"]) or "<li><em>no edges yet</em></li>"
        macro = ""
        if d["macros"]:
            rows = "".join(
                f"<tr><td>{html.escape(' → '.join(m['ctx']))}</td><td>{''.join(li(c) for c in m['children'])}</td></tr>"
                for m in d["macros"][:40])
            macro = f"<h3>Macros</h3><table><tr><th>after</th><th>next</th></tr>{rows}</table>"
        sections.append(
            f"<section><h2>Family <code>{d['family']}</code></h2>"
            f"<p class=\"meta\">{d['runs']} runs · {d['pass']} pass · {d['fail']} fail</p>"
            f"<ul class=\"tree\">{tree}</ul>{macro}</section>")
    legend = " ".join(f'<span class="tier" style="background:{c}">{t}</span>' for t, c in TIER_COLORS.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>treejit tree</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1f23; --muted:#6b7280; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15171a; --fg:#e8e6e1; --muted:#9aa0a6; --line:#333; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.5 ui-sans-serif,system-ui,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1100px; margin:auto; }}
code {{ font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace; }}
.tier {{ color:#fff; border-radius:4px; padding:0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
.meta {{ color:var(--muted); font-size:12px; }}
ul.tree, ul.tree ul {{ list-style:none; padding-left:18px; border-left:1px solid var(--line); margin:2px 0; }}
li {{ margin:3px 0; overflow-wrap:anywhere; }}
summary {{ cursor:pointer; }}
table {{ border-collapse:collapse; width:100%; }} td,th {{ border-top:1px solid var(--line); padding:6px; vertical-align:top; text-align:left; }}
</style></head><body><main>
<h1>treejit execution tree</h1><p>{legend}</p>
{''.join(sections) or '<p>No recorded runs yet.</p>'}
</main></body></html>
"""


# ------------------------------------------------------------------ SKILL.md


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "path"


def hot_paths(d: dict, min_len: int = 2) -> list[list[dict]]:
    paths = []

    def rec(items: list[dict], path: list[dict]) -> None:
        good = [it for it in items if it["tier"] in ("hot", "live")]
        if not good:
            if len(path) >= min_len:
                paths.append(path)
            return
        for it in good:
            rec(it["children"], path + [it])

    rec(d["root"], [])
    return paths


def export_skills(store: Store, out_dir: str, family: str | None = None) -> list[str]:
    written = []
    for fam in family_ids(store, family):
        d = tree_data(store, fam)
        for path in hot_paths(d):
            first = path[0]
            name = f"treejit-{_slug(first['tool'] + '-' + path[-1]['label'])}-{path[-1]['node'][:6]}"
            runs = min(it["pass"] for it in path)
            when = ""
            if d["root_stump"]:
                for r in d["root_stump"]["rules"]:
                    if r["edge"] == first["edge"]:
                        when = f" Use when {pred_label(r['pred'])}."
            desc = f"Proven {len(path)}-step procedure ({runs}+ successful runs): " + " → ".join(it["tool"] for it in path) + "." + when
            lines = ["---", f"name: {name}", f"description: {json.dumps(desc)}", "---", "",
                     f"# {short(first['label'], 60)} … {short(path[-1]['label'], 60)}", "",
                     "This sequence was learned by treejit from successful runs. Follow it step by step;",
                     "if a step's result does not match what is expected, stop following it and reason normally.", ""]
            for i, it in enumerate(path, 1):
                lines.append(f"{i}. `{it['label']}`")
                for slot, rule in it["bindings"].items():
                    lines.append(f"   - `{slot}` ← {rule}")
                if it["post"]:
                    lines.append("   - expect: " + ", ".join(f"{k}={v}" for k, v in it["post"].items()))
                if it["commit_point"]:
                    lines.append("   - **commit point**: irreversible; confirm before running.")
                if it["reasons"]:
                    lines.append(f"   - known failure: {it['reasons'][0]}")
            path_dir = os.path.join(out_dir, name)
            os.makedirs(path_dir, exist_ok=True)
            fp = os.path.join(path_dir, "SKILL.md")
            with open(fp, "w") as f:
                f.write("\n".join(lines) + "\n")
            written.append(fp)
    return written
