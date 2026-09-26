"""Self-contained HTML learning-curve report (inline SVG, no dependencies)."""

from __future__ import annotations

import html
import json
from typing import Any

MODE_LABELS = {
    "baseline": "Plain agent",
    "treejit": "treejit (read-only allowlist)",
    "treejit+ok": "treejit (edges approved)",
    "treejit+ok+compact": "treejit (edges approved, compaction)",
}


def _label(mode: str) -> str:
    base = mode.replace("@proxy", "")
    return MODE_LABELS.get(base, base) + (" via proxy" if mode.endswith("@proxy") else "")


def report_html(series: dict[str, list[dict]], summary: dict[str, Any], meta: dict[str, Any], window: int = 10) -> str:
    """series: mode -> per-task rows (from runner.to_rows)."""
    data = {
        "window": window,
        "modes": [{"key": m, "label": _label(m),
                   "calls": [r["model_calls"] for r in rows],
                   "tokens": [r["tokens"] for r in rows],
                   "served": [(r["replayed_calls"], r["tool_calls"]) for r in rows],
                   "wall": [r["wall_ms"] for r in rows],
                   "success": [1 if r["success"] else 0 for r in rows]} for m, rows in series.items()],
    }
    tiles = []
    base = summary.get("baseline", {}).get("last", {})
    for m, s in summary.items():
        if m == "baseline":
            continue
        last = s["last"]
        calls_cut = 1 - last["calls_per_task"] / base["calls_per_task"] if base.get("calls_per_task") else 0
        tok_cut = 1 - last["tokens_per_task"] / base["tokens_per_task"] if base.get("tokens_per_task") else 0
        tiles.append(f"""<div class="tile"><div class="tile-k">{html.escape(_label(m))} · last {last['n']} tasks</div>
<div class="tile-row"><div><div class="big">{last['served_pct']:.0f}%</div><div class="sub">tool calls served by replay</div></div>
<div><div class="big">−{calls_cut * 100:.0f}%</div><div class="sub">model calls / task ({last['calls_per_task']:.2f} vs {base.get('calls_per_task', 0):.2f})</div></div>
<div><div class="big">−{tok_cut * 100:.0f}%</div><div class="sub">tokens / task</div></div>
<div><div class="big">{last['success_pct']:.0f}%</div><div class="sub">success (plain agent {base.get('success_pct', 0):.0f}%)</div></div></div></div>""")
    rows = []
    for m, s in summary.items():
        for wname in ("first", "mid", "last"):
            w = s[wname]
            rows.append(f"<tr><td>{html.escape(_label(m))}</td><td>{html.escape(w['range'])}</td>"
                        f"<td>{w['calls_per_task']:.2f}</td><td>{w['tokens_per_task']:,.0f}</td><td>{w['served_pct']:.0f}%</td>"
                        f"<td>{w['success_pct']:.0f}%</td><td>{w['wall_s_per_task']:.1f}</td><td>{w['side_exits']}</td></tr>")
    meta_line = " · ".join(f"{k}={v}" for k, v in meta.items())
    return TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__TILES__", "".join(tiles)) \
        .replace("__ROWS__", "".join(rows)).replace("__META__", html.escape(meta_line))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>treejit learning curve</title>
<style>
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#8a5cf5; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19;
  --ink:#ffffff; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9d74f7; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10); --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9d74f7; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } .lede { color:var(--ink2); margin:0 0 20px; max-width:70ch; }
.tiles { display:grid; gap:12px; margin-bottom:20px; }
.tile { background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px 16px; }
.tile-k { color:var(--ink2); font-size:12.5px; margin-bottom:8px; }
.tile-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.big { font-size:28px; font-weight:600; } .sub { color:var(--ink2); font-size:12.5px; }
.legend { display:flex; flex-wrap:wrap; gap:14px; margin:4px 0 12px; color:var(--ink2); font-size:13px; }
.legend span::before { content:""; display:inline-block; width:14px; height:3px; border-radius:2px; background:var(--c); margin-right:6px; vertical-align:middle; }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }
figure { margin:0; background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:12px 12px 6px; position:relative; }
figcaption { font-weight:600; font-size:13.5px; } figcaption small { display:block; font-weight:400; color:var(--ink2); }
svg { display:block; width:100%; height:auto; overflow:visible; }
.tick { fill:var(--muted); font-size:10.5px; font-variant-numeric:tabular-nums; }
.dl { font-size:10.5px; fill:var(--ink2); }
.tip { position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--ring); border-radius:8px;
  padding:6px 8px; font-size:12px; box-shadow:0 4px 14px rgba(0,0,0,.12); display:none; white-space:nowrap; z-index:2; }
.tip b { font-variant-numeric:tabular-nums; } .tip .k { color:var(--ink2); }
.tip i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
details { margin-top:18px; background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:10px 14px; }
summary { cursor:pointer; font-weight:600; }
.tablewrap { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:13px; margin-top:8px; }
th,td { text-align:left; padding:6px 8px; border-top:1px solid var(--grid); font-variant-numeric:tabular-nums; white-space:nowrap; }
th { color:var(--ink2); font-weight:500; }
.meta { color:var(--muted); font-size:12px; margin-top:16px; }
</style></head>
<body><main>
<h1>treejit learning curve</h1>
<p class="lede">The same synthetic task stream (coding + retail workflows) run by a simulated agent with treejit off and on.
Curves are rolling means over the previous tasks; the tree starts empty.</p>
<div class="tiles">__TILES__</div>
<div class="legend" id="legend"></div>
<div class="charts" id="charts"></div>
<details><summary>Table view</summary><div class="tablewrap"><table>
<tr><th>mode</th><th>tasks</th><th>model calls / task</th><th>tokens / task</th><th>served by replay</th><th>success</th><th>sim. wall-clock s / task</th><th>side exits</th></tr>
__ROWS__</table></div></details>
<p class="meta">__META__</p>
</main>
<script>
const D = __DATA__;
const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
function roll(xs, w) { const out = []; let s = 0; for (let i = 0; i < xs.length; i++) { s += xs[i]; if (i >= w) s -= xs[i - w]; out.push(s / Math.min(i + 1, w)); } return out; }
function rollRatio(pairs, w) { const out = []; for (let i = 0; i < pairs.length; i++) { let a = 0, b = 0; for (let j = Math.max(0, i - w + 1); j <= i; j++) { a += pairs[j][0]; b += pairs[j][1]; } out.push(b ? 100 * a / b : 0); } return out; }
const METRICS = [
  { key: "calls", title: "Model calls per task", sub: "lower is better", f: m => roll(m.calls, D.window), fmt: v => v.toFixed(2) },
  { key: "tokens", title: "Tokens per task", sub: "prompt + completion sent to the model", f: m => roll(m.tokens, D.window), fmt: v => Math.round(v).toLocaleString() },
  { key: "served", title: "Tool calls served by replay", sub: "% of the task's tool calls, no model call", f: m => rollRatio(m.served, D.window), fmt: v => v.toFixed(0) + "%" },
];
const legend = document.getElementById("legend");
D.modes.forEach((m, i) => { const s = document.createElement("span"); s.style.setProperty("--c", COLORS[i]); s.textContent = m.label; legend.appendChild(s); });
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function nice(max) { if (max <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(max))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= max) return m * p; return 10 * p; }
const charts = document.getElementById("charts");
METRICS.forEach(M => {
  const fig = document.createElement("figure");
  const cap = document.createElement("figcaption"); cap.textContent = M.title;
  const sm = document.createElement("small"); sm.textContent = M.sub + " · rolling mean of " + D.window + " tasks"; cap.appendChild(sm); fig.appendChild(cap);
  const W = 340, H = 200, L = 44, R = 12, T = 12, B = 26;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": M.title }, null);
  fig.appendChild(svg);
  const ys = D.modes.map(M.f);
  const n = Math.max(...D.modes.map(m => m.calls.length));
  const ymax = M.key === "served" ? 100 : nice(Math.max(...ys.flat()) * 1.05);
  const x = i => L + (W - L - R) * (n <= 1 ? 0 : i / (n - 1));
  const y = v => T + (H - T - B) * (1 - v / ymax);
  for (let k = 0; k <= 4; k++) {
    const v = ymax * k / 4;
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: k ? "var(--grid)" : "var(--axis)", "stroke-width": 1 }, svg);
    const t = el("text", { x: L - 6, y: y(v) + 3.5, "text-anchor": "end", class: "tick" }, svg); t.textContent = M.key === "served" ? v + "%" : (v >= 1000 ? (v / 1000) + "k" : +v.toFixed(2));
  }
  const step = n > 150 ? 50 : n > 60 ? 20 : 10;
  for (let i = 0; i < n; i += step) { const t = el("text", { x: x(i), y: H - 8, "text-anchor": "middle", class: "tick" }, svg); t.textContent = i + 1; }
  const tl = el("text", { x: W - R, y: H - 8, "text-anchor": "end", class: "tick" }, svg); tl.textContent = "task #";
  ys.forEach((vals, si) => {
    if (M.key === "served" && D.modes[si].key === "baseline") return;
    const d = vals.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
    el("path", { d, fill: "none", stroke: COLORS[si], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  });
  const hair = el("line", { y1: T, y2: H - B, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = ys.map((_, si) => el("circle", { r: 4, fill: COLORS[si], stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg));
  const tip = document.createElement("div"); tip.className = "tip"; fig.appendChild(tip);
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
  function show(evt) {
    const r = svg.getBoundingClientRect(); const px = (evt.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - L) / (W - L - R) * (n - 1))));
    hair.setAttribute("x1", x(i)); hair.setAttribute("x2", x(i)); hair.setAttribute("visibility", "visible");
    tip.replaceChildren();
    const head = document.createElement("div"); head.className = "k"; head.textContent = "task " + (i + 1); tip.appendChild(head);
    ys.forEach((vals, si) => {
      if (M.key === "served" && D.modes[si].key === "baseline") { dots[si].setAttribute("visibility", "hidden"); return; }
      dots[si].setAttribute("cx", x(i)); dots[si].setAttribute("cy", y(vals[i])); dots[si].setAttribute("visibility", "visible");
      const row = document.createElement("div"); const sw = document.createElement("i"); sw.style.background = COLORS[si];
      const b = document.createElement("b"); b.textContent = M.fmt(vals[i]) + " ";
      const k = document.createElement("span"); k.className = "k"; k.textContent = D.modes[si].label;
      row.append(sw, b, k); tip.appendChild(row);
    });
    tip.style.display = "block";
    const fx = svg.offsetLeft + (x(i) / W) * r.width;
    let left = fx > fig.clientWidth / 2 ? fx - tip.offsetWidth - 12 : fx + 12;
    left = Math.max(4, Math.min(fig.clientWidth - tip.offsetWidth - 4, left));
    tip.style.left = left + "px"; tip.style.top = (svg.offsetTop + 6) + "px";
  }
  function hide() { hair.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); tip.style.display = "none"; }
  hit.addEventListener("pointermove", show); hit.addEventListener("pointerleave", hide);
  charts.appendChild(fig);
});
</script>
</body></html>
"""
