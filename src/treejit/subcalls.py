"""T2/T3 subcalls: one small, forced-tool model call instead of a full frontier call.

  T3 fill    the next call is known except for some values; the model supplies only those
  T2 choose  the model picks among the node's known children (or 0 = "something else")

The prompt is short and deterministic: the task, the calls made so far (arguments
truncated), the last few tool results, and the call(s) with placeholders for holes.
The full conversation is never sent. Output is structured by forcing a tool call
(Anthropic `tool_choice: {"type": "tool"}`, OpenAI `tool_choice: {"type": "function"}`).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Config
from .model import NormRequest, ToolCall
from .replay import Option, Subcall
from .templates import Val, call_slots, cook, render

FILL_TOOL = "treejit_fill"
CHOOSE_TOOL = "treejit_choose"
N_OBS = 3             # tool results shown (most recent)
OBS_CHARS = 2000
TASK_CHARS = 4000
ARG_CHARS = 200
MAX_STEPS = 30        # earlier calls shown (most recent)

FILL_SYSTEM = (
    "You assist a tool-using agent. Its next tool call is already decided except for the <placeholders>. "
    "Give the value each placeholder should take, exactly as the agent would write it (plain text, no shell quoting). "
    f"If this call is not the right next step, set not_this_step to true. Answer only by calling {FILL_TOOL}."
)
CHOOSE_SYSTEM = (
    "You assist a tool-using agent. Pick its next tool call from the numbered options, given the task and the "
    "latest tool results; answer 0 if none of them is right. If the option you pick has <placeholders>, also give "
    f"their values in the fields o<N>_<placeholder> (plain text, no shell quoting). Answer only by calling {CHOOSE_TOOL}."
)


def prop_names(slots: list[str]) -> dict[str, str]:
    """Tool-schema-safe property name per slot (`command#0` -> `command_0`)."""
    out: dict[str, str] = {}
    for s in slots:
        base = re.sub(r"[^A-Za-z0-9_]", "_", s)[:48] or "value"
        name, n = base, 1
        while name in out.values() or name == "not_this_step":
            n += 1
            name = f"{base}_{n}"
        out[s] = name
    return out


def placeholder_args(opt: Option, names: dict[str, str]) -> dict | None:
    values = dict(opt.values)
    for slot in opt.holes:
        ph = f"<{names[slot]}>"
        values[slot] = Val(ph, raw=ph, js=ph)
    try:
        return render(opt.edge.template, opt.ne.ref, values)
    except (KeyError, ValueError):
        return None


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


def _clip(v: Any) -> Any:
    if isinstance(v, str):
        return _short(v, ARG_CHARS)
    if isinstance(v, dict):
        return {k: _clip(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clip(x) for x in v[:20]]
    return v


def _call_json(tool: str, args: dict) -> str:
    return json.dumps({"tool": tool, "input": _clip(args)}, ensure_ascii=False)


def _context(req: NormRequest) -> str:
    ep = req.episode
    lines = ["<task>", _short(ep.task, TASK_CHARS), "</task>", "<steps>"]
    first = max(0, len(ep.steps) - MAX_STEPS)
    for i in range(first, len(ep.steps)):
        st = ep.steps[i]
        lines.append(f'<step n="{i + 1}">{_call_json(st.call.name, st.call.args)}</step>')
        if i >= len(ep.steps) - N_OBS and st.obs is not None:
            err = "true" if st.obs.is_error else "false"
            lines += [f'<result n="{i + 1}" error="{err}">', _short(st.obs.text, OBS_CHARS), "</result>"]
    lines.append("</steps>")
    return "\n".join(lines)


def _example(opt: Option, slot: str) -> str:
    try:
        vals = call_slots(opt.edge.template, ToolCall("", opt.edge.tool, opt.ne.ref)) or {}
    except (KeyError, ValueError):
        vals = {}
    v = vals.get(slot)
    return _short(v.cooked, ARG_CHARS) if v is not None else ""


def _hole_schema(opt: Option, slot: str) -> dict:
    ref = opt.ne.ref.get(slot) if "#" not in slot else None
    desc = f"value for {slot} in {opt.edge.tool}"
    if ref is not None and not isinstance(ref, str):
        desc += f" (JSON {type(ref).__name__})"
    ex = _example(opt, slot)
    if ex:
        desc += f"; an earlier run used: {ex}"
    return {"type": "string", "description": desc}


def build(dialect: str, sub: Subcall, req: NormRequest, cfg: Config) -> dict | None:
    """Request body for the subcall, in the client's dialect (always non-streaming)."""
    ctx = _context(req)
    if sub.kind == "fill":
        opt = sub.options[0]
        names = prop_names(opt.holes)
        shown = placeholder_args(opt, names)
        if shown is None:
            return None
        props = {names[s]: _hole_schema(opt, s) for s in opt.holes}
        props["not_this_step"] = {"type": "boolean", "description": "true if this call is not the right next step"}
        schema = {"type": "object", "properties": props, "required": [names[s] for s in opt.holes]}
        user = f"{ctx}\nNext call: {_call_json(opt.edge.tool, shown)}"
        system, name, desc = FILL_SYSTEM, FILL_TOOL, "Fill in the placeholders of the next tool call."
    else:
        props = {"choice": {"type": "integer", "minimum": 0, "maximum": len(sub.options),
                            "description": "number of the option to run next, or 0 for something else"}}
        lines = ["Options:"]
        for n, opt in enumerate(sub.options, 1):
            names = prop_names(opt.holes)
            shown = opt.args if not opt.holes else placeholder_args(opt, names)
            if shown is None:
                return None
            runs = opt.ne.pass_runs
            lines.append(f"{n}. {_call_json(opt.edge.tool, shown)}  (used in {runs} earlier successful run{'s' * (runs != 1)})")
            for s in opt.holes:
                hs = _hole_schema(opt, s)
                props[f"o{n}_{names[s]}"] = dict(hs, description=f"only if choice={n}: {hs['description']}")
        lines.append("0. something else")
        schema = {"type": "object", "properties": props, "required": ["choice"]}
        user = ctx + "\n" + "\n".join(lines)
        system, name, desc = CHOOSE_SYSTEM, CHOOSE_TOOL, "Choose the agent's next tool call."
    model = cfg.small_model or req.model
    if dialect == "openai":
        key = "max_completion_tokens" if "max_completion_tokens" in req.raw else "max_tokens"
        return {"model": model, key: cfg.subcall_max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "tools": [{"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}],
                "tool_choice": {"type": "function", "function": {"name": name}}}
    return {"model": model, "max_tokens": cfg.subcall_max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": name, "description": desc, "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": name}}


def tool_input(dialect: str, sub: Subcall, body: dict) -> dict | None:
    """The forced tool call's arguments from the upstream response, or None."""
    name = FILL_TOOL if sub.kind == "fill" else CHOOSE_TOOL
    if dialect == "openai":
        for ch in body.get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") == name:
                    try:
                        args = json.loads(fn.get("arguments") or "")
                    except (json.JSONDecodeError, TypeError):
                        return None
                    return args if isinstance(args, dict) else None
        return None
    for b in body.get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == name:
            return b.get("input") if isinstance(b.get("input"), dict) else None
    return None


def _values(opt: Option, out: dict, prefix: str = "") -> dict[str, Val] | None:
    names = prop_names(opt.holes)
    vals: dict[str, Val] = {}
    for slot in opt.holes:
        v = out.get(prefix + names[slot])
        ref = opt.ne.ref.get(slot) if "#" not in slot else None
        if isinstance(v, str):
            vals[slot] = Val(v)
        elif v is not None and ref is not None and not isinstance(ref, str):
            vals[slot] = Val(cook(v), js=v)
        else:
            return None
    return vals


def resolve(sub: Subcall, out: dict | None) -> tuple[Option | None, dict[str, Val] | None, str]:
    """Interpret the subcall's output: (option, hole values, why-not)."""
    if out is None:
        return None, None, "no_tool_call"
    if sub.kind == "fill":
        if out.get("not_this_step") is True:
            return None, None, "declined"
        opt = sub.options[0]
        vals = _values(opt, out)
        return (opt, vals, "") if vals is not None else (None, None, "missing_values")
    raw = out.get("choice")
    try:
        n = int(raw) if not isinstance(raw, bool) else -1
    except (TypeError, ValueError):
        return None, None, "bad_choice"
    if n == 0:
        return None, None, "chose_new"
    if not 1 <= n <= len(sub.options):
        return None, None, "bad_choice"
    opt = sub.options[n - 1]
    if not opt.holes:
        return opt, {}, ""
    vals = _values(opt, out, f"o{n}_")
    return (opt, vals, "") if vals is not None else (None, None, "missing_values")
