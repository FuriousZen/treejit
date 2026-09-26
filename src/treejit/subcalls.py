"""T2/T3 subcalls: one small, structured model call instead of a full frontier call.

  T3 fill    the next call is known except for some values; the model supplies only those
  T2 choose  the model picks among the node's known children (or 0 = "something else")

The prompt is short and deterministic: the task, the calls made so far (arguments
truncated), the last few tool results, and the call(s) with placeholders for holes.
The full conversation is never sent. How the answer is structured depends on the dialect
(`Config.subcall_format`):

  Anthropic   structured outputs (default, "json_schema"): `output_config.format` with a JSON
              schema; the answer is a JSON text block. Forced tool use (`tool_choice` "tool"/"any")
              returns a 400 on Claude Fable 5.1, Mythos 5.1 and Opus 5.5, so it is never sent to a
              current model. "tool_auto" is the fallback shape: one strict tool, `tool_choice: auto`
              and an explicit instruction (no call = failed subcall). "tool" (forced) is kept only
              for legacy models without structured outputs (claude-3*, claude-2*, claude-instant).
              `thinking` is never sent (it can't be disabled on the always-thinking models); on
              models that take `output_config.effort` the subcall asks for `Config.subcall_effort`
              ("low"), and on models that think by default max_tokens leaves room for the thinking.
  OpenAI      a forced function call (`tool_choice: {"type": "function", ...}`), Chat Completions
              or Responses shape.

Any answer that doesn't parse or doesn't fit the schema falls back to the T4 forward.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Config
from .dialects import thinking_default_on
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

_FILL_BASE = (
    "You assist a tool-using agent. Its next tool call is already decided except for the <placeholders>. "
    "Give the value each placeholder should take, exactly as the agent would write it (plain text, no shell quoting). "
    "If this call is not the right next step, set not_this_step to true."
)
_CHOOSE_BASE = (
    "You assist a tool-using agent. Pick its next tool call from the numbered options, given the task and the "
    "latest tool results; answer 0 if none of them is right. If the option you pick has <placeholders>, also give "
    "their values in the fields o<N>_<placeholder> (plain text, no shell quoting)."
)
FILL_SYSTEM = f"{_FILL_BASE} Answer only by calling {FILL_TOOL}."
CHOOSE_SYSTEM = f"{_CHOOSE_BASE} Answer only by calling {CHOOSE_TOOL}."
# structured outputs (Anthropic default): the answer is the response text itself
FILL_SYSTEM_JSON = f"{_FILL_BASE} Answer with the JSON object only ({FILL_TOOL})."
CHOOSE_SYSTEM_JSON = f"{_CHOOSE_BASE} Answer with the JSON object only ({CHOOSE_TOOL})."
# fallback without forced tool use: say it, and check a call was made (no call -> T4)
FILL_SYSTEM_AUTO = f"{_FILL_BASE} You must answer by calling the {FILL_TOOL} tool exactly once, with no other text."
CHOOSE_SYSTEM_AUTO = f"{_CHOOSE_BASE} You must answer by calling the {CHOOSE_TOOL} tool exactly once, with no other text."
_SYSTEMS = {FILL_SYSTEM: FILL_TOOL, CHOOSE_SYSTEM: CHOOSE_TOOL, FILL_SYSTEM_JSON: FILL_TOOL,
            CHOOSE_SYSTEM_JSON: CHOOSE_TOOL, FILL_SYSTEM_AUTO: FILL_TOOL, CHOOSE_SYSTEM_AUTO: CHOOSE_TOOL}

FORMATS = ("auto", "json_schema", "tool_auto", "tool")
# Anthropic models without structured outputs: the forced tool call is the only structured shape there
_LEGACY_TOOL = re.compile(r"claude-(?:instant|2|3)(?:[-.]|$)")
# models that take output_config.effort (Opus 4.5+, Sonnet 4.6+, Fable, Mythos); others 400 on it
_EFFORT = re.compile(r"claude-(?:opus-4-[5-9]|opus-[5-9]|sonnet-4-[6-9]|sonnet-[5-9]|fable|mythos)")
THINKING_MAX_TOKENS = 4096  # floor for max_tokens where thinking is on by default (it counts toward the cap)


def anthropic_format(model: str, cfg: Config) -> str:
    """The subcall shape for an Anthropic model: json_schema | tool_auto | tool."""
    fmt = cfg.subcall_format if cfg.subcall_format in FORMATS else "auto"
    if fmt == "auto":
        return "tool" if _LEGACY_TOOL.search(model or "") else "json_schema"
    return fmt


def subcall_tool(body: dict) -> str:
    """The treejit subcall a request body is (FILL_TOOL / CHOOSE_TOOL), or "" for any other request.
    For fake models and harness-side accounting: works for every shape `build` produces."""
    tc = body.get("tool_choice")
    name = ""
    if isinstance(tc, dict):
        name = tc.get("name") or (tc.get("function") or {}).get("name") or ""
    if name in (FILL_TOOL, CHOOSE_TOOL):
        return name
    system = body.get("system")
    if isinstance(system, str) and system in _SYSTEMS:
        return _SYSTEMS[system]
    for m in (body.get("messages") or body.get("input") or [])[:1]:
        if isinstance(m, dict) and m.get("role") in ("system", "developer") and m.get("content") in _SYSTEMS:
            return _SYSTEMS[m["content"]]
    instr = body.get("instructions")
    return _SYSTEMS.get(instr, "") if isinstance(instr, str) else ""


def answer_content(body: dict, answer: dict, tool_use_id: str = "toolu_treejit_sub") -> list[dict]:
    """Anthropic response content answering subcall `body` with `answer` (for scripted models):
    a JSON text block for structured outputs, a tool_use block for the tool shapes."""
    name = subcall_tool(body)
    if isinstance(body.get("output_config"), dict) and body["output_config"].get("format"):
        return [{"type": "text", "text": json.dumps(answer)}]
    return [{"type": "tool_use", "id": tool_use_id, "name": name, "input": answer}]


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
    if dialect == "responses":
        return {"model": model, "max_output_tokens": max(16, cfg.subcall_max_tokens), "instructions": system,
                "input": [{"role": "user", "content": user}], "store": False, "parallel_tool_calls": False,
                "tools": [{"type": "function", "name": name, "description": desc, "parameters": schema}],
                "tool_choice": {"type": "function", "name": name}}
    fmt = anthropic_format(model, cfg)
    if fmt == "tool":  # legacy models only: forced tool use 400s on Fable 5.1 / Mythos 5.1 / Opus 5.5
        return {"model": model, "max_tokens": cfg.subcall_max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}],
                "tools": [{"name": name, "description": desc, "input_schema": schema}],
                "tool_choice": {"type": "tool", "name": name}}
    strict = strict_schema(schema)
    out: dict = {"model": model, "max_tokens": cfg.subcall_max_tokens,
                 "messages": [{"role": "user", "content": user}]}
    if thinking_default_on(model):
        out["max_tokens"] = max(cfg.subcall_max_tokens, THINKING_MAX_TOKENS)
    oc: dict = {}
    if fmt == "tool_auto":
        out["system"] = FILL_SYSTEM_AUTO if sub.kind == "fill" else CHOOSE_SYSTEM_AUTO
        out["tools"] = [{"name": name, "description": desc, "input_schema": strict, "strict": True}]
        out["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    else:
        out["system"] = FILL_SYSTEM_JSON if sub.kind == "fill" else CHOOSE_SYSTEM_JSON
        oc["format"] = {"type": "json_schema", "schema": strict}
    if cfg.subcall_effort and _EFFORT.search(model or ""):
        oc["effort"] = cfg.subcall_effort
    if oc:
        out["output_config"] = oc
    return out


def strict_schema(schema: dict) -> dict:
    """The subcall schema in the subset structured outputs / strict tools accept: every object has
    `additionalProperties: false`, and numeric bounds become an enum (minimum/maximum are rejected)."""
    props = {}
    for k, p in schema["properties"].items():
        p = dict(p)
        lo, hi = p.pop("minimum", None), p.pop("maximum", None)
        if p.get("type") == "integer" and lo is not None and hi is not None:
            p["enum"] = list(range(int(lo), int(hi) + 1))
        props[k] = p
    return {"type": "object", "properties": props, "required": list(schema.get("required") or []),
            "additionalProperties": False}


def _json_text(text: str) -> Any:
    """The JSON object in a structured-output text block (tolerating code fences and stray prose)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        pass
    a, b = t.find("{"), t.rfind("}")
    if 0 <= a < b:
        try:
            return json.loads(t[a : b + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def tool_input(dialect: str, sub: Subcall, body: dict) -> dict | None:
    """The subcall's answer (the tool call's arguments, or the structured-output JSON), or None."""
    name = FILL_TOOL if sub.kind == "fill" else CHOOSE_TOOL
    if dialect == "openai":
        for ch in body.get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") == name:
                    return _args(fn.get("arguments"))
        return None
    if dialect == "responses":
        for item in body.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("name") == name:
                return _args(item.get("arguments"))
        return None
    texts = []
    for b in body.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use" and b.get("name") == name:
            return b.get("input") if isinstance(b.get("input"), dict) else None
        if b.get("type") == "text":
            texts.append(b.get("text") or "")
    if not texts:
        return None  # no call, no text (tool_auto without a call, a refusal, thinking only): T4
    out = _json_text("".join(texts))
    return out if isinstance(out, dict) else None


def _args(raw: Any) -> dict | None:
    try:
        args = json.loads(raw or "") if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    return args if isinstance(args, dict) else None


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
