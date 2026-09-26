"""tau-bench suite: real tau-bench tasks, tools and reward, driven through treejit inline mode.

tau-bench (https://github.com/sierra-research/tau-bench) is not on PyPI. Point TAUBENCH_PATH at a
clone (or install it so `import tau_bench` works); its environments need only `pydantic`. Its user
simulator imports `litellm` at module level; when litellm isn't importable a stub module is inserted
(the stub raises if anything calls it; this runner never does).

Pieces
  make_suite(env, split)  tools_info -> Anthropic tools, wiki -> system prompt, tasks, and the
                          environment's data, cached as JSON and restored per task
  OracleAgent(noise)      deterministic agent: a canonical read prefix (authenticate, get_user_details,
                          get_order_details, get_product_details for new items; airline:
                          get_user_details, get_reservation_details) followed by the task's ground-truth
                          actions, then a final answer containing task.outputs. With probability `noise`
                          per write step it *slips* (wrong argument). Slips are a function of
                          (seed, env, split, task index) only, so every mode sees the same mistakes.
                          It answers treejit T2/T3 subcalls with the same policy (sim._match_proposal).
  ClaudeAgent             a real model through the Anthropic SDK (`--agent claude`, needs
                          ANTHROPIC_API_KEY); treejit modes use `jit.wrap(agent)`. Prompt caching is on
                          (system breakpoint + automatic caching) and cache tokens are accounted.
  ScriptedUser            the task instruction is the first user message. Default single-turn: the
                          episode ends at the agent's first text reply. `confirm` answers an agent
                          question with a confirmation (for real models, which ask before writes; see
                          runs send `X-TreeJIT-Episode: conversation` so one task is one episode).
  run_taubench(...)       -> list[runner.TaskResult], scored by tau-bench's own Env.calculate_reward
                          (database hash after the episode == hash after the ground-truth actions, and
                          every expected output appears in a reply).

The airline env works with the same oracle (test split only; tau-bench ships no airline train split).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import re
import sys
import time
import types
from dataclasses import dataclass
from typing import Any, Callable

from .runner import TaskResult, _account, _compacted, _fresh_jit, _side_exits, _snap, record_outcome
from .sim import UsageModel, _history, _match_proposal

WRITES = {
    "retail": {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
               "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
               "return_delivered_order_items", "transfer_to_human_agents"},
    "airline": {"book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages",
                "update_reservation_flights", "update_reservation_passengers", "transfer_to_human_agents"},
}


class TauBenchUnavailable(ImportError):
    pass


def load_taubench(path: str | None = None) -> types.ModuleType:
    """Import tau_bench from `path` / $TAUBENCH_PATH (or site-packages). Raises TauBenchUnavailable."""
    path = path or os.environ.get("TAUBENCH_PATH")
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(os.path.join(path, "tau_bench")):
            raise TauBenchUnavailable(f"no tau_bench package under {path}")
        if path not in sys.path:
            sys.path.insert(0, path)
    if "litellm" not in sys.modules and importlib.util.find_spec("litellm") is None:
        stub = types.ModuleType("litellm")

        def _no_llm(*a: Any, **kw: Any) -> Any:
            raise RuntimeError("litellm stub: the treejit tau-bench runner never calls an LLM user simulator")

        stub.completion = _no_llm  # type: ignore[attr-defined]
        sys.modules["litellm"] = stub
    try:
        import tau_bench
        import tau_bench.envs  # noqa: F401
        import tau_bench.types  # noqa: F401
    except ImportError as e:
        raise TauBenchUnavailable(f"tau-bench is not importable ({e}); set TAUBENCH_PATH to a clone of "
                                  "https://github.com/sierra-research/tau-bench and install pydantic") from e
    return tau_bench


def taubench_available(path: str | None = None) -> bool:
    try:
        load_taubench(path)
        return True
    except TauBenchUnavailable:
        return False


def anthropic_tools(tools_info: list[dict]) -> list[dict]:
    """tau-bench's OpenAI-style function specs -> Anthropic tool definitions."""
    out = []
    for t in tools_info:
        f = t["function"]
        out.append({"name": f["name"], "description": f.get("description", ""),
                    "input_schema": f.get("parameters") or {"type": "object", "properties": {}}})
    return out


_DATA_JSON: dict[str, str] = {}


@dataclass
class TauSuite:
    env_name: str
    split: str
    env: Any
    tasks: list
    tools: list[dict]
    system: str
    data_json: str

    def fresh_data(self) -> dict:
        return json.loads(self.data_json)

    def reset(self, index: int) -> Any:
        """Start task `index` on a fresh copy of the data (Env.reset without the user simulator)."""
        env = self.env
        env.task_index, env.task, env.data, env.actions = index, self.tasks[index], self.fresh_data(), []
        return env.task

    def data(self) -> dict:
        return self.env.data


def make_suite(env_name: str = "retail", split: str = "test", path: str | None = None) -> TauSuite:
    load_taubench(path)
    from tau_bench.envs import get_env

    env = get_env(env_name, user_strategy="human", user_model="none", task_split=split, task_index=0)
    if env_name not in _DATA_JSON:
        _DATA_JSON[env_name] = json.dumps(env.data)
    suite = TauSuite(env_name, split, env, list(env.tasks), anthropic_tools(env.tools_info), env.wiki, _DATA_JSON[env_name])
    env.data_load_func = suite.fresh_data  # calculate_reward reloads the data: restore from the cached JSON
    return suite


# ============================================================================ oracle agent


def _product_of_item(data: dict, item_id: str) -> str | None:
    for pid, p in data.get("products", {}).items():
        if item_id in p.get("variants", {}):
            return pid
    return None


def oracle_plan(task: Any, data: dict, env_name: str) -> list[tuple[str, dict]]:
    """Canonical trajectory: authentication + the reads the policy asks for + the ground-truth actions."""
    out: list[tuple[str, dict]] = []

    def add(name: str, args: dict) -> None:
        if (name, args) not in out:
            out.append((name, dict(args)))

    if env_name == "retail":
        u = data["users"][task.user_id]
        if u["email"] in task.instruction:
            add("find_user_id_by_email", {"email": u["email"]})
        else:
            add("find_user_id_by_name_zip", {"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"],
                                             "zip": u["address"]["zip"]})
    add("get_user_details", {"user_id": task.user_id})
    for a in task.actions:
        k = a.kwargs
        if a.name.startswith("find_user_id") or a.name == "respond":
            continue
        if "order_id" in k:
            add("get_order_details", {"order_id": k["order_id"]})
        if "reservation_id" in k:
            add("get_reservation_details", {"reservation_id": k["reservation_id"]})
        for it in k.get("new_item_ids") or []:
            pid = _product_of_item(data, it)
            if pid:
                add("get_product_details", {"product_id": pid})
        add(a.name, k)
    return out


_SWAPS = {"no longer needed": "ordered by mistake", "ordered by mistake": "no longer needed", "yes": "no", "no": "yes",
          "economy": "business", "business": "economy", "basic_economy": "economy"}


def slip_args(name: str, args: dict, rng: random.Random) -> dict:
    """A plausible model mistake on a write: wrong reason, a dropped item, a wrong payment method, ..."""
    a = copy.deepcopy(args)
    if isinstance(a.get("reason"), str) and a["reason"] in _SWAPS:
        a["reason"] = _SWAPS[a["reason"]]
    elif len(a.get("item_ids") or []) > 1:
        i = rng.randrange(len(a["item_ids"]))
        a["item_ids"].pop(i)
        if "new_item_ids" in a:
            a["new_item_ids"].pop(i)
    elif "payment_method_id" in a:
        a["payment_method_id"] = "gift_card_0000000"
    elif "payment_id" in a:
        a["payment_id"] = "gift_card_0000000"
    else:
        for k in sorted(a, reverse=True):  # generic: perturb one string argument (never the id it acts on)
            v = a[k]
            if isinstance(v, str) and not k.endswith("_id"):
                a[k] = _SWAPS.get(v, v + " 2")
                break
            if isinstance(v, int) and not isinstance(v, bool):
                a[k] = v + 1
                break
    return a


def _message(content: list[dict], stop: str, usage: dict, model: str = "oracle") -> dict:
    return {"id": f"msg_{random.getrandbits(48):012x}", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage}


class OracleAgent:
    """Anthropic-Messages-shaped fake model that follows oracle_plan(task), with per-task slips on writes."""

    def __init__(self, suite: TauSuite, seed: int = 0, noise: float = 0.05, payload: str = "none", cache: bool = False) -> None:
        self.suite = suite
        self.seed = seed
        self.noise = noise
        self.rng = random.Random(seed)
        self.usage = UsageModel(payload, cache)
        self.calls = self.small_calls = 0
        self.small_tokens = [0, 0]
        self.plan: list[tuple[str, dict]] = []
        self.slips: set[int] = set()
        self.task: Any = None
        self.slipped: dict[int, list[str]] = {}  # task index -> slipped write names (for reporting)

    def begin(self, index: int, task: Any) -> None:
        self.task, self.index = task, index
        self.plan = oracle_plan(task, self.suite.data(), self.suite.env_name)
        r = random.Random(f"{self.seed}:{self.suite.env_name}:{self.suite.split}:{index}")
        writes = WRITES.get(self.suite.env_name, set())
        self.slips = {i for i, (n, _) in enumerate(self.plan) if n in writes and r.random() < self.noise}
        self.slipped[index] = [self.plan[i][0] for i in sorted(self.slips)]

    def body(self, system: str, tools: list[dict], messages: list[dict]) -> dict:
        return {"model": "oracle", "max_tokens": 1024, "system": system, "tools": tools, "messages": messages}

    def next_action(self, hist: list) -> tuple[str, dict] | None:
        done = [(n, a) for n, a, *_ in hist]
        for i, (n, a) in enumerate(self.plan):
            wrong = slip_args(n, a, random.Random(i)) if i in self.slips else a
            if (n, a) not in done and (n, wrong) not in done:
                return n, wrong
        return None

    def __call__(self, body: dict) -> dict:
        forced = (body.get("tool_choice") or {}).get("name", "")
        msgs = body["messages"]
        if forced.startswith("treejit_"):
            self.small_calls += 1
            try:
                ans = self._subcall_answer(msgs[0]["content"], forced)
            except Exception:  # a confused model: nothing useful
                ans = {"not_this_step": True} if forced == "treejit_fill" else {"choice": 0}
            content = [{"type": "tool_use", "id": f"toolu_{self.rng.randrange(16 ** 20):020x}", "name": forced, "input": ans}]
            chars = len(body.get("system", "")) + len(json.dumps(body.get("tools", []))) + len(json.dumps(msgs))
            usage = {"input_tokens": chars // 4, "output_tokens": len(json.dumps(content)) // 4 + 10}
            self.small_tokens[0] += usage["input_tokens"]
            self.small_tokens[1] += usage["output_tokens"]
            return _message(content, "tool_use", usage)
        self.calls += 1
        act = self.next_action(_history(msgs))
        if act is None:
            outs = list(self.task.outputs) if self.task is not None else []
            content = [{"type": "text", "text": "All done." + (" " + "; ".join(outs) if outs else "")}]
            stop = "end_turn"
        else:
            content = [{"type": "tool_use", "id": f"toolu_{self.rng.randrange(16 ** 20):020x}", "name": act[0], "input": act[1]}]
            stop = "tool_use"
        return _message(content, stop, self.usage.full(body, len(json.dumps(content)) // 4 + 40))

    def _subcall_answer(self, prompt: str, tool: str) -> dict:
        hist = [(st["tool"], st["input"]) for st in (json.loads(m.group(1)) for m in re.finditer(r'<step n="\d+">(.*?)</step>', prompt))]
        act = self.next_action(hist)
        if tool == "treejit_fill":
            props = [(0, json.loads(re.search(r"^Next call: (.*)$", prompt, re.M).group(1)))]
        else:
            props = [(int(m.group(1)), json.loads(m.group(2))) for m in re.finditer(r"^(\d+)\. (\{.*\})  \(used in", prompt, re.M)]
        for n, p in props:
            vals = _match_proposal(p, act)
            if vals is not None:
                return vals if tool == "treejit_fill" else {"choice": n, **{f"o{n}_{k}": v for k, v in vals.items()}}
        return {"not_this_step": True} if tool == "treejit_fill" else {"choice": 0}


# ============================================================================ real model


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    fn = getattr(obj, "model_dump", None)
    if callable(fn):
        return fn(mode="json", exclude_none=True)
    return obj.to_dict()


class _Messages:
    def __init__(self, agent: "ClaudeAgent") -> None:
        self._agent = agent

    def create(self, **kw: Any) -> Any:
        return self._agent._create(**kw)


class ClaudeAgent:
    """A real Anthropic model. SDK-shaped (`.messages.create`) so `jit.wrap(agent)` treats it like an
    `anthropic.Anthropic()` client; it counts full vs small (treejit_* forced-tool) calls for the runner.
    Prompt caching: a breakpoint on the system prompt plus top-level automatic caching."""

    def __init__(self, client: Any = None, model: str = "claude-opus-5", max_tokens: int = 4096, cache: bool = True) -> None:
        if client is None:
            import anthropic  # type: ignore

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.cache = cache
        self.calls = self.small_calls = 0
        self.small_tokens = [0, 0]
        self.messages = _Messages(self)
        self.slipped: dict[int, list[str]] = {}

    def begin(self, index: int, task: Any) -> None:
        self.slipped[index] = []

    def body(self, system: str, tools: list[dict], messages: list[dict]) -> dict:
        sys_block: dict = {"type": "text", "text": system}
        body: dict = {"model": self.model, "max_tokens": self.max_tokens, "tools": tools, "messages": messages}
        if self.cache:
            sys_block["cache_control"] = {"type": "ephemeral"}
            body["cache_control"] = {"type": "ephemeral"}
        body["system"] = [sys_block]
        return body

    def _create(self, **kw: Any) -> Any:
        forced = (kw.get("tool_choice") or {}).get("name", "")
        resp = self.client.messages.create(**kw)
        if forced.startswith("treejit_"):
            u = _to_dict(resp).get("usage") or {}
            self.small_calls += 1
            self.small_tokens[0] += sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            self.small_tokens[1] += int(u.get("output_tokens") or 0)
        else:
            self.calls += 1
        return resp


# ============================================================================ user


class ScriptedUser:
    """single: one user turn (the instruction); the episode ends at the agent's first text reply.
    confirm: also answers up to `max_turns` agent questions with a confirmation."""

    CONFIRM = "Yes, I confirm. Please go ahead."

    def __init__(self, mode: str = "single", max_turns: int = 4) -> None:
        if mode not in ("single", "confirm"):
            raise ValueError("ScriptedUser mode must be single|confirm")
        self.mode, self.max_turns, self.turns = mode, max_turns, 0

    def first(self, task: Any) -> str:
        self.turns = 0
        return task.instruction

    def reply(self, agent_text: str) -> str | None:
        if self.mode == "single" or self.turns >= self.max_turns or "?" not in agent_text:
            return None
        self.turns += 1
        return self.CONFIRM


# ============================================================================ runner


def _clean_content(content: list) -> list[dict]:
    """Assistant content as request blocks (SDK dumps carry extra response-only fields)."""
    out = []
    for b in content:
        b = _to_dict(b) if not isinstance(b, dict) else b
        if b.get("type") == "text":
            out.append({"type": "text", "text": b.get("text", "")})
        elif b.get("type") == "tool_use":
            out.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}})
        else:
            out.append({k: v for k, v in b.items() if v is not None})
    return out


def _task_kind(task: Any, env_name: str) -> str:
    names = [a.name for a in task.actions if a.name in WRITES.get(env_name, set())]
    return "+".join(dict.fromkeys(names)) or "read-only"


def run_taubench(n_tasks: int | None = None, env_name: str = "retail", split: str = "test", mode: str = "treejit",
                 noise: float = 0.05, seed: int = 0, db: str | None = None, start: int = 0, agent: str = "oracle",
                 payload: str = "none", cache: bool = False, rebuild_every: int = 1, max_steps: int = 30,
                 path: str | None = None, user: str = "single", claude_model: str = "claude-opus-5",
                 client: Any = None, progress: Callable[[TaskResult], None] | None = None,
                 **overrides: Any) -> list[TaskResult]:
    suite = make_suite(env_name, split, path)
    from tau_bench.types import RESPOND_ACTION_NAME, Action  # importable only after load_taubench

    end = len(suite.tasks) if n_tasks is None else min(len(suite.tasks), start + n_tasks)
    if agent == "oracle":
        model: Any = OracleAgent(suite, seed, noise, payload, cache)
    elif agent == "claude":
        model = ClaudeAgent(client, claude_model, cache=True)
    else:
        raise ValueError("agent must be oracle|claude")
    jit = None
    if mode == "baseline":
        call = (lambda body, rid: model.messages.create(**body)) if agent == "claude" else (lambda body, rid: model(body))  # noqa: E731
    else:
        jit = _fresh_jit(db, mode, **overrides)
        wrapped = jit.wrap(model, dialect="anthropic")  # ClaudeAgent is SDK-shaped, OracleAgent a plain callable
        if agent == "claude":
            call = lambda body, rid: wrapped.messages.create(**body, extra_headers={"X-TreeJIT-Run": rid, "X-TreeJIT-Episode": "conversation"})  # noqa: E731
        else:
            call = lambda body, rid: wrapped(body, extra_headers={"X-TreeJIT-Run": rid, "X-TreeJIT-Episode": "conversation"})  # noqa: E731
    scripted = ScriptedUser(user)
    env = suite.env
    results = []
    for i in range(start, end):
        task = suite.reset(i)
        model.begin(i, task)
        run_id = f"tau-{env_name}-{split}-{seed}-{i}"
        r = TaskResult(mode, i, env_name, _task_kind(task, env_name), False, "")
        msgs: list[dict] = [{"role": "user", "content": scripted.first(task)}]
        for _ in range(max_steps):
            body = model.body(suite.system, suite.tools, msgs)
            before = _snap(model)
            t0 = time.perf_counter()
            resp = _to_dict(call(body, run_id))
            ms = (time.perf_counter() - t0) * 1000
            _account(r, resp, before, _snap(model), ms)
            content = _clean_content(resp.get("content") or [])
            msgs.append({"role": "assistant", "content": content})
            uses = [b for b in content if b["type"] == "tool_use"]
            if not uses:
                text = "".join(b.get("text", "") for b in content if b["type"] == "text")
                env.actions.append(Action(name=RESPOND_ACTION_NAME, kwargs={"content": text}))
                reply = scripted.reply(text)
                if reply is None:
                    break
                msgs.append({"role": "user", "content": reply})
                continue
            blocks = []
            for u in uses:
                er = env.step(Action(name=u["name"], kwargs=u["input"]))
                blocks.append({"type": "tool_result", "tool_use_id": u["id"], "content": er.observation,
                               "is_error": er.observation.startswith("Error")})
            msgs.append({"role": "user", "content": blocks})
        res = env.calculate_reward()
        r.reward = float(res.reward)
        r.success = r.reward >= 1.0
        info = res.info  # RewardOutputInfo when the task has outputs, else RewardActionInfo
        if r.success:
            r.reason = "ok"
        elif hasattr(info, "r_outputs") and not info.r_outputs:
            r.reason = "missing output"
        else:
            r.reason = "wrong database state"
        if model.slipped.get(i):
            r.reason += " (slip: " + ",".join(model.slipped[i]) + ")"
        if jit is not None:
            r.side_exits = _side_exits(jit, run_id)
            r.compacted_chars = _compacted(jit, run_id)
            record_outcome(jit, run_id, r.success, None if r.success else r.reason, i - start, rebuild_every)
        results.append(r)
        if progress is not None:
            progress(r)
    if jit is not None:
        jit.close()
    return results
