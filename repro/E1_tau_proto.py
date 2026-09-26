"""E1 prototype: tau-bench retail tasks through treejit inline mode, no LLM.

An oracle-with-noise agent follows each task's ground-truth write actions, preceded by the
read steps a real agent makes (authenticate, get_user_details, get_order_details, get_product_details),
emits Anthropic-format tool calls, and ends with a final text answer containing task.outputs.
tau-bench's own Env.step executes tools; Env.calculate_reward (data-hash + outputs) scores.

Historical: the prototype behind PLAN E1/B1, written against the base commit 19a53f5. Superseded by
`python -m treejit_bench --suite taubench` (bench/src/treejit_bench/taubench.py). Subcall detection was
updated to treejit.subcalls.subcall_tool/answer_content so it still runs on the current code.

usage: python E1_tau_proto.py [--split test|train] [--n 5] [--modes baseline,treejit,treejit+ok] [--noise 0.05]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "E1_tau-bench"), os.path.join(HERE, "E1_deps"), os.path.join(HERE, "..", "src"), os.path.join(HERE, "..", "bench", "src")]
_ll = types.ModuleType("litellm")          # tau_bench.envs.user imports litellm at module level; never called here
_ll.completion = lambda **kw: (_ for _ in ()).throw(RuntimeError("no LLM in this run"))
sys.modules["litellm"] = _ll

from tau_bench.envs.retail.data import load_data  # noqa: E402
from tau_bench.envs.retail.env import MockRetailDomainEnv  # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME  # noqa: E402
from treejit import TreeJIT  # noqa: E402
from treejit.subcalls import answer_content, subcall_tool  # noqa: E402
from treejit.util import now  # noqa: E402
from treejit_bench.sim import _history, _match_proposal  # noqa: E402

WRITES = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items",
          "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items", "transfer_to_human_agents"}
_DATA = load_data()
_DATA_JSON = json.dumps(_DATA)


def fresh_data():
    return json.loads(_DATA_JSON)


def product_of_item(item_id: str) -> str | None:
    for pid, p in _DATA["products"].items():
        if item_id in p.get("variants", {}):
            return pid
    return None


def plan_for(task) -> list[tuple[str, dict]]:
    """Canonical trajectory: auth + reads the policy asks for + ground-truth actions."""
    u = _DATA["users"][task.user_id]
    out: list[tuple[str, dict]] = []
    if u["email"] in task.instruction:
        out.append(("find_user_id_by_email", {"email": u["email"]}))
    else:
        out.append(("find_user_id_by_name_zip", {"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"],
                                                 "zip": u["address"]["zip"]}))
    out.append(("get_user_details", {"user_id": task.user_id}))
    for a in task.actions:
        k = a.kwargs
        if a.name.startswith("find_user_id"):
            continue
        if "order_id" in k and ("get_order_details", {"order_id": k["order_id"]}) not in out:
            out.append(("get_order_details", {"order_id": k["order_id"]}))
        for it in k.get("new_item_ids") or []:
            pid = product_of_item(it)
            if pid and ("get_product_details", {"product_id": pid}) not in out:
                out.append(("get_product_details", {"product_id": pid}))
        if (a.name, k) not in out:
            out.append((a.name, dict(k)))
    return out


def mistake(name: str, args: dict, rng: random.Random) -> dict:
    a = copy.deepcopy(args)
    if name == "cancel_pending_order":
        a["reason"] = "ordered by mistake" if a.get("reason") == "no longer needed" else "no longer needed"
    elif name in ("return_delivered_order_items", "exchange_delivered_order_items") and len(a.get("item_ids", [])) > 1:
        i = rng.randrange(len(a["item_ids"]))
        a["item_ids"].pop(i)
        if "new_item_ids" in a:
            a["new_item_ids"].pop(i)
    elif "payment_method_id" in a:
        a["payment_method_id"] = "gift_card_0000000"
    return a


class OracleModel:
    """Anthropic-Messages-shaped fake agent that follows plan_for(task), with noise on write steps."""

    def __init__(self, tasks, seed: int = 0, noise: float = 0.05):
        self.by_instr = {t.instruction: t for t in tasks}
        self.plans = {t.instruction: plan_for(t) for t in tasks}
        self.rng = random.Random(seed)
        self.seed = seed
        self.noise = noise
        self.calls = self.small_calls = 0
        self.small_tokens = [0, 0]
        self.slips: dict[str, set[int]] = {}   # instruction -> plan idx the model will get wrong (decided once per task)

    def next_action(self, task_text: str, hist: list) -> tuple[str, dict] | None:
        plan = self.plans[task_text]
        # slips are a function of (seed, task) only, so every mode sees the same model mistakes
        r = random.Random(f"{self.seed}:{task_text}")
        slips = self.slips.setdefault(task_text, {i for i, (n, _) in enumerate(plan) if n in WRITES and r.random() < self.noise})
        done = [(n, a) for n, a, *_ in hist]
        for i, (n, a) in enumerate(plan):
            wrong = mistake(n, a, random.Random(i)) if i in slips else a
            if (n, a) not in done and (n, wrong) not in done:
                return n, wrong
        return None

    def __call__(self, body: dict) -> dict:
        forced = subcall_tool(body)  # T2/T3 subcall: structured outputs (output_config.format) or a forced tool
        msgs = body["messages"]
        if forced:
            self.small_calls += 1
            ans = self._sub(msgs[0]["content"], forced)
            content = answer_content(body, ans, f"toolu_{self.rng.randrange(16 ** 20):020x}")
            chars = (len(body.get("system", "")) + len(json.dumps(body.get("tools", []))) + len(json.dumps(msgs))
                     + (len(json.dumps(body["output_config"])) if "output_config" in body else 0))
            usage = {"input_tokens": chars // 4, "output_tokens": len(json.dumps(content)) // 4 + 10}
            self.small_tokens[0] += usage["input_tokens"]
            self.small_tokens[1] += usage["output_tokens"]
            return {"id": "msg_x", "type": "message", "role": "assistant", "model": "oracle", "content": content,
                    "stop_reason": "tool_use" if content[0]["type"] == "tool_use" else "end_turn",
                    "stop_sequence": None, "usage": usage}
        self.calls += 1
        task_text = msgs[0]["content"] if isinstance(msgs[0]["content"], str) else msgs[0]["content"][0]["text"]
        act = self.next_action(task_text, _history(msgs))
        if act is None:
            outs = self.by_instr[task_text].outputs
            content = [{"type": "text", "text": "All done." + (" " + "; ".join(outs) if outs else "")}]
            stop = "end_turn"
        else:
            content = [{"type": "tool_use", "id": f"toolu_{self.rng.randrange(16 ** 20):020x}", "name": act[0], "input": act[1]}]
            stop = "tool_use"
        chars = len(body.get("system", "")) + len(json.dumps(body.get("tools", []))) + len(json.dumps(msgs))
        return {"id": "msg_y", "type": "message", "role": "assistant", "model": "oracle", "content": content, "stop_reason": stop,
                "stop_sequence": None, "usage": {"input_tokens": chars // 4, "output_tokens": len(json.dumps(content)) // 4 + 40}}

    def _sub(self, prompt: str, tool: str) -> dict:
        try:
            task = re.search(r"<task>\n(.*?)\n</task>", prompt, re.S).group(1)
            hist = [(st["tool"], st["input"]) for st in (json.loads(m.group(1)) for m in re.finditer(r'<step n="\d+">(.*?)</step>', prompt))]
            act = self.next_action(task, hist)
            if tool == "treejit_fill":
                props = [(0, json.loads(re.search(r"^Next call: (.*)$", prompt, re.M).group(1)))]
            else:
                props = [(int(m.group(1)), json.loads(m.group(2))) for m in re.finditer(r"^(\d+)\. (\{.*\})  \(used in", prompt, re.M)]
            for n, p in props:
                vals = _match_proposal(p, act)
                if vals is not None:
                    return vals if tool == "treejit_fill" else {"choice": n, **{f"o{n}_{k}": v for k, v in vals.items()}}
        except Exception:
            pass
        return {"not_this_step": True} if tool == "treejit_fill" else {"choice": 0}


def anthropic_tools(env) -> list[dict]:
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in env.tools_info]


def run(split: str, n: int, mode: str, noise: float, seed: int, db: str | None, start: int = 0):
    env = MockRetailDomainEnv(user_strategy="human", task_split=split, task_index=0)
    env.data_load_func = fresh_data
    tasks = env.tasks[start:start + n]
    model = OracleModel(env.tasks, seed + 1, noise)
    tools, system = anthropic_tools(env), env.wiki
    if mode == "baseline":
        call = lambda body, rid: model(body)  # noqa: E731
        jit = None
    else:
        if db and os.path.exists(db):
            os.remove(db)
        jit = TreeJIT(db or ":memory:")
        if "ok" in mode.split("+"):
            jit.store.x("INSERT OR REPLACE INTO approvals(edge, node, ts) VALUES('*', '', ?)", (now(),))
        w = jit.wrap(model, dialect="anthropic")
        call = lambda body, rid: w(body, extra_headers={"X-TreeJIT-Run": rid})  # noqa: E731
    rows = []
    for i, task in enumerate(tasks, start):
        env.task_index, env.task, env.data, env.actions = i, task, fresh_data(), []
        msgs = [{"role": "user", "content": task.instruction}]
        r = {"i": i, "full": 0, "small": 0, "tok": 0, "tools": 0, "replayed": 0, "tiers": ""}
        for _ in range(30):
            b0, s0, st0 = model.calls, model.small_calls, sum(model.small_tokens)
            resp = call({"model": "oracle", "max_tokens": 1024, "system": system, "tools": tools, "messages": msgs}, f"tau-{split}-{i}")
            uses = [c for c in resp["content"] if c["type"] == "tool_use"]
            r["tools"] += len(uses)
            r["small"] += model.small_calls - s0
            r["tok"] += sum(model.small_tokens) - st0
            if model.calls > b0:
                r["full"] += 1
                r["tok"] += resp["usage"]["input_tokens"] + resp["usage"]["output_tokens"]
                r["tiers"] += "M"
            else:
                r["replayed"] += len(uses)
                r["tiers"] += "S" if model.small_calls > s0 else "R"
            msgs.append({"role": "assistant", "content": resp["content"]})
            if not uses:
                txt = "".join(c.get("text", "") for c in resp["content"])
                env.actions.append(Action(name=RESPOND_ACTION_NAME, kwargs={"content": txt}))
                break
            results = []
            for u in uses:
                er = env.step(Action(name=u["name"], kwargs=u["input"]))
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": er.observation,
                                "is_error": er.observation.startswith("Error")})
            msgs.append({"role": "user", "content": results})
        rew = env.calculate_reward()
        r["reward"] = rew.reward
        if jit is not None:
            jit.outcome(f"tau-{split}-{i}", rew.reward >= 1.0)
        r["slip"] = bool(model.slips.get(task.instruction))
        rows.append(r)
    if jit is not None:
        jit.close()
    return rows


def summarize(mode, rows):
    k = len(rows) or 1
    tools = sum(r["tools"] for r in rows) or 1
    return (f"{mode:<12} n={len(rows):<4} full/task={sum(r['full'] for r in rows) / k:5.2f} small/task={sum(r['small'] for r in rows) / k:4.2f} "
            f"tok/task={sum(r['tok'] for r in rows) / k:8,.0f} served={100 * sum(r['replayed'] for r in rows) / tools:4.0f}% "
            f"reward={sum(r['reward'] for r in rows) / k:.2f} slip_tasks={sum(r.get('slip', 0) for r in rows)} "
            f"fail_without_slip={[r['i'] for r in rows if r['reward'] < 1 and not r.get('slip')]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--modes", default="baseline,treejit,treejit+ok")
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=0, help="also summarize the last W tasks")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    for mode in a.modes.split(","):
        t0 = time.time()
        rows = run(a.split, a.n, mode, a.noise, a.seed, os.path.join(HERE, f"E1_{mode}_{a.split}.db") if mode != "baseline" else None, a.start)
        print(summarize(mode, rows), f"({time.time() - t0:.1f}s)")
        if a.window:
            print("  last", summarize(mode, rows[-a.window:]))
        if a.verbose:
            for r in rows:
                print("   ", r)
