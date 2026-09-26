"""Generate a JSON-heavy (tau-bench-like) retail trace log with the bench's SimModel.

The stock retail family returns small JSON; tau-bench's get_user_details / get_order_details return nested
objects of ~1-2k chars (address, payment methods, several orders, item dicts with options, fulfillments,
payment history). This wraps RetailEnv.run to pad the observations the same way, keeping the keys the
SimModel reads (`status`, `items`, `payment_methods[0].id`, `user_id`).

usage: P1_json_gen.py <db> <n_tasks> [rebuild_every=10]
"""
import json
import random
import sys
import time

import treejit.engine as E
from treejit_bench import sim
from treejit_bench.runner import run_suite

db, n = sys.argv[1], int(sys.argv[2])
every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
_run = sim.RetailEnv.run
PRODUCTS = ["T-Shirt", "Water Bottle", "Desk Lamp", "Backpack", "Headphones", "Office Chair", "Sneakers", "Tablet"]


def _rng(env):
    return random.Random(env.user_id + env.order_id)


def _address(r):
    return {"address1": f"{r.randint(1, 999)} {r.choice(['Main', 'Oak', 'Pine', 'Elm'])} Street",
            "address2": f"Suite {r.randint(100, 999)}", "city": r.choice(["Austin", "Denver", "Boston", "Seattle"]),
            "country": "USA", "state": r.choice(["TX", "CO", "MA", "WA"]), "zip": f"{r.randint(10000, 99999)}"}


def run(self, name, args):
    out, err = _run(self, name, args)
    if err:
        return out, err
    r = _rng(self)
    if name == "get_user_details":
        d = json.loads(out)
        others = [f"#W{r.randint(1000000, 9999999)}" for _ in range(r.randint(1, 3))]
        d.update({"name": {"first_name": self.user_id.split("_")[0].title(), "last_name": self.user_id.split("_")[1].title()},
                  "address": _address(r),
                  "orders": d["orders"] + others})
        d["payment_methods"] = d["payment_methods"] + [
            {"id": f"gift_card_{r.randint(1000000, 9999999)}", "source": "gift_card", "balance": r.randint(1, 300)},
            {"id": f"paypal_{r.randint(1000000, 9999999)}", "source": "paypal"}]
        return json.dumps(d), False
    if name == "get_order_details":
        d = json.loads(out)
        d["address"] = _address(r)
        d["item_details"] = [{"name": r.choice(PRODUCTS), "product_id": str(r.randint(10 ** 9, 10 ** 10 - 1)), "item_id": it,
                              "price": round(r.uniform(5, 500), 2),
                              "options": {"color": r.choice(["red", "blue", "black"]), "size": r.choice(["S", "M", "L"]),
                                          "material": r.choice(["cotton", "polyester", "metal"])}} for it in self.items]
        d["fulfillments"] = [{"tracking_id": [str(r.randint(10 ** 11, 10 ** 12 - 1))], "item_ids": list(self.items)}] \
            if self.status == "delivered" else []
        d["payment_history"] = [{"transaction_type": "payment", "amount": d["total"], "payment_method_id": self.payment}]
        return json.dumps(d), False
    return out, err


sim.RetailEnv.run = run

# tau-bench's oracle also looks up every product before a write: get_product_details returns the product with all its
# variants (~1-2k chars). The product id comes from the order JSON.
sim.RETAIL_TOOLS.append({"name": "get_product_details", "description": "Get product details.",
                         "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}},
                                          "required": ["product_id"]}})
_retail = sim.SimModel._retail
_run2 = sim.RetailEnv.run


def run_p(self, name, args):
    if name != "get_product_details":
        return _run2(self, name, args)
    r = random.Random(args.get("product_id", ""))
    variants = {}
    for _ in range(r.randint(6, 12)):
        k = str(r.randint(10 ** 9, 10 ** 10 - 1))
        variants[k] = {"item_id": k, "options": {"color": r.choice(["red", "blue", "black", "white"]),
                                                 "size": r.choice(["S", "M", "L", "XL"]),
                                                 "material": r.choice(["cotton", "polyester", "metal", "wood"])},
                       "available": r.random() < 0.7, "price": round(r.uniform(5, 500), 2)}
    return json.dumps({"name": r.choice(PRODUCTS), "product_id": args.get("product_id"), "variants": variants}), False


def retail(self, task, hist, hints):
    names = [n for n, *_ in hist]
    if "get_order_details" in names and hist[-1][0] in ("get_order_details", "get_product_details"):
        order = json.loads(next(res for n, _, res, _ in hist if n == "get_order_details"))
        seen = {a.get("product_id") for n, a, *_ in hist if n == "get_product_details"}
        for it in order.get("item_details", []):
            if it["product_id"] not in seen:
                return "get_product_details", {"product_id": it["product_id"]}
    return _retail(self, task, hist, hints)


sim.RetailEnv.run = run_p
sim.SimModel._retail = retail

orig = E.TreeJIT.outcome
state = {"k": 0}


def outcome(self, run_id, result, reason=None):
    state["k"] += 1
    if every > 1 and state["k"] % every:
        ids = self.store.set_outcome(run_id, "pass" if result is True else "fail" if result is False else result, reason)
        self.store.x("UPDATE families SET dirty=0")
        return ids
    return orig(self, run_id, result, reason)


E.TreeJIT.outcome = outcome
t0 = time.time()
res = run_suite(n, seed=7, family="retail", mode="treejit+ok", db=db)
print(f"{n} tasks in {time.time() - t0:.0f}s; success {sum(r.success for r in res)}/{n}")
import sqlite3  # noqa: E402

c = sqlite3.connect(db)
print("avg obs chars:", c.execute("SELECT AVG(LENGTH(obs)) FROM steps WHERE obs IS NOT NULL").fetchone()[0],
      "| avg for JSON obs:", c.execute("SELECT AVG(LENGTH(obs)) FROM steps WHERE obs LIKE '{%'").fetchone()[0])
