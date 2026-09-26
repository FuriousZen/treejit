"""Synthetic workflow suite: two task families driven by a simulated model.

The simulated model sees only the conversation (like a real LLM): it parses the
task text and tool results in the messages and decides the next call. It is
stochastic: argument formatting varies, commit messages are sometimes free-form,
and with probability `noise` it takes a known-bad shortcut (skipping tests,
cancelling without checking order status). It reads <treejit-hints> and avoids
shortcuts the hints flag, the way a real model would use them.

Token usage and latency are modelled from prompt/response size, not slept.

Families
  coding  Bash/Read/Edit tools, Claude-Code-like system prompt with a varying env block.
          typo fix | version bump | delete module   (shared tail: git add/rm -> commit -> push)
  retail  tau-bench-like customer service tools.
          "I don't want order X": branch on order status (pending -> cancel, delivered -> return,
          processed -> transfer to a human)
"""

from __future__ import annotations

import json
import random
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

# ============================================================================ coding

CODING_TOOLS = [
    {"name": "Bash", "description": "Run a shell command in the repository.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "Read", "description": "Read a file.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
    {"name": "Edit", "description": "Replace old_string with new_string in a file.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                                       "new_string": {"type": "string"}},
                      "required": ["file_path", "old_string", "new_string"]}},
]

CODING_SYSTEM = (
    "You are a coding agent working in a git repository. Use the tools to inspect files, make minimal edits, "
    "run the test suite before committing, and push when the task asks for it. Keep commits focused.\n"
    "Always check `git status` before you start. Never push code that has not passed the tests.\n"
    + "Guidelines: prefer small diffs; explain what you did at the end.\n" * 12
)

PATHS = [
    "src/net/client.py", "src/net/server.py", "src/db/models.py", "src/db/queries.py", "src/api/routes.py",
    "src/api/auth.py", "src/utils/strings.py", "src/utils/timeparse.py", "src/cli/main.py", "src/core/engine.py",
    "src/core/scheduler.py", "src/io/reader.py", "src/io/writer.py", "docs/guide.md", "README.md",
]
TYPOS = [("recieve", "receive"), ("seperate", "separate"), ("occured", "occurred"), ("definately", "definitely"),
         ("accomodate", "accommodate"), ("untill", "until"), ("wierd", "weird"), ("adress", "address")]
LEGACY = ["src/legacy/old_api.py", "src/legacy/compat.py", "src/legacy/v1_client.py", "src/utils/deprecated.py",
          "src/core/old_scheduler.py", "src/io/xml_reader.py"]


@dataclass
class CodingEnv:
    kind: str
    files: dict[str, str]
    target: str = ""
    typo: tuple[str, str] = ("", "")
    version: str = ""
    flaky: bool = False
    staged: set = field(default_factory=set)
    modified: set = field(default_factory=set)
    deleted: set = field(default_factory=set)
    commits: list = field(default_factory=list)
    pushed: int = 0
    tests_since_change: bool = True
    tests_ok_since_change: bool = True
    pytest_runs: int = 0
    untracked: list = field(default_factory=list)
    n_tests: int = 42
    warnings: list = field(default_factory=list)

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "Read":
            p = args.get("file_path", "")
            if p not in self.files:
                return f"File does not exist: {p}", True
            return "\n".join(f"{i + 1:>4}\t{line}" for i, line in enumerate(self.files[p].split("\n"))), False
        if name == "Edit":
            p, old, new = args.get("file_path", ""), args.get("old_string", ""), args.get("new_string", "")
            if p not in self.files:
                return f"File does not exist: {p}", True
            if not old or old not in self.files[p]:
                return "String to replace not found in file.", True
            self.files[p] = self.files[p].replace(old, new, 1)
            self.modified.add(p)
            self.tests_since_change = self.tests_ok_since_change = False
            return f"The file {p} has been updated.", False
        if name == "Bash":
            return self.bash(args.get("command", ""))
        return f"Unknown tool {name}", True

    def bash(self, cmd: str) -> tuple[str, bool]:
        outs, err = [], False
        for part in re.split(r"\s*&&\s*", cmd):
            o, e = self._one(part)
            outs.append(o)
            if e:
                err = True
                break
        return "\n".join(x for x in outs if x), err

    def _one(self, cmd: str) -> tuple[str, bool]:
        try:
            w = shlex.split(cmd)
        except ValueError:
            return "syntax error", True
        if not w:
            return "", False
        if w[:2] == ["git", "status"]:
            lines = [f" M {p}" for p in sorted(self.modified - self.staged)] + [f"M  {p}" for p in sorted(self.staged - self.deleted)] \
                + [f"D  {p}" for p in sorted(self.deleted)]
            if "--short" in w or "-s" in w:
                return "\n".join(lines + [f"?? {p}" for p in self.untracked]), False
            body = "\n".join(lines) if lines else ("" if self.untracked else "nothing to commit, working tree clean")
            if self.untracked:
                body += ("\n" if body else "") + "Untracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n" \
                    + "\n".join(f"\t{p}" for p in self.untracked) \
                    + "\n\nnothing added to commit but untracked files present (use \"git add\" to track)"
            return "On branch main\nYour branch is up to date with 'origin/main'.\n" + body, False
        if w[:2] == ["git", "add"]:
            for p in w[2:]:
                if p in ("-A", "."):
                    self.staged |= self.modified
                elif p in self.modified:
                    self.staged.add(p)
                else:
                    return f"fatal: pathspec '{p}' did not match any files", True
            return "", False
        if w[:2] == ["git", "rm"]:
            p = w[-1]
            if p not in self.files:
                return f"fatal: pathspec '{p}' did not match any files", True
            del self.files[p]
            self.deleted.add(p)
            self.staged.add(p)
            self.tests_since_change = self.tests_ok_since_change = False
            return f"rm '{p}'", False
        if w[:2] == ["git", "commit"]:
            if not self.staged:
                return "nothing to commit, working tree clean\nExit code 1", True
            msg = w[w.index("-m") + 1] if "-m" in w else "wip"
            sha = f"{random.randrange(16 ** 7):07x}"
            self.commits.append((sha, msg, set(self.staged), self.tests_ok_since_change))
            n = len(self.staged)
            self.staged, self.modified = set(), self.modified - self.staged
            return f"[main {sha}] {msg}\n {n} file{'s' if n > 1 else ''} changed", False
        if w[:2] == ["git", "push"]:
            if not self.commits or self.pushed == len(self.commits):
                return "Everything up-to-date", False
            old = self.commits[self.pushed - 1][0] if self.pushed else "3f2a1c9"
            self.pushed = len(self.commits)
            return f"To github.com:acme/app.git\n   {old}..{self.commits[-1][0]}  main -> main", False
        if w[:3] == ["python", "-m", "pytest"]:
            self.pytest_runs += 1
            self.tests_since_change = True
            broken = self.kind == "typo" and self.typo[0] in self.files.get(self.target, "")
            n_all = self.n_tests
            if self.flaky and "--lf" not in w and self.pytest_runs == 1:
                self.tests_ok_since_change = False
                return (_progress("F" + "." * (n_all - 1)) + "\nFAILED tests/test_net.py::test_timeout - TimeoutError: timed out after 5s\n"
                        f"1 failed, {n_all - 1} passed in 3.12s\nExit code 1"), True
            if broken:
                self.tests_ok_since_change = False
                return (_progress("." * (n_all - 1) + "F") + f"\nFAILED tests/test_spelling.py::test_no_typos - AssertionError: '{self.typo[0]}'\n"
                        f"1 failed, {n_all - 1} passed in 2.9s\nExit code 1"), True
            self.tests_ok_since_change = True
            n = 1 if "--lf" in w else n_all
            warn = "" if n == 1 or not self.warnings else _warnings_summary(self.warnings)
            tail = f"{n} passed, {len(self.warnings)} warnings in {1.1 + n / 40:.2f}s" if warn else f"{n} passed in {1.1 + n / 40:.2f}s"
            return f"{_progress('.' * n)}{warn}\n{tail}", False
        if w[0] in ("ls", "cat"):
            return "\n".join(sorted(self.files)), False
        return f"bash: {w[0]}: command not found\nExit code 127", True

    def verify(self) -> tuple[bool, str]:
        if not self.commits:
            return False, "nothing was committed"
        if self.pushed != len(self.commits):
            return False, "commit was not pushed"
        if not all(ok for *_, ok in self.commits):
            return False, "pushed without a passing test run"
        if self.kind == "typo":
            if self.target not in self.files:
                return False, "target file was deleted"
            if self.typo[0] in self.files[self.target] or self.typo[1] not in self.files[self.target]:
                return False, "typo not fixed"
        elif self.kind == "bump":
            if f'version = "{self.version}"' not in self.files["pyproject.toml"]:
                return False, "version not bumped"
        elif self.kind == "remove":
            if self.target in self.files:
                return False, "module not removed"
        return True, "ok"


def _progress(marks: str, width: int = 80) -> str:
    """pytest -q progress rows: `....F....  [ 42%]`."""
    rows, n = [], len(marks)
    for i in range(0, n, width):
        chunk = marks[i : i + width]
        rows.append(f"{chunk:<{width}} [{round(100 * (i + len(chunk)) / n):>3}%]")
    return "\n".join(rows)


def _warnings_summary(warnings: list[str]) -> str:
    body = "\n".join(f"{w}\n    warnings.warn(\n" for w in warnings)
    return (f"\n{'=' * 26} warnings summary {'=' * 26}\n{body}\n"
            "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html")


WARNINGS = [
    "src/db/models.py:{n}: DeprecationWarning: declarative_base() is deprecated; use DeclarativeBase",
    "src/utils/timeparse.py:{n}: DeprecationWarning: datetime.utcnow() is deprecated and scheduled for removal",
    "tests/test_api.py::test_login_{n}: PytestUnraisableExceptionWarning: Exception ignored in socket.close",
    "src/net/client.py:{n}: ResourceWarning: unclosed <ssl.SSLSocket fd={n}>",
    "src/core/scheduler.py:{n}: RuntimeWarning: coroutine 'Scheduler.tick' was never awaited",
]
BUILD_JUNK = ["build/lib/acme/__init__.py", "build/lib/acme/net/client.py", "dist/acme_app-{v}.tar.gz", ".coverage",
              "htmlcov/index.html", "htmlcov/status.json", "notes/todo-{n}.md", "scratch/profile-{n}.prof", ".env.local",
              "tmp/cache-{n}.json", "src/acme.egg-info/PKG-INFO", "src/acme.egg-info/SOURCES.txt"]
_BODY = [
    "import logging", "from dataclasses import dataclass", "from typing import Any", "",
    "log = logging.getLogger(__name__)", "", "", "@dataclass", "class Config:", "    retries: int = 3",
    "    timeout: float = 5.0", "    verbose: bool = False", "", "", "def _normalize(value: Any) -> Any:",
    "    if isinstance(value, str):", "        return value.strip()", "    return value", "", "",
    "def handler(value):", "    value = _normalize(value)", "    log.debug(\"handling %r\", value)", "    return value", "", "",
    "class Worker:", "    def __init__(self, cfg: Config) -> None:", "        self.cfg = cfg", "        self.done = 0", "",
    "    def run(self, items: list) -> list:", "        out = []", "        for item in items:",
    "            out.append(handler(item))", "            self.done += 1", "        return out", "",
    "    def reset(self) -> None:", "        self.done = 0", "", "", "def main(argv: list | None = None) -> int:",
    "    cfg = Config()", "    w = Worker(cfg)", "    w.run(argv or [])", "    return 0", "", "",
    "def _retry(fn, attempts: int = 3):", "    last = None", "    for _ in range(attempts):", "        try:",
    "            return fn()", "        except OSError as e:  # transient", "            last = e", "    raise last", "", "",
    "if __name__ == \"__main__\":", "    raise SystemExit(main())",
]


def _module(rng: random.Random, path: str, special: str | None = None) -> str:
    n = rng.randint(40, len(_BODY))
    lines = [f"# {path}", f'"""{path.rsplit("/", 1)[-1]}: part of acme-app."""', ""] + _BODY[:n]
    if special is not None:
        at = lines.index("def handler(value):") + 1 if "def handler(value):" in lines else len(lines)
        lines.insert(at, special)
    return "\n".join(lines) + "\n"


def _pyproject(version: str) -> str:
    return (f'[project]\nname = "acme-app"\nversion = "{version}"\nrequires-python = ">=3.10"\n'
            'description = "Acme application server and client"\nreadme = "README.md"\nlicense = {text = "MIT"}\n'
            'authors = [{name = "Acme Engineering", email = "eng@acme.example"}]\n'
            'dependencies = ["httpx", "pydantic", "sqlalchemy", "click", "rich", "uvicorn"]\n\n'
            '[project.optional-dependencies]\ndev = ["pytest", "pytest-cov", "ruff", "mypy", "types-requests"]\n'
            'docs = ["mkdocs", "mkdocs-material"]\n\n[project.scripts]\nacme = "acme.cli.main:main"\n\n'
            '[build-system]\nrequires = ["setuptools", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n'
            '[tool.setuptools.packages.find]\nwhere = ["src"]\n\n'
            '[tool.pytest.ini_options]\naddopts = "-q"\ntestpaths = ["tests"]\nfilterwarnings = ["default"]\n\n'
            '[tool.ruff]\nline-length = 120\ntarget-version = "py310"\n\n[tool.ruff.lint]\n'
            'select = ["E", "F", "W", "I", "B", "UP"]\nignore = ["E501"]\n\n'
            '[tool.mypy]\nstrict = true\nwarn_unused_ignores = true\nplugins = ["pydantic.mypy"]\n\n'
            '[tool.coverage.run]\nsource = ["src"]\nbranch = true\n\n[tool.coverage.report]\nshow_missing = true\n'
            'skip_covered = true\nfail_under = 80\n')


def make_coding_task(rng: random.Random, i: int) -> tuple[str, CodingEnv]:
    files = {p: _module(rng, p) for p in PATHS}
    files["pyproject.toml"] = _pyproject(f"{rng.randint(0, 3)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}")
    for p in LEGACY:
        files[p] = f"# {p}\n# deprecated\n" + _module(rng, p)
    kind = rng.choices(["typo", "bump", "remove"], weights=[5, 3, 2])[0]
    flaky = rng.random() < 0.25
    extra = {
        "untracked": sorted({j.format(n=rng.randint(1, 99), v="0.9.0") for j in rng.sample(BUILD_JUNK, rng.randint(4, 12))})
        if rng.random() < 0.35 else [],
        "n_tests": rng.randint(150, 420),
        "warnings": [w.format(n=rng.randint(10, 400)) for w in rng.sample(WARNINGS, rng.randint(0, 5))],
    }
    if kind == "typo":
        path, (bad, good) = rng.choice(PATHS), rng.choice(TYPOS)
        files[path] = _module(rng, path, f"    # we {bad} the value here")
        text = rng.choice([
            f"Fix the typo '{bad}' -> '{good}' in {path}, run the tests, then commit and push.",
            f"There's a spelling mistake in {path}: '{bad}' should be '{good}'. Please fix it, test, commit and push.",
            f"Please correct '{bad}' to '{good}' in {path} and ship it (tests, commit, push).",
        ])
        return text, CodingEnv("typo", files, path, (bad, good), flaky=flaky, **extra)
    if kind == "bump":
        v = f"{rng.randint(1, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 20)}"
        text = rng.choice([
            f"Bump the package version to {v} and release it.",
            f"Release version {v}: update pyproject.toml, run the tests, commit and push.",
        ])
        return text, CodingEnv("bump", files, "pyproject.toml", version=v, flaky=flaky, **extra)
    path = rng.choice(LEGACY)
    text = rng.choice([
        f"Delete the unused module {path} and push the change.",
        f"{path} is dead code. Remove it, make sure tests pass, commit and push.",
    ])
    return text, CodingEnv("remove", files, path, flaky=flaky, **extra)


# ============================================================================ retail

RETAIL_TOOLS = [
    {"name": "find_user_id_by_email", "description": "Find a user id by email.",
     "input_schema": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}},
    {"name": "get_user_details", "description": "Get user details.",
     "input_schema": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}},
    {"name": "get_order_details", "description": "Get order details.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "cancel_pending_order", "description": "Cancel a pending order.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                      "required": ["order_id", "reason"]}},
    {"name": "return_delivered_order_items", "description": "Return items of a delivered order.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "item_ids": {"type": "array"},
                                                       "payment_method_id": {"type": "string"}},
                      "required": ["order_id", "item_ids", "payment_method_id"]}},
    {"name": "transfer_to_human_agents", "description": "Transfer to a human agent.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]
RETAIL_SYSTEM = (
    "You are a customer service agent for an online retailer. Authenticate the user by email, look up their "
    "details and the order before acting. Pending orders can be cancelled (reason 'no longer needed'); delivered "
    "orders can be returned to the original payment method; anything else goes to a human agent.\n"
    + "Policy: be concise, confirm actions, never act on an order you have not looked up.\n" * 10
)
FIRST = ["mia", "noah", "liam", "emma", "olivia", "ava", "lucas", "yusuf", "aarav", "sofia", "chen", "fatima"]
LAST = ["garcia", "smith", "khan", "nguyen", "silva", "kim", "müller", "rossi", "patel", "cohen"]


@dataclass
class RetailEnv:
    email: str
    user_id: str
    order_id: str
    status: str
    items: list
    payment: str
    actions: list = field(default_factory=list)
    looked_up: bool = False

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "find_user_id_by_email":
            return (self.user_id, False) if args.get("email") == self.email else ("Error: user not found", True)
        if name == "get_user_details":
            if args.get("user_id") != self.user_id:
                return "Error: user not found", True
            return json.dumps({"user_id": self.user_id, "email": self.email, "orders": [self.order_id],
                               "payment_methods": [{"id": self.payment, "source": "credit_card"}]}), False
        if name == "get_order_details":
            if args.get("order_id") != self.order_id:
                return "Error: order not found", True
            self.looked_up = True
            return json.dumps({"order_id": self.order_id, "user_id": self.user_id, "status": self.status,
                               "items": self.items, "total": round(19.99 * len(self.items), 2)}), False
        if name == "cancel_pending_order":
            if args.get("order_id") != self.order_id:
                return "Error: order not found", True
            if self.status != "pending":
                return f"Error: order status is {self.status}, cannot cancel", True
            self.actions.append(("cancel", args.get("reason")))
            self.status = "cancelled"
            return json.dumps({"order_id": self.order_id, "status": "cancelled"}), False
        if name == "return_delivered_order_items":
            if self.status != "delivered":
                return f"Error: order status is {self.status}", True
            self.actions.append(("return", tuple(args.get("item_ids") or []), args.get("payment_method_id")))
            self.status = "return requested"
            return json.dumps({"order_id": self.order_id, "status": "return requested"}), False
        if name == "transfer_to_human_agents":
            self.actions.append(("transfer",))
            return "Transfer successful", False
        return f"Unknown tool {name}", True

    def verify(self) -> tuple[bool, str]:
        if not self.actions:
            return False, "no action taken"
        a = self.actions[-1]
        if not self.looked_up:
            return False, "acted without looking up the order"
        if a[0] == "cancel":
            return (a[1] == "no longer needed", "ok" if a[1] == "no longer needed" else "wrong cancel reason")
        if a[0] == "return":
            ok = set(a[1]) == set(self.items) and a[2] == self.payment
            return ok, "ok" if ok else "wrong return items or payment method"
        return True, "ok"


def make_retail_task(rng: random.Random, i: int) -> tuple[str, RetailEnv]:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    uid = f"{first}_{last}_{rng.randint(1000, 9999)}"
    email = f"{first}.{last}{rng.randint(1, 99)}@example.com"
    oid = f"#W{rng.randint(1000000, 9999999)}"
    status = rng.choices(["pending", "delivered", "processed"], weights=[5, 4, 1])[0]
    items = [str(rng.randint(10 ** 9, 10 ** 10 - 1)) for _ in range(rng.randint(1, 3))]
    env = RetailEnv(email, uid, oid, status, items, f"credit_card_{rng.randint(1000000, 9999999)}")
    text = rng.choice([
        f"Hi! My email is {email}. I don't want order {oid} anymore.",
        f"Hello, this is {first.title()} ({email}). Please take care of order {oid}, I no longer need it.",
        f"email: {email}\nI changed my mind about order {oid}. Can you help?",
    ])
    return text, env


# ============================================================================ simulated model


def _history(messages: list[dict]) -> list[tuple[str, dict, str, bool]]:
    calls: dict[str, tuple[str, dict]] = {}
    order: list[str] = []
    results: dict[str, tuple[str, bool]] = {}
    for m in messages[1:]:
        if not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                calls[b["id"]] = (b["name"], b["input"])
                order.append(b["id"])
            elif b.get("type") == "tool_result":
                c = b.get("content")
                text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or [])
                results[b["tool_use_id"]] = (text, bool(b.get("is_error")))
    return [(calls[i][0], calls[i][1], *results.get(i, ("", False))) for i in order]


def _hints(messages: list[dict]) -> str:
    last = messages[-1].get("content") if messages else None
    if isinstance(last, list):
        return "\n".join(b.get("text", "") for b in last if b.get("type") == "text" and "treejit-hints" in b.get("text", ""))
    return ""


class SimModel:
    """Anthropic-Messages-shaped fake model. `__call__(body) -> response dict`."""

    def __init__(self, seed: int = 0, noise: float = 0.06) -> None:
        self.rng = random.Random(seed)
        self.noise = noise
        self.calls = 0           # full calls
        self.small_calls = 0     # treejit T2/T3 subcalls (forced treejit_* tool)
        self.small_tokens = [0, 0]

    def __call__(self, body: dict) -> dict:
        forced = (body.get("tool_choice") or {}).get("name", "")
        if forced.startswith("treejit_"):
            return self._subcall(body, forced)
        self.calls += 1
        msgs = body["messages"]
        task = msgs[0]["content"] if isinstance(msgs[0]["content"], str) else msgs[0]["content"][0]["text"]
        hist = _history(msgs)
        hints = _hints(msgs)
        tools = {t["name"] for t in body.get("tools", [])}
        if "Bash" in tools:
            action = self._coding(task, hist, hints)
        else:
            action = self._retail(task, hist, hints)
        if action is None:
            content = [{"type": "text", "text": "Done. " + self.rng.choice(["All set.", "Task complete.", "Finished."])}]
            stop = "end_turn"
        else:
            name, args = action
            content = [{"type": "tool_use", "id": f"toolu_{self.rng.randrange(16 ** 20):020x}", "name": name, "input": args}]
            stop = "tool_use"
        prompt_chars = len(body.get("system", "")) + len(json.dumps(body.get("tools", []))) + len(json.dumps(msgs))
        out_tokens = len(json.dumps(content)) // 4 + 40  # + hidden reasoning
        return {"id": f"msg_{self.rng.randrange(16 ** 12):012x}", "type": "message", "role": "assistant",
                "model": body.get("model", "sim"), "content": content, "stop_reason": stop, "stop_sequence": None,
                "usage": {"input_tokens": prompt_chars // 4, "output_tokens": out_tokens}}

    # ---------------------------------------------------------------- treejit subcalls
    def _subcall(self, body: dict, tool: str) -> dict:
        """Answer a T3 fill / T2 choose from the short prompt alone, with the same policy:
        work out what it would do next and match that against the proposed call(s)."""
        self.small_calls += 1
        prompt = body["messages"][0]["content"]
        try:
            answer = self._subcall_answer(prompt, tool)
        except Exception:  # a confused model: say nothing useful
            answer = {"choice": 0} if tool == "treejit_choose" else {"not_this_step": True}
        content = [{"type": "tool_use", "id": f"toolu_{self.rng.randrange(16 ** 20):020x}", "name": tool, "input": answer}]
        prompt_chars = len(body.get("system", "")) + len(json.dumps(body.get("tools", []))) + len(json.dumps(body["messages"]))
        usage = {"input_tokens": prompt_chars // 4, "output_tokens": len(json.dumps(content)) // 4 + 10}
        self.small_tokens[0] += usage["input_tokens"]
        self.small_tokens[1] += usage["output_tokens"]
        return {"id": f"msg_{self.rng.randrange(16 ** 12):012x}", "type": "message", "role": "assistant",
                "model": body.get("model", "sim"), "content": content, "stop_reason": "tool_use", "stop_sequence": None,
                "usage": usage}

    def _subcall_answer(self, prompt: str, tool: str) -> dict:
        task = re.search(r"<task>\n(.*?)\n</task>", prompt, re.S).group(1)
        results = {m.group(1): (m.group(3), m.group(2) == "true")
                   for m in re.finditer(r'<result n="(\d+)" error="(true|false)">\n(.*?)\n</result>', prompt, re.S)}
        hist = []
        for m in re.finditer(r'<step n="(\d+)">(.*?)</step>', prompt):
            st = json.loads(m.group(2))
            hist.append((st["tool"], st["input"], *results.get(m.group(1), ("", False))))
        if tool == "treejit_fill":
            proposals = [(0, json.loads(re.search(r"^Next call: (.*)$", prompt, re.M).group(1)))]
        else:
            proposals = [(int(m.group(1)), json.loads(m.group(2)))
                         for m in re.finditer(r"^(\d+)\. (\{.*\})  \(used in", prompt, re.M)]
        names = {n for n, *_ in hist} | {p["tool"] for _, p in proposals}
        action = self._coding(task, hist, "") if names & {"Bash", "Read", "Edit"} else self._retail(task, hist, "")
        for n, p in proposals:
            vals = _match_proposal(p, action)
            if vals is not None:
                return vals if tool == "treejit_fill" else {"choice": n, **{f"o{n}_{k}": v for k, v in vals.items()}}
        return {"not_this_step": True} if tool == "treejit_fill" else {"choice": 0}

    # ---------------------------------------------------------------- coding policy
    def _coding(self, task: str, hist: list, hints: str) -> tuple[str, dict] | None:
        rng = self.rng
        done = [(n, a) for n, a, _, _ in hist]
        cmds = [a.get("command", "") for n, a in done if n == "Bash"]
        last = hist[-1] if hist else None
        m_typo = re.search(r"'([a-z]+)' (?:->|should be|to) '([a-z]+)'", task)
        m_path = re.search(r"((?:src|docs)/[\w/]+\.\w+|README\.md)", task)
        m_ver = re.search(r"\b(\d+\.\d+\.\d+)\b", task)
        kind = "typo" if m_typo else "bump" if m_ver else "remove"
        if not hist:
            if kind == "bump":
                return "Read", {"file_path": "pyproject.toml"}
            return "Bash", {"command": "git status" if rng.random() < 0.15 else "git status --short"}
        tested = any(c.startswith("python -m pytest") for c in cmds)
        last_tests_failed = bool(last and last[0] == "Bash" and last[1].get("command", "").startswith("python -m pytest") and last[3])
        committed = any(c.startswith("git commit") for c in cmds)
        pushed = any(c.startswith("git push") for c in cmds)
        staged = any(c.startswith("git add") or c.startswith("git rm") for c in cmds)
        if pushed:
            return None
        if committed:
            return "Bash", {"command": "git push origin main"}
        if last_tests_failed:
            if "--lf" in last[1]["command"] or "TimeoutError" not in last[2]:
                return None  # give up
            return "Bash", {"command": "python -m pytest -q --lf"}
        skip_ok = "AVOID" not in hints
        if kind == "typo":
            path, (bad, good) = m_path.group(1), m_typo.groups()
            if not any(n == "Read" for n, _ in done):
                return "Read", {"file_path": path}
            if not any(n == "Edit" for n, _ in done):
                return "Edit", {"file_path": path, "old_string": bad, "new_string": good}
            if not tested and not staged:
                if skip_ok and rng.random() < self.noise:
                    return "Bash", {"command": f"git add {path}"}  # shortcut: goes on to commit untested
                return "Bash", {"command": "python -m pytest -q"}
            if not staged:
                return "Bash", {"command": f"git add {path}"}
            return "Bash", {"command": f"git commit -m {shlex.quote('Fix typo in ' + path)}"}
        if kind == "bump":
            v = m_ver.group(1)
            reads = [r for n, a, r, _ in hist if n == "Read" and a.get("file_path") == "pyproject.toml"]
            if not reads:
                return "Read", {"file_path": "pyproject.toml"}
            if not any(n == "Edit" for n, _ in done):
                m_cur = re.search(r'version = "([^"]+)"', reads[-1])
                if m_cur is None:
                    return None  # pyproject.toml is gone (e.g. deleted by a misrouted step): give up
                cur = m_cur.group(1)
                return "Edit", {"file_path": "pyproject.toml", "old_string": f'version = "{cur}"', "new_string": f'version = "{v}"'}
            if not tested:
                return "Bash", {"command": "python -m pytest -q"}
            if not staged:
                return "Bash", {"command": "git add pyproject.toml"}
            return "Bash", {"command": f"git commit -m {shlex.quote('Bump version to ' + v)}"}
        path = m_path.group(1)
        if not staged:
            return "Bash", {"command": f"git rm {path}"}
        if not tested:
            if skip_ok and rng.random() < self.noise:
                return "Bash", {"command": f"git commit -m {shlex.quote('Remove ' + path)}"}
            return "Bash", {"command": "python -m pytest -q"}
        msg = rng.choice([f"Remove {path}", f"Delete unused module {path}", f"Drop dead code in {path}", "Remove legacy module"])
        return "Bash", {"command": f"git commit -m {shlex.quote(msg)}"}

    # ---------------------------------------------------------------- retail policy
    def _retail(self, task: str, hist: list, hints: str) -> tuple[str, dict] | None:
        email = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", task).group(0)
        oid = re.search(r"#W\d+", task).group(0)
        names = [n for n, *_ in hist]
        if not hist:
            return "find_user_id_by_email", {"email": email}
        last = hist[-1]
        if last[0] in ("cancel_pending_order", "return_delivered_order_items", "transfer_to_human_agents"):
            return None
        if "get_user_details" not in names:
            return "get_user_details", {"user_id": last[2].strip()}
        if "get_order_details" not in names:
            if "AVOID" not in hints and self.rng.random() < self.noise:
                return "cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}  # shortcut
            return "get_order_details", {"order_id": oid}
        order = json.loads(next(r for n, _, r, _ in hist if n == "get_order_details"))
        user = json.loads(next(r for n, _, r, _ in hist if n == "get_user_details"))
        if order["status"] == "pending":
            return "cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}
        if order["status"] == "delivered":
            return "return_delivered_order_items", {"order_id": oid, "item_ids": order["items"],
                                                    "payment_method_id": user["payment_methods"][0]["id"]}
        return "transfer_to_human_agents", {"summary": f"User {user['user_id']} wants to drop order {oid} (status {order['status']})."}


_PH = re.compile(r"<([A-Za-z0-9_]+)>")


def _match_proposal(prop: dict, action: tuple[str, dict] | None) -> dict | None:
    """Placeholder values that turn the proposed call into `action`, or None if it's a different call."""
    if action is None or prop["tool"] != action[0] or set(prop["input"]) != set(action[1]):
        return None
    out: dict[str, str] = {}
    for k, tv in prop["input"].items():
        av = action[1][k]
        if not (isinstance(tv, str) and _PH.search(tv)):
            if tv != av:
                return None
            continue
        whole = _PH.fullmatch(tv)
        if whole and not isinstance(av, str):
            out[whole.group(1)] = json.dumps(av)
            continue
        if not isinstance(av, str):
            return None
        pattern, seen = "", set()
        for i, part in enumerate(_PH.split(tv)):
            if i % 2 == 0:
                pattern += re.escape(part)
            else:
                pattern += f"(?P={part})" if part in seen else f"(?P<{part}>.+?)"
                seen.add(part)
        m = re.fullmatch(pattern, av, re.S)
        if m is None:
            return None
        for name, val in m.groupdict().items():
            if k == "command":  # the model gives plain values; treejit does the shell quoting
                try:
                    words = shlex.split(val)
                    val = words[0] if len(words) == 1 else val
                except ValueError:
                    pass
            out[name] = val
    return out


def make_task(rng: random.Random, i: int, family: str) -> tuple[str, str, Any, list, str]:
    """Returns (family, task_text, env, tools, system)."""
    if family == "mixed":
        family = "coding" if rng.random() < 0.5 else "retail"
    if family == "coding":
        text, env = make_coding_task(rng, i)
        day = rng.randint(1, 28)
        system = CODING_SYSTEM + f"<env>\nWorking directory: /work/acme-app-{rng.randint(1, 5)}\nToday's date: 2026-09-{day:02d}\n</env>\n"
        return "coding", text, env, CODING_TOOLS, system
    text, env = make_retail_task(rng, i)
    return "retail", text, env, RETAIL_TOOLS, RETAIL_SYSTEM
