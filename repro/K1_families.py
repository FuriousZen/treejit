"""K1: stable-prefix family learning vs dynamic blocks early in the system prompt.

Cases (tools identical everywhere):
  A. Claude-Code-like: short identity line, then <env> (cwd, date, git status), then a large static section.
  B. same, <env> placed after 60% of the prompt.
  C. two DIFFERENT agents sharing a 150-char preamble (must NOT merge), where agent 1's family already
     shrank to the preamble because its own <env> follows the preamble.
Then a prototype of line-set keying (proposed fix) on the same inputs.
"""
import hashlib
import re
import sys
import tempfile

sys.path.insert(0, "/home/user/treejit/tests")
from conftest import TOOLS  # noqa: E402

from treejit import families  # noqa: E402
from treejit.store import Store  # noqa: E402

IDENT = "You are Claude Code, Anthropic's official CLI for Claude.\n"
STATIC = "".join(f"# Rule {i}: when doing thing {i}, prefer the careful option and explain briefly.\n" for i in range(250))  # ~20k chars
PREAMBLE = "You are an interactive agent that helps users with software engineering tasks. Use the tools below.\n" \
           "Be concise. Never guess file contents.\n" \
           "Always read a file before editing it. Prefer small, reviewable changes.\n" \
           "Run the tests after every change and report failures verbatim.\n" \
           "Ask before destructive operations such as deleting files or force-pushing.\n"


def env(i):
    return (f"<env>\nWorking directory: /home/u/proj{i}\nIs directory a git repo: {'Yes' if i % 2 else 'No'}\n"
            f"Platform: linux\nToday's date: 2026-09-{10 + i:02d}\n</env>\n"
            f"gitStatus: Current branch: feat-{i}\nStatus:\nM src/a{i}.py\n?? tmp{i}.txt\n")


def case_a(i):
    return IDENT + env(i) + STATIC


def case_b(i):
    cut = int(len(STATIC) * 0.6)
    return IDENT + STATIC[:cut] + env(i) + STATIC[cut:]


AGENT2 = "".join(f"# Reviewer rule {i}: flag risky diffs of kind {i}; never edit files yourself.\n" for i in range(40))


def case_c1(i):  # agent 1: small prompt, env right after the shared preamble
    return PREAMBLE + env(i) + "Agent 1 extra line.\n"


def case_c2(i):  # agent 2: a different agent that shares the preamble
    return PREAMBLE + AGENT2 + env(i)


def run(label, prompts, store):
    fids = [families.resolve(store, p, TOOLS, "anthropic") for p in prompts]
    return fids


def show(label, fids, store):
    distinct = list(dict.fromkeys(fids))
    prefixes = {r["id"]: len(r["prefix"]) for r in store.q("SELECT id, prefix FROM families")}
    print(f"  {label}: families={[distinct.index(f) for f in fids]}  learned prefix chars={[prefixes[f] for f in distinct]}")


print("== current families.resolve ==")
with tempfile.TemporaryDirectory() as t:
    s = Store(t + "/a.db")
    show(f"A (<env> near top, system {len(case_a(0))} chars) x5 sessions", run("A", [case_a(i) for i in range(5)], s), s)
    s = Store(t + "/b.db")
    show("B (<env> at 60%) x5 sessions", run("B", [case_b(i) for i in range(5)], s), s)
    s = Store(t + "/c.db")
    f1 = run("C1", [case_c1(i) for i in range(3)], s)
    f2 = run("C2", [case_c2(i) for i in range(3)], s)
    show("C agent1 x3 then agent2 x3", f1 + f2, s)
    print("   -> agent 2 (a different system prompt) " + ("JOINED agent 1's family" if set(f2) & set(f1) else "got its own family"))


# ------------------------------------------------------------------ prototype of the proposed fix
MASK = [(re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"), (re.compile(r"(/[\w.\-]+){2,}"), "<path>"),
        (re.compile(r"\b[0-9a-f]{7,40}\b"), "<hex>"), (re.compile(r"\d+"), "<n>")]


def norm(line):
    line = line.strip()
    for rx, rep in MASK:
        line = rx.sub(rep, line)
    return line


def lines_of(text):
    out = {}
    for ln in text.splitlines():
        n = norm(ln)
        if n:
            k = hashlib.sha1(n.encode()).hexdigest()[:12]
            out[k] = out.get(k, 0) + len(n)  # multiset: repeated (masked) lines weigh by count
    return out


class LineFamilies:
    """family = tools_hash + weighted line set. Each family keeps per-line membership counts; stable lines are
    those present in >= STABLE of its members. A prompt joins the best family if (a) it contains >= COVER of the
    family's stable chars and (b) the family's stable lines cover >= COVER_NEW of the prompt's chars
    (dynamic lines anywhere are tolerated up to 1-COVER_NEW; a different agent sharing a short preamble fails (a)/(b))."""
    STABLE, COVER, COVER_NEW = 0.8, 0.9, 0.8

    def __init__(self):
        self.fams = []   # dicts: {"n": members, "cnt": {line: count}, "w": {line: chars}}

    def stable(self, f):
        return {k for k, c in f["cnt"].items() if c >= self.STABLE * f["n"]}

    def resolve(self, text):
        L = lines_of(text)
        tot = sum(L.values()) or 1
        best, score = None, 0.0
        for i, f in enumerate(self.fams):
            st = self.stable(f)
            st_chars = sum(f["w"][k] for k in st) or 1
            a = sum(f["w"][k] for k in st if k in L) / st_chars
            b = sum(v for k, v in L.items() if k in st) / tot
            if a >= self.COVER and b >= self.COVER_NEW and a + b > score:
                best, score = i, a + b
        if best is None:
            self.fams.append({"n": 0, "cnt": {}, "w": {}})
            best = len(self.fams) - 1
        f = self.fams[best]
        f["n"] += 1
        for k, v in L.items():
            f["cnt"][k] = f["cnt"].get(k, 0) + 1
            f["w"][k] = v
        return best


print("\n== prototype: masked line-set keying ==")
for label, prompts in (("A", [case_a(i) for i in range(5)]), ("B", [case_b(i) for i in range(5)]),
                       ("C agent1 x3 then agent2 x3", [case_c1(i) for i in range(3)] + [case_c2(i) for i in range(3)]),
                       ("D Claude Code vs a custom agent that appends 30% extra lines", [case_a(i) for i in range(3)] +
                        [case_a(i) + STATIC[: len(STATIC) // 2].replace("Rule", "Custom rule") for i in range(3)])):
    lf = LineFamilies()
    print(f"  {label}: families={[lf.resolve(p) for p in prompts]}")
