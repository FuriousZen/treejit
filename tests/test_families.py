"""K1: family keying by masked line sets (tools + stable lines of the system prompt)."""

from __future__ import annotations

import random
import sqlite3
import time

from treejit import families
from treejit.families import lines_of, normalize, resolve
from treejit.store import Store

TOOLS = [{"name": "Bash", "input_schema": {}}, {"name": "Read", "input_schema": {}}]

IDENT = "You are Claude Code, Anthropic's official CLI for Claude.\n"
STATIC = "".join(f"# Rule {i}: when doing thing {i}, prefer the careful option and explain briefly.\n" for i in range(250))
PREAMBLE = ("You are an interactive agent that helps users with software engineering tasks. Use the tools below.\n"
            "Be concise. Never guess file contents.\n"
            "Always read a file before editing it. Prefer small, reviewable changes.\n"
            "Run the tests after every change and report failures verbatim.\n"
            "Ask before destructive operations such as deleting files or force-pushing.\n")
AGENT2 = "".join(f"# Reviewer rule {i}: flag risky diffs of kind {i}; never edit files yourself.\n" for i in range(40))


def env(i):
    return (f"<env>\nWorking directory: /home/u/proj{i}\nIs directory a git repo: {'Yes' if i % 2 else 'No'}\n"
            f"Platform: linux\nToday's date: 2026-09-{10 + i:02d}\n</env>\n"
            f"gitStatus: Current branch: feat-{i}\nStatus:\nM src/a{i}.py\n?? tmp{i}.txt\n")


def fams(st):
    return st.q("SELECT * FROM families ORDER BY created")


# ------------------------------------------------------------------ acceptance (PLAN.md K1)


def test_early_env_block_gives_one_family(tmp_path):
    st = Store(str(tmp_path / "a.db"))
    ids = {resolve(st, IDENT + env(i) + STATIC, TOOLS, "anthropic") for i in range(5)}
    assert len(ids) == 1 and len(fams(st)) == 1
    cut = len(STATIC) * 6 // 10
    st2 = Store(str(tmp_path / "b.db"))
    assert len({resolve(st2, IDENT + STATIC[:cut] + env(i) + STATIC[cut:], TOOLS, "anthropic") for i in range(5)}) == 1


def test_agents_sharing_a_short_preamble_stay_apart(tmp_path):
    def a1(i):
        return PREAMBLE + env(i) + "Agent 1 extra line.\n"

    def a2(i):
        return PREAMBLE + AGENT2 + env(i)

    for order in ((a1, a2), (a2, a1)):
        st = Store(str(tmp_path / f"{order[0].__name__}.db"))
        f1 = {resolve(st, order[0](i), TOOLS, "anthropic") for i in range(3)}
        f2 = {resolve(st, order[1](i), TOOLS, "anthropic") for i in range(3)}
        assert len(f1) == 1 and len(f2) == 1 and f1 != f2


def test_agent_with_much_extra_content_splits(tmp_path):
    st = Store(str(tmp_path / "d.db"))
    base = {resolve(st, IDENT + env(i) + STATIC, TOOLS, "anthropic") for i in range(3)}
    extra = STATIC[: len(STATIC) // 2].replace("Rule", "Custom rule")
    custom = {resolve(st, IDENT + env(i) + STATIC + extra, TOOLS, "anthropic") for i in range(3)}
    assert len(base) == 1 and len(custom) == 1 and base != custom


def _old_db(path, rows):
    """A database as the prefix-based keying left it (no n_members / legacy_prefix, no line tables)."""
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE families(id TEXT PRIMARY KEY, tools_hash TEXT, prefix TEXT, dialect TEXT,
                        created REAL, updated REAL, built_at REAL DEFAULT 0, dirty INTEGER DEFAULT 1);""")
    db.executemany("INSERT INTO families(id, tools_hash, prefix, dialect, created, updated) VALUES(?,?,?,?,?,?)", rows)
    db.commit()
    db.close()


def test_old_prefix_families_resolve_to_the_same_id(tmp_path):
    th = families.h(families.canon(TOOLS))
    # fam A: the old code shrank the prefix to what precedes the dynamic tail (<env> + git status at the end)
    pa = IDENT + STATIC[:4000]
    # fam B: a family that saw a single prompt (prefix = the whole prompt)
    pb = "You are a retail support agent.\n" + "Policy: be concise, confirm actions, never act on an order you have not looked up.\n" * 10
    # fam C: short static part and a long dynamic tail: only the old startswith rule can keep it together
    pc = PREAMBLE
    path = str(tmp_path / "old.db")
    _old_db(path, [("famAAAAAAAAA", th, pa, "anthropic", 1.0, 1.0), ("famBBBBBBBBB", th, pb, "anthropic", 2.0, 2.0),
                   ("famCCCCCCCCC", th, pc, "anthropic", 3.0, 3.0)])
    st = Store(path)
    for i in range(4):
        assert resolve(st, pa + env(i), TOOLS, "anthropic") == "famAAAAAAAAA"
        assert resolve(st, pb + f"Today's date: 2026-10-{i + 1:02d}\n", TOOLS, "anthropic") == "famBBBBBBBBB"
        assert resolve(st, pc + AGENT2.replace("rule", f"rule v{i}") + env(i), TOOLS, "anthropic") == "famCCCCCCCCC"
    assert len(fams(st)) == 3
    row = st.q1("SELECT * FROM families WHERE id='famAAAAAAAAA'")
    assert row["legacy_prefix"] == pa and row["n_members"] == 5
    assert "Claude Code" in row["prefix"]  # display: the stable lines
    # the migrated state survives a restart
    st.close()
    st = Store(path)
    assert resolve(st, pa + env(9), TOOLS, "anthropic") == "famAAAAAAAAA"


# ------------------------------------------------------------------ Claude-Code-like prompts

CC_BODY = (
    "You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below "
    "and the tools available to you to assist the user.\n\n"
    "IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used "
    "maliciously.\n"
    "If the user asks for help or wants to give feedback inform them of the following:\n"
    "- /help: Get help with using Claude Code\n"
    "- To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues\n\n"
    "# Tone and style\n"
    "You should be concise, direct, and to the point. You MUST answer concisely with fewer than 4 lines of text "
    "(not including tool use or code generation), unless user asks for detail.\n"
    + "".join(f"- Guideline {i}: {w} carefully, keep changes minimal, and verify with the test suite before reporting.\n"
              for i, w in enumerate(["read", "search", "edit", "plan", "test", "commit", "review", "refactor"] * 12))
    + "# Following conventions\nWhen making changes to files, first understand the file's code conventions.\n"
    "Here is useful information about the environment you are running in:\n"
)
CLAUDE_MD = ("Contents of {cwd}/CLAUDE.md (project instructions, checked into the codebase):\n\n"
             "# Project notes\n- Run `make test` before committing.\n- Use ruff for linting; line length 120.\n"
             "- The API layer lives in src/api and must not import from src/cli.\n")
WORDS = "fix add refactor bump update remove tune rename clean docs tests parser cache proxy cli engine tree".split()


def cc_prompt(rng, n_status=None, n_commits=5, early_env=False):
    cwd = f"/Users/{rng.choice(['ana', 'bo', 'cy'])}/code/{rng.choice(['api', 'web', 'infra'])}-{rng.randint(1, 99)}"
    env_block = (f"<env>\nWorking directory: {cwd}\nIs directory a git repo: {rng.choice(['Yes', 'No'])}\n"
                 f"Platform: {rng.choice(['darwin', 'linux'])}\nOS Version: Darwin {rng.randint(20, 24)}.{rng.randint(0, 6)}.0\n"
                 f"Today's date: 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}\n</env>\n"
                 f"You are powered by the model named Opus. The exact model ID is claude-opus-{rng.randint(4, 5)}-{rng.randint(0, 9)}.\n")
    n_status = rng.randint(0, 40) if n_status is None else n_status
    status = "".join(f"{rng.choice(['M', 'A', 'D', '??', 'R'])} {rng.choice(['src', 'tests', 'docs'])}/{rng.choice(WORDS)}_{j}.py\n"
                     for j in range(n_status)) or "(clean)\n"
    commits = "".join(f"{rng.getrandbits(28):07x} {' '.join(rng.choice(WORDS) for _ in range(rng.randint(2, 9)))} (#{rng.randint(100, 9999)})\n"
                      for _ in range(n_commits))
    git = ("gitStatus: This is the git status at the start of the conversation. Note that this status is a snapshot "
           "in time, and will not update during the conversation.\n"
           f"Current branch: {rng.choice(WORDS)}-{rng.choice(WORDS)}\n\nMain branch (you will usually use this for PRs): main\n\n"
           f"Status:\n{status}\nRecent commits:\n{commits}")
    head = "You are Claude Code, Anthropic's official CLI for Claude.\n"
    if early_env:
        return head + env_block + CC_BODY + CLAUDE_MD.format(cwd=cwd) + git
    return head + CC_BODY + env_block + CLAUDE_MD.format(cwd=cwd) + git


def test_claude_code_like_prompts_one_family(tmp_path):
    rng = random.Random(7)
    st = Store(str(tmp_path / "cc.db"))
    ids = [resolve(st, cc_prompt(rng, early_env=bool(i % 2)), TOOLS, "anthropic") for i in range(12)]
    assert len(set(ids)) == 1
    prefix = fams(st)[0]["prefix"]
    assert "Tone and style" in prefix and "Today's date: <date>" in prefix and "Current branch" in prefix
    assert "(#" not in prefix  # commit messages are not stable


def test_large_varying_git_status_does_not_split(tmp_path):
    rng = random.Random(3)
    st = Store(str(tmp_path / "git.db"))
    sizes = [0, 300, 5, 800, 0, 120, 40, 600]
    ids = [resolve(st, cc_prompt(rng, n_status=n, n_commits=rng.randint(5, 40)), TOOLS, "anthropic") for n in sizes]
    assert len(set(ids)) == 1
    n_lines = st.q1("SELECT COUNT(*) c FROM family_lines")["c"]
    assert n_lines < 400  # git output collapses instead of piling up per file


def test_different_harness_with_same_tools_splits(tmp_path):
    rng = random.Random(1)
    st = Store(str(tmp_path / "h.db"))
    cc = {resolve(st, cc_prompt(rng), TOOLS, "anthropic") for _ in range(4)}
    other = {resolve(st, "You are a data-analysis agent.\n" + AGENT2 + STATIC[:6000] + env(i), TOOLS, "anthropic") for i in range(4)}
    assert len(cc) == 1 and len(other) == 1 and cc != other


# ------------------------------------------------------------------ mechanics


def test_normalize_masks_volatile_tokens_only():
    assert normalize("Today's date: 2026-09-26") == "Today's date: <date>"
    assert normalize("  Working directory:   /home/u/proj-3/sub ") == "Working directory: <path>"
    assert normalize("Is at C:\\Users\\bo\\x now") == "Is at <path> now"
    assert normalize("session 3f2a9c1e-1b2c-4d5e-8f90-a1b2c3d4e5f6 at 12:30:01") == "session <uuid> at <time>"
    assert normalize("HEAD is deadbee1 now") == "HEAD is <hex> now"
    assert normalize("OS Version: Linux 6.18.44-fc-v37") == "OS Version: Linux <n>-fc-v<n>"
    assert normalize("Today is Friday, September 26, 2026") == "Today is <day>, <date>"
    assert normalize("M src/api/routes.py") == "<git-status>" and normalize("?? build/") == "<git-status>"
    assert normalize("a1b2c3d Fix the parser (#123)") == "<git-commit>"
    assert normalize("Current branch: feat/k1-lines") == "Current branch: <branch>"
    # not over-masked: words with slashes, plain words that happen to be hex, instruction lines
    assert normalize("Use and/or when listing input/output options.") == "Use and/or when listing input/output options."
    assert normalize("A decade of facade effaced it") == "A decade of facade effaced it"
    assert normalize("Always read a file before editing it.") == "Always read a file before editing it."


def test_repeated_lines_keep_weight_data_lines_count_once():
    text = "Rule 1: be careful.\nRule 2: be careful.\n" + "M a.py\n" * 3 + "/x/y\n/z\n"
    ls = lines_of(text)
    assert [m for _, m, _ in ls] == ["Rule <n>: be careful.", "Rule <n>: be careful.", "<git-status>", "<path>"]
    assert len({k for k, _, _ in ls}) == 4


def test_tools_are_part_of_the_key(tmp_path):
    st = Store(str(tmp_path / "t.db"))
    p = IDENT + STATIC
    assert resolve(st, p, TOOLS, "anthropic") != resolve(st, p, [{"name": "Other"}], "anthropic")


def test_repeated_prompt_counts_once_and_is_sticky(tmp_path):
    st = Store(str(tmp_path / "s.db"))
    p = IDENT + env(1) + STATIC
    fid = resolve(st, p, TOOLS, "anthropic")
    for _ in range(5):
        assert resolve(st, p, TOOLS, "anthropic") == fid
    assert fams(st)[0]["n_members"] == 1
    # a fresh process (no memo, no cache) sees the same membership
    st2 = Store(str(tmp_path / "s.db"))
    assert resolve(st2, p, TOOLS, "anthropic") == fid
    assert fams(st2)[0]["n_members"] == 1


def test_deterministic_choice_and_second_process_sees_new_families(tmp_path):
    path = str(tmp_path / "p.db")
    a, b = Store(path), Store(path)
    fa = resolve(a, PREAMBLE + AGENT2, TOOLS, "anthropic")
    # b loaded the (empty) state before a created the family? it must reload, not create a twin
    fb = resolve(b, PREAMBLE + AGENT2 + "Today's date: 2026-01-02\n", TOOLS, "anthropic")
    assert fa == fb
    # two families that both qualify: the better-covering one wins, independent of insertion order
    x = STATIC[:3000]
    y = STATIC[:3000] + STATIC[3000:3400]
    for order in ((x, y), (y, x)):
        st = Store(str(tmp_path / f"d{len(order[0])}.db"))
        ids = {p: resolve(st, p, [{"name": "Z"}], "anthropic") for p in order}
        if ids[x] == ids[y]:
            continue  # y joined x's family (or vice versa): nothing to choose between
        probe = STATIC[:3000] + STATIC[3000:3350] + "Today's date: 2026-02-02\n"
        assert resolve(st, probe, [{"name": "Z"}], "anthropic") == ids[y]


def test_counts_stay_bounded_with_many_sessions(tmp_path):
    st = Store(str(tmp_path / "m.db"))
    rng = random.Random(5)
    ids = {resolve(st, cc_prompt(rng, n_status=rng.randint(0, 50)) + f"Session note: {rng.choice(WORDS)} {rng.choice(WORDS)} {i}\n",
                   TOOLS, "anthropic") for i in range(150)}
    assert len(ids) == 1
    row = fams(st)[0]
    assert 1 <= row["n_members"] < families.HALVE_AT
    assert st.q1("SELECT COUNT(*) c FROM family_lines")["c"] < 600


def test_resolve_is_fast(tmp_path):
    st = Store(str(tmp_path / "f.db"))
    rng = random.Random(11)
    for _ in range(10):
        resolve(st, cc_prompt(rng, n_status=200), TOOLS, "anthropic")
    p = cc_prompt(rng, n_status=200)
    t = time.perf_counter()
    resolve(st, p, TOOLS, "anthropic")  # new member: parse + score + write
    new_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    for _ in range(100):
        resolve(st, p, TOOLS, "anthropic")  # memo hit
    hit_ms = (time.perf_counter() - t) * 10
    assert new_ms < 200 and hit_ms < 2
