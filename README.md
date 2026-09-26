# treejit

An inference proxy with memory. treejit sits between an LLM agent harness and the model API. It learns a persistent execution tree from successful runs, and replays proven tool-call sequences without calling the model. The model is only called at the frontier: nodes the tree hasn't seen, branch points it can't resolve, and arguments it can't derive. For the last two, a small constrained call often suffices: it picks among known branches (T2) or fills in just the missing values (T3), instead of re-reading the whole conversation. Failed branches are recorded and avoided.

In short, it's a tracing JIT for agent trajectories, combined with a tree search whose tree persists across tasks.

```
harness ──► treejit ──► model API
               │  replayed step: tool call returned in ms, 0 tokens
               └─ frontier step: forwarded, recorded, learned from
```

The core (`src/treejit`, except `proxy.py`) uses only the standard library. Proxy mode also needs `httpx` and `uvicorn`.

**Nothing here has run against a real model yet** (there is no API key in the development environment). Every result below comes from a simulated agent or from an oracle agent on tau-bench, and the compatibility work for current Claude models is checked against documentation and fakes only. See [Compatibility](#compatibility-with-current-claude-models) and [Known limits](#known-limits).

## Contents

[Status](#status) · [Quickstart](#quickstart) · [Results](#results) · [How it works](#how-it-works) · [Replay safety](#replay-safety) · [Compatibility with current Claude models](#compatibility-with-current-claude-models) · [Decisions](#decisions) · [Known limits](#known-limits) · [CLI reference](#cli-reference) · [Configuration](#configuration) · [Development](#development)

## Status

States: **done** (implemented and tested offline), **opt-in** (done, off by default), **unverified live** (done, but only checked against documentation, fakes or a design note), **not yet**.

| Feature | State |
|---|---|
| Anthropic Messages and OpenAI Chat Completions dialects, JSON and SSE (replay and pass-through) | done |
| OpenAI Responses API (`POST /v1/responses`, inline `client.responses.create`, JSON and SSE, Codex argv shell args) | done for stateless requests; **unverified live** (built from the public reference, not a captured Codex trace); `previous_response_id` / `conversation` pass through unlearned |
| Current Claude models (Fable 5.1, Mythos 5.1, Opus 5.5): structured-output subcalls, sticky append-only hints, model-aware thinking drop | done, **unverified live** (`repro/X2_live_check.py` checks it once a key exists) |
| Trace recorder, system-prompt family keying (masked line sets), span-preserving shell tokenizer | done |
| Tree builder: anti-unification, provenance bindings, last-k-edge macros, root depth cap D | done |
| Failed replays as negative evidence, earned task-word rules with a similarity gate, per-input-class negatives, END as a choice | done |
| T0 replay / T1 guarded branch, postcondition side exits, confidence budget, hard cap K, batching | done |
| T2 choose / budget checkpoint, T3 hole filling (one small structured subcall, proxy and inline) | done |
| Read-only allowlist, commit points (incl. opaque executors), operator approval, `--not-commit`, repo-config taint | done |
| Contested commit points go to the model (T2) instead of a majority replay | done |
| Run identity (header, harness session id, derived; forks instead of extending finished runs), multi-turn episodes | done |
| Background rebuilds under the proxy, memoized builder, cross-process view reload | done |
| Frontier prefix compaction (first-sight, append-only; `epoch` and `window` modes) | **opt-in** (`compact = true`) |
| Frontier hints on T4 calls (sticky, append-only) | done (`hints = "failures"` by default) |
| Inline mode, including streaming (`stream=True`, Anthropic `messages.stream()`) | done for sync clients; async clients: use the proxy |
| CLI (`serve show runs explain outcome pending pin approve revoke prune export build stats`), HTML / Mermaid / SKILL.md export | done |
| Benchmark: synthetic suite (inline and real-proxy modes), learning-curve report, cost model (`--payload`, `--cache`) | done |
| tau-bench runner (`--suite taubench`, oracle-with-noise agent, tau-bench's own reward) | done |
| tau-bench with a real model (`--agent claude`) | **not yet run** (ready; needs `ANTHROPIC_API_KEY`) |
| Macros-as-tools | **not yet** (deferred, PLAN E5: measured headroom too small) |
| Incremental (dirty-node) rebuild | **not yet** |

## Quickstart

```bash
pip install -e '.[proxy]'          # core is zero-dependency; [proxy] adds httpx + uvicorn
treejit serve --port 8787          # db: ./treejit.db (or --db / $TREEJIT_DB)
```

**Claude Code**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

No run header is needed: Claude Code's session id (in `metadata.user_id`) plus the episode index names each task's run, and every prompt after a finished answer starts a new episode (see [Runs and episodes](#runs-and-episodes)).

**OpenAI-compatible harnesses (Chat Completions):** point `OPENAI_BASE_URL` at `http://127.0.0.1:8787/v1`.

**Codex-style harnesses (OpenAI Responses API)**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1     # treejit forwards /v1/responses to openai_upstream
codex                                                # or any client of POST /v1/responses
```

treejit learns from and replays *stateless* Responses traffic, where every request carries the whole conversation in `input` (as Codex CLI does with `store: false`, according to `repro/R1_responses_design.md`; not yet checked against a captured Codex trace). The run is named from `prompt_cache_key` plus the episode. Codex's `shell` tool sends argv lists (`{"command": ["bash", "-lc", "git status"]}`): policy and templates read the script inside `bash -lc` / `sh -c`, and any other argv as `shlex.join(argv)`, so read-only commands replay without approval as the string form does. Requests that chain on the server (`previous_response_id`, `conversation`) are forwarded untouched and never learned from (tier `pass`, note `stateful`), because a replayed response id (`resp_tj...`) would be unknown upstream.

**Inline mode** (harnesses you own, tests)

```python
from treejit import TreeJIT
jit = TreeJIT("treejit.db")
client = jit.wrap(anthropic.Anthropic())             # or openai.OpenAI(), or a callable body -> dict
client.messages.create(..., extra_headers={"X-TreeJIT-Run": "task-17"})
with client.messages.stream(..., extra_headers={"X-TreeJIT-Run": "task-17"}) as s:
    msg = s.get_final_message()
jit.outcome("task-17", "pass")
```

`jit.wrap(openai.OpenAI())` wraps both `chat.completions.create` and `responses.create`; a plain callable takes `dialect="anthropic" | "openai" | "responses"`. Streaming takes the same path as JSON. A replay returns a `ReplayStream` (the replay's events, served locally, as SDK event models when the SDK is installed and as dicts otherwise). A forward returns a `TeeStream`: the upstream events, unchanged, recorded (usage, run, END) when the stream is exhausted; closing it before the stop reason records status 499 and no END. Anthropic's `messages.stream()` is the SDK's own `MessageStreamManager` fed by those streams (a minimal shim without the SDK). `X-TreeJIT-Run` never goes upstream. Everything else on the client (`messages.count_tokens`, `batches`, `beta`, `with_raw_response`, ...) is the real client's and is not recorded.

**Outcomes.** Only runs reported as passing ever promote an edge.

```bash
curl -s localhost:8787/outcome -d '{"run_id": "latest", "outcome": "pass"}'
treejit outcome <run_id|latest> fail --reason "tests failed in CI"
```

In Claude Code, a `Stop` hook that runs your verifier and then posts `/outcome` closes the loop. Outcome `error` (timeouts, 429s) is recorded but never counted as evidence.

**Rebuilds.** Each outcome rebuilds its family's tree. Under `treejit serve` the rebuild always runs on a background worker (even with `rebuild = "sync"`), so an outcome never stalls other requests; outcomes that arrive during a build are coalesced. `/outcome` answers once the tree is rebuilt (post `"wait": false` to get the answer as soon as the outcome is recorded). Inline mode and the CLI rebuild synchronously, which keeps benchmark numbers deterministic; `jit.wait_rebuilds()` waits for pending background builds. A running proxy also notices a rebuild made by another process (the CLI, a second instance on the same db) through the family's `built_at`. Operator changes (approve, revoke, `--not-commit`, pin, unpin, eviction) take effect even while a rebuild is running: a build whose operator state changed before it wrote is discarded and run again, and an outcome recorded mid-build leaves the family dirty. A tree built by another treejit version is not served until rebuilt, so the first requests after an upgrade go to the model.

**Operating it.** `treejit show --ids` prints the tree, `treejit pending` lists edges held back only by policy, `treejit approve --review` walks that queue, and `treejit explain <run|latest>` shows who decided each step of a run. See the [CLI reference](#cli-reference).

## Results

All numbers in this section were measured at the same commit, with the commands given. The model is simulated (synthetic suite) or an oracle (tau-bench): treat absolute numbers as illustrative, and the comparisons between modes as the result.

Terms: *full calls* are T4 calls (the whole conversation). *Small calls* are T2/T3 subcalls (a short prompt and a structured answer); their tokens are included in *tokens / task*. *Served* is the share of tool calls that no full model call produced (T0/T1/T2/T3 steps). Modes: *plain agent* (`baseline`), *read-only allowlist* (`treejit`: only read-only steps replay unattended), *edges approved* (`treejit+ok`: `treejit approve '*'`, simulating operator review of write steps and commit points), and *+ compaction* (`treejit+ok+compact`).

### Synthetic suite

A simulated agent works a mixed stream of coding tasks (typo fix / version bump / delete module, each with a flaky-test branch) and tau-bench-style retail tasks (branching on order status). The tree starts empty. The simulated model sees only the conversation, is stochastic (argument formatting varies, free-form commit messages), and takes a known-bad shortcut 6% of the time. Observations are realistically sized (a 45–65-line `Read`, 150–420 pytest progress rows, untracked build junk in `git status`). Tokens ≈ prompt chars / 4; latency = 600 ms + 15 ms per output token.

Seed 0 (`python -m treejit_bench --tasks 200 --seed 0 --modes baseline,treejit,treejit+ok,treejit+ok+compact`):

| mode | tasks | full calls / task | small calls / task | tokens / task | served | success | sim. wall-clock / task |
|---|---|---|---|---|---|---|---|
| plain agent | 151–200 | 6.02 | – | 6,196 | 0% | 96% | 10.0 s |
| read-only allowlist | 151–200 | 3.70 | 0.14 | 4,932 | 48.5% | 100% | 6.2 s |
| edges approved | 41–50 | 1.10 | 0.80 | 2,152 | 98.4% | 100% | 2.4 s |
| edges approved | 151–200 | **1.02** | 0.92 | 2,000 | **99.6%** | 100% | 2.3 s |
| edges approved + compaction | 151–200 | **1.02** | 0.92 | **1,848** | **99.6%** | 100% | 2.3 s |

Seeds 0–5 (same command with `--seed S`). Success is over all 200 tasks; the other columns are tasks 151–200. Each cell lists plain agent / allowlist / approved / approved + compaction (compaction doesn't change calls or served share, so those columns list three values):

| seed | success (of 200) | full calls / task | small calls / task | tokens / task | served % |
|---|---|---|---|---|---|
| 0 | 193 / 198 / 200 / 200 | 6.02 / 3.70 / 1.02 | – / 0.14 / 0.92 | 6,196 / 4,932 / 2,000 / 1,848 | 0 / 48.5 / 99.6 |
| 1 | 189 / 199 / 200 / 200 | 6.36 / 4.26 / 1.06 | – / 0.06 / 0.30 | 6,684 / 5,342 / 1,531 / 1,378 | 0 / 39.6 / 98.9 |
| 2 | 192 / 198 / 200 / 200 | 6.32 / 4.12 / 1.04 | – / 0.00 / 0.80 | 6,711 / 5,423 / 1,972 / 1,775 | 0 / 38.6 / 99.2 |
| 3 | 189 / 198 / 200 / 200 | 6.26 / 4.20 / 1.06 | – / 0.10 / 0.94 | 6,590 / 5,554 / 2,031 / 1,861 | 0 / 41.8 / 98.9 |
| 4 | 185 / 198 / 199 / 199 | 6.28 / 4.08 / 1.06 | – / 0.10 / 1.20 | 6,744 / 5,652 / 2,337 / 2,165 | 0 / 44.8 / 98.9 |
| 5 | 192 / 199 / 200 / 200 | 6.66 / 4.62 / 1.02 | – / 0.06 / 0.06 | 7,401 / 6,203 / 1,484 / 1,262 | 0 / 36.9 / 99.7 |

- **Invariant check (PLAN): holds on every seed.** Both treejit modes succeed at least as often as the plain agent. Every failure in a treejit mode is the simulated model's own shortcut ("pushed without a passing test run") at a step the model decided (tier `M` in `results.csv`), never at a replayed step. Because the model is stochastic and replay changes how many draws it makes, a treejit mode can fail a task the plain agent passed (allowlist mode: 1–2 tasks per seed; edges approved: seed 4 task 0, where the tree is still empty); the plain agent fails 8–15 per seed.
- With edges approved, almost every tool call is served from about task 40 on; the only full call left is usually the final answer. The floor is one full call per task, because the final answer is always generated.
- With the default read-only allowlist, only read steps replay, and T2/T3 rarely apply (their options must be replayable too).
- Small calls are mostly T3 fills (the free-form commit message, `Edit` strings) and budget checkpoints. Few fall back to T4: at seed 0 with edges approved, 5 of 132 small calls failed (`failed:` in the request note).
- Seed 3 used to regress (171/200 with edges approved) because a chance task-word rule (`task~src → git rm`) misrouted typo tasks. Decisions 11–16 fixed it; seed 3 is 200/200.
- Running the same stream through the real ASGI proxy with SSE streaming (`--via-proxy`) gives identical numbers (checked at seed 0 with `--via-proxy --modes baseline,treejit,treejit+ok`).

Report: [`docs/learning_curve.html`](docs/learning_curve.html) (seed 0; model calls, tokens, replay share and small calls vs task index, with a table view). Summary: [`docs/bench_summary.json`](docs/bench_summary.json).

**Compaction.** Over all 200 tasks, `treejit+ok+compact` (first-sight) vs `treejit+ok`; trajectories (success, calls, tiers) are identical task for task on every seed:

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| tokens / task, off → on | 2,154 → 1,986 | 1,815 → 1,674 | 2,271 → 2,098 | 2,287 → 2,129 | 2,276 → 2,118 | 1,932 → 1,752 |
| change | −7.8% | −7.8% | −7.6% | −6.9% | −7.0% | −9.3% |
| compacted chars / task | 624 | 523 | 618 | 583 | 585 | 660 |

### Cost model: harness payload and prompt caching

In the sim a small call costs about half a full call (seed 0, edges approved, tasks 151–200: 696 vs 1,333 tokens), because the simulated system prompt, tools and conversation are tiny. A real harness sends far more per full call. `--payload none|tau|claude-code` adds a virtual harness system prompt plus tool schemas to every **full** call (0 / ~5k / ~24k tokens); T2/T3 subcalls are built by treejit and never carry it. `--cache` models Anthropic prompt caching (breakpoints after the static prefix, after the system prompt, and at the last message; a request reads the longest prefix an earlier one wrote). `cost_tokens` is in input-token equivalents: uncached input 1×, cache write 1.25×, cache read 0.1×, output 5× (`--cost-weights`). The default output is unchanged.

Seed 0, tasks 151–200, cost / task (`--payload P [--cache]`, same trajectories as above):

| scenario | plain agent | read-only allowlist | edges approved | cut (approved) | small-call cost / full-call cost |
|---|---|---|---|---|---|
| `--payload tau` | 37,989 | 24,475 | 7,411 | −80% | 0.12 |
| `--payload claude-code` | 152,369 | 94,775 | 26,791 | −82% | **0.03** |
| `--payload claude-code --cache` | 18,032 | 11,672 | 4,552 (4,362 with compaction) | −75% (−76%) | 0.22 |

With a Claude Code-sized payload a small call costs about 3% of a full call. With caching, full calls get about 8× cheaper, because most of the prompt is a cache read; small calls then cost about a fifth of a full call. T2/T3 still pay off, by less.

### tau-bench

`bench/src/treejit_bench/taubench.py` runs [tau-bench](https://github.com/sierra-research/tau-bench) tasks through treejit inline mode and scores them with tau-bench's own `Env.calculate_reward` (the database hash after the episode must equal the ground truth's, and every expected output must appear in a reply). tau-bench isn't on PyPI; its environments need only `pydantic`, and a `litellm` stub is inserted when it isn't installed (the user simulator is never called).

```bash
git clone --depth 1 https://github.com/sierra-research/tau-bench && pip install pydantic
export TAUBENCH_PATH=$PWD/tau-bench        # tests/test_taubench.py skips without it
python -m treejit_bench --suite taubench --tau-env retail --tau-split test --modes baseline,treejit,treejit+ok --out tau_out
#   --tasks N --tau-start I     a slice (default: the whole split; retail test 115, train 500, dev 20; airline test 50)
#   --noise P                   the oracle's per-write slip probability (default 0.05)
#   --agent claude [--claude-model M] [--tau-user confirm]    a real model (needs ANTHROPIC_API_KEY and anthropic)
#   --rebuild-every K           rebuild the tree after every K-th outcome (faster; the last K-1 runs aren't visible yet)
```

- **OracleAgent.** A deterministic agent: a canonical read prefix (retail: `find_user_id_by_*`, `get_user_details`, `get_order_details` per order, `get_product_details` for new items; airline: `get_user_details`, `get_reservation_details`), then the task's ground-truth actions, then a final answer with the expected outputs. With probability `--noise` per write, it *slips* (a wrong reason, a dropped item, a wrong payment method). Slips depend only on (seed, env, split, task index), so every mode sees the same mistakes. The oracle answers T2/T3 subcalls with the same policy. With noise 0 it scores 1.0 on every task of retail test, retail train and airline test.
- **ClaudeAgent** (`--agent claude`) sends the same bodies to a real model through the Anthropic SDK (through `jit.wrap` in treejit modes), with a cache breakpoint on the system prompt and cache usage accounted. Real models ask before writes, so use `--tau-user confirm` (answers up to 4 questions with "Yes, I confirm."; episodes stay one run, see [Runs and episodes](#runs-and-episodes)). Tested only against a fake SDK client.
- The tools are tau-bench's `tools_info` as Anthropic tools, the policy wiki is the system prompt (~5k tokens with the tool schemas, in every full call; no extra payload), and the task instruction is the first user message.

**Results.** Oracle, seed 0, noise 0.05, `--rebuild-every 1` (the default). Command: `python -m treejit_bench --suite taubench --tau-env E --tau-split S --modes ...`. Cost is in input-token equivalents without the cache model.

| split | mode | tasks | full calls / task | small calls / task | tokens / task | cost / task | served | reward |
|---|---|---|---|---|---|---|---|---|
| retail test | plain agent | 1–115 | 7.75 | – | 41,627 | 43,938 | 0% | 0.939 |
| retail test | read-only allowlist | 1–115 | 4.04 | 3.33 | 29,404 | 31,000 | 66% | 0.939 |
| retail test | edges approved | 1–115 | 3.64 | 3.74 | 27,655 | 29,179 | 70% | 0.939 |
| retail test | edges approved + compaction | 1–115 | 3.64 | 3.74 | 27,425 | 28,949 | 70% | 0.939 |
| retail test | edges approved | 66–115 | 2.92 | 3.80 | 23,239 | 24,579 | 77% | 0.900 (plain agent 0.900) |
| retail train | plain agent | 1–300 | 6.94 | – | 36,684 | 38,737 | 0% | 0.940 |
| retail train | read-only allowlist | 1–300 | 3.54 | 2.09 | 23,802 | 25,051 | 61% | 0.940 |
| retail train | edges approved | 1–300 | 2.44 | 3.27 | 19,273 | 20,300 | 78% | 0.940 |
| retail train | edges approved | 251–300 | 2.02 | 3.16 | 16,127 | 17,004 | 83% | 0.920 (plain agent 0.920) |
| airline test | plain agent | 1–50 | 5.20 | – | 25,409 | 27,036 | 0% | 0.980 |
| airline test | read-only allowlist | 1–50 | 3.48 | 1.16 | 19,294 | 20,543 | 41% | 0.980 |
| airline test | edges approved | 1–50 | 3.30 | 1.88 | 19,408 | 20,669 | 45% | 0.980 |

Commands: retail test `--tau-env retail --tau-split test --modes baseline,treejit,treejit+ok,treejit+ok+compact`; retail train `--tau-env retail --tau-split train --tasks 300 --modes baseline,treejit,treejit+ok`; airline `--tau-env airline --tau-split test --modes baseline,treejit,treejit+ok`.

- **Invariant check (PLAN): holds on every split, task for task.** Every treejit mode fails exactly the plain agent's tasks, which are the oracle's slips (retail test: 7 tasks; retail train 300: 18; airline test: 1; ids in each run's `results.csv`). No treejit mode fails a task the plain agent passes. With noise 0 the plain agent scores 1.000 on retail test 115, retail train 500 and airline test 50.
- **The B1 misroutes are gone.** Before decision 21, edges-approved runs failed retail train tasks 149, 191, 197, 238, 243, 245, 298 (later 160) and airline test task 29, which the plain agent passes: T0/T1 replayed an irreversible `cancel_pending_order` into tasks that wanted a *modify* (`treejit explain tau-retail-train-0-238` showed `T1@r6 cancel_pending_order(...) conf=0.70`, from a rule `json.status == "pending" → cancel` right in 4 of 5 examples), or `cancel_reservation` into a read-only airline task. All of these tasks now pass.
- Small calls are frequent (1.2–3.8 per task): most retail steps have their structure decided, but no binding produces a value this input needs (order id, item ids, reason), so a T3 fill asks for it, and contested commit points get a T2 call. A small call costs 0.13–0.20 of a full call here (`small/full cost` column of the runner), because every full call carries the ~5k-token wiki and tools.
- On airline, edges approved spends slightly more tokens than the allowlist (19,408 vs 19,294 per task) for 0.18 fewer full calls and 0.72 more small calls: on a split this small, the small calls cost about what the saved full calls do.
- Compaction saves 0.8% of tokens on retail test (775 compacted chars per task): most frontier calls follow few replayed steps.
- **Run time.** With the default `--rebuild-every 1`, each treejit mode took about 2.5 minutes on retail test 115 and 12 minutes on retail train 300 (4 runs in parallel on 4 cores), against 40 s and 91 s for the plain agent. `--rebuild-every K` is faster but changes the learning dynamics (the last K−1 runs aren't visible yet).

## How it works

| module | job |
|---|---|
| `dialects.py` | Anthropic Messages, OpenAI Chat Completions, OpenAI Responses. Parse requests into an *episode* (task + (call, observation) steps, later user turns as user steps) and read harness session ids. Build replay responses as JSON or SSE. Accumulate upstream SSE for usage. Insert hints at a history position. Classify models by how `thinking` behaves. |
| `families.py` | Tree key = tool schemas + the stable lines of the system prompt. Lines are masked (dates, paths, hashes, uuids, URLs, numbers; git status and log lines collapse), hashed and counted per family. A line is stable when ≥80% of the family's distinct prompts have it. A prompt joins a family when it has ≥90% of the family's stable characters and the stable lines make up ≥80% of its own. Volatile lines (an early `<env>`, a large `gitStatus`) don't split a family; a different agent sharing a short preamble doesn't merge. Ids never change. |
| `shellwords.py` | Span-preserving shell tokenizer (quotes, `$(...)`, heredocs, operators, full ANSI-C `$'…'` decoding). Replayed commands are spliced into the original text, so quoting survives. |
| `policy.py` | What replay may emit unattended: the read-only allowlist, commit points and repository taint (see [Replay safety](#replay-safety)). |
| `templates.py` | Edge = tool + arg keys + per-segment command heads (`git commit`, `npm test`). Calls with one shape are anti-unified per token into constants and variables. argv shell arguments are templated on their command text and re-wrapped on render. |
| `bindings.py` | Provenance search. Each variable binds to `$task` / `$obs[-k]` / `$arg[-k]` via extractors (JSON path, `key: value`, regex types, after-word, line, token), `fmt` templates (`Bump version to {$task.version}`) or `case` (value chosen by predicates). A variable with no rule is a hole. |
| `features.py` | Predicate set: error/exit code, empty, JSON field equals, substring, task keyword. Guards are conjunctions of stable features; branches are decision lists with per-rule leak and support. |
| `builder.py` | Deterministic, memoized rebuild from the trace log (bounded to the last `max_runs` runs): promotion, bindings, guards, postconditions, credit assignment, tombstones, tiers, stored commit reasons. |
| `tree.py`, `replay.py` | Stateless recognition (root path ≤ D, then last-k n-grams). Climbs T0 → T1 → T3/T2 → T4, bounded by budget, cap, allowlist, commit points and taint. Returns T2/T3 opportunities as `plan.sub` and never calls a model itself. |
| `subcalls.py` | Builds the T2/T3 subcall and parses its answer. |
| `compaction.py` | Opt-in frontier prefix compaction and sticky frontier hints. |
| `engine.py`, `proxy.py`, `inline.py` | The engine (run identity, recording, rebuild scheduling) and the two entry points. |
| `cli.py`, `export.py`, `operate.py` | CLI, views, and operator tools (approval queue, run timelines, short ids). |

Each tree edge has a tier: **hot** (replayable), **live** (promoted but blocked), **warm** (one passing run), **cold**, or **tomb**. `show` prints why a live edge doesn't replay: `LIVE:holes` (an argument has no binding), `LIVE:needs_approval` (not read-only, or a commit point), or `LIVE:commit_point_needs_evidence` (approved, but a commit point also needs `promote_runs + 1` passing runs). Only the last two appear in `pending`.

### T2 and T3

`TreeJIT.handle()` may return `Result(kind="subcall")`: a small request body in the client's dialect. The transport (proxy or inline) sends it upstream non-streaming with the client's auth headers, then calls `jit.resume(result, response_json, status)`, which returns a normal replay (as SSE if the client streamed) or the T4 forward. There is at most one subcall per incoming request, only for the first call of a response. Any failure falls back to exactly the T4 forward the request would have had: an HTTP error, no answer, missing or empty values, "something else", or a value that fails the safety re-checks.

- **T3 (fill).** T0/T1 picked an edge that may replay, but some variables have no rule, or their rule abstains for this input. The subcall shows the task, the calls so far (arguments truncated to 200 chars), the last 3 tool results (2,000 chars each), and the next call with `<placeholders>`. The model answers one string per hole, or `not_this_step`. Values are spliced in as data (one quoted shell word), then the call is re-rendered and re-checked: a read-only call must stay read-only, its commit reason must equal the one stored for the edge (so `sh -c 'echo a'` can't become `sh -c 'git push'`, nor `make test` become `make deploy`), and it must keep the edge's shape.
- **T2 (choose).** Used at an ambiguous node (at least 2 children the model has chosen, the replayable ones at least 50% of those choices), at a contested commit point (decision 21), for an unproven task-word rule (decision 15), and as a *checkpoint* when the confidence budget runs out on an otherwise confident step. The model answers an option number, or 0 for "something else" (→ T4). If the chosen option has holes, the same call asks for their values. A T2 step restarts the confidence budget; the hard cap K still counts it.
- **Wire shape.** Anthropic: structured outputs (`output_config.format`, a JSON schema with `additionalProperties: false` and `enum`s); the answer is a JSON text block, parsed tolerantly (thinking blocks, code fences, stray prose), and anything that doesn't fit the schema falls back to T4. `thinking` is never sent. `output_config.effort` is `subcall_effort` on models that take it; where thinking is on by default, `max_tokens` is at least 4096. `subcall_format = "tool_auto"` sends one `strict` tool with `tool_choice: auto` instead (no call → T4); a forced tool is used only for legacy `claude-3*`/`claude-2*` models. OpenAI Chat Completions and Responses use a forced function call.
- Subcalls use `small_model` if set (else the request's model). They are logged in `requests` with tier `T2` / `T3` and their own usage, and they carry the run id. Subcall steps are marked in the call id (`..._t3`, `..._t2`, `..._ck`), so recognition, side exits and the budget stay stateless.

### Runs and episodes

An **episode** is one task: every turn of a conversation since the last task boundary (`dialects.episode_of`, the same rules for every dialect). The task is the episode's first user text. A later user turn either continues the episode or starts a new one:

- **Continues, as a `steer` step**, when the user cut in while the agent was working: text next to tool results, an interrupt (`[Request interrupted by user…]`), or a user message with no finished agent turn since the previous one.
- **Continues, as a `yes` / `no` / `text` step**, when the agent's turn ended by asking something (a question that isn't a generic closer, or a confirmation prompt). A bare "ok" or "no thanks" after a turn that asked nothing also continues, as `text`.
- **Starts a new episode** otherwise. Claude Code's pattern, one conversation with several prompts, is several episodes. Claude Code's local slash-command transcripts are dropped from the text.

`episode_mode` (config) or the `X-TreeJIT-Episode` header overrides this: `conversation` makes the whole conversation one episode (tau-bench), `turn` makes every user message a new one.

A **user step** is a pseudo-call `$user:<kind>` whose observation is the user's text. It gets an edge like any call, so "after the user said yes" is a context; bindings and features read it like any observation. No user edge is ever replayable. When the user answers a finished agent turn, the agent's text reply before it is END evidence at that context. `yes` and `no` are different edges, so a write learned after "yes" is never proposed after "no".

A **run** is one episode of one conversation. Its id is, in priority order:

1. `X-TreeJIT-Run: H`: `H`, or `H.<task hash>` for another task under the same header.
2. A harness session id: Claude Code's `metadata.user_id` (`…_session_<uuid>`, or a JSON object with `session_id`), the `X-Claude-Code-Session-Id` header, or OpenAI's `prompt_cache_key`. The id is `r_` + hash(family, session, first user text, episode index, task hash).
3. Otherwise hash(family, first user text, the conversation's first assistant turn, episode index, task hash). Weak call ids (`call_0`, `toolu_01`, counter-like) are salted with the first observation, so such a conversation's first request gets its run id one request later. OpenAI's `user` is mixed in but never names a run alone.

A run whose recorded steps the conversation doesn't continue is someone else's, so the next id is tried (`<id>.2`, …). **A run with an outcome is never extended:** a conversation that goes on after its outcome continues in a fork `<id>.2`, whose copied steps (`runs.inherited`) are context only. An outcome posted for `H` also reaches its forks that have none yet.

### Frontier prefix compaction (opt-in)

With `compact = true`, a request that goes to the model (T4) is forwarded with the raw observations of *verified replayed steps* replaced by a short deterministic digest. Only the forwarded copy changes; the harness's history and treejit's trace keep the full text.

```
[treejit: replayed & verified step — Bash(python -m pytest -q) → ok, 21 lines, 1001 chars; first line: "…… [ 34%]"; last line: "233 passed, 5 warnings in 6.93s"]
[output elided by treejit; call the tool again if you need it]
```

Rules:

1. **Eligible** only if the call id carries the replay marker, the edge is known at the node encoded in the id, the observation satisfies the edge's (non-empty) learned postcondition, and its features show no error. Model-chosen, side-exited and errored steps are always sent in full.
2. **Keep the last few.** The last `compact_keep_last` (3) observations go in full; observations under `compact_min_chars` (400) are left alone.
3. **Keep what the next decision reads**: an observation read by a binding, guard or decision list of any child of a current frontier context. (The older "keep what an earlier step's bindings read" rule protected nothing and is off; `compact_keep_path = true` restores it.)
4. **Append-only (`compact_mode = "first_sight"`, the default).** A step may be compacted only in the *first* forwarded request that contains it; once sent in full it stays full, and once compacted it stays compacted, across rebuilds and restarts (decisions are stored in `compactions`, or reconstructed from the conversation when missing). Consecutive forwards of one conversation are byte-identical up to the previous request's last message, so the provider's prompt cache keeps the whole prefix. In practice this compacts *bursts*: more than `compact_keep_last` steps replayed between two frontier calls.
5. **`compact_mode = "epoch"`** additionally re-compacts the whole prefix when the conversation's previous forward is older than `compact_epoch_ttl` (the cache went cold). **`"window"`** (legacy) moves the keep-last window on every request; it sends the fewest raw tokens but changes an earlier message on every forward, so it only suits providers without prompt caching. Neither is compatible with preserved thinking (see [Compatibility](#compatibility-with-current-claude-models)).

Each forwarded request records `compacted N obs/C chars` in its note. Compaction decisions and hints are kept indefinitely, because a conversation can be resumed at any time and removing a decision would change its forwarded history. `treejit prune --compact-days N` drops the compaction decisions of runs idle N days on request; a conversation resumed after that may be sent a different history. `compact_retention_days` only governs epoch-mode timestamps.

**Compaction and the prompt cache.** `tests/cache_model.py` bills request sequences the way the Anthropic prompt cache does (reads 0.1×, writes 1.25×, a breakpoint after `system` and one at the end, 5-minute TTL), driving the real engine. Billed input tokens against compaction off (`PYTHONPATH=src:bench/src python3 repro/C1_after.py`, output in `repro/C1_after_small.txt`):

| pattern (15 steps, ~3.3k-char observations) | window | first_sight | epoch |
|---|---|---|---|
| dense: forwarded after every replayed step | +119% (+43% with a 62.5k-char system prompt) | ±0 | ±0 |
| interleaved: every other step is the model's | +76% (+20%) | ±0 | ±0 |
| bursty: 4 replayed steps between frontier calls | +9% (+3%) | **−20%** (−8%) | −20% (−8%) |
| interleaved, a 10-minute pause before forward 6 | +15% | ±0 | **−27%** |
| bursty, a 10-minute pause before forward 2 | −28% | −20% | **−45%** |

First-sight never bills more than compaction off: the bodies are the same sequence with some observations shorter, and every prefix is preserved. `tests/test_compaction.py` checks this for all patterns and both breakpoint placements. On the synthetic suite with edges approved (`repro/C1_bench_billed.py --seed S`, each task's full calls billed as one conversation), first-sight bills −11.8% to −14.3% against compaction off on seeds 0–5, and window −10.9% to −13.1%; with a Claude Code-sized system prompt (`--bigsys`) both save under 1%. Trajectories are identical in all three modes.

**Frontier hints** (`hints = "failures"`: on a T4 call at a node with tombstoned or failing children, a short note of what failed and what worked there) are sticky: each is stored under an anchor (a chained hash of the system prompt, tool names and the history up to the item it follows) and re-inserted, byte-identical, at the same position in every later forward. A hint is given only at a history position that no earlier forward passed without one (every forward records "no hint" at the positions it sends); the first decision wins, so a conversation's forwarded history never changes because of another conversation. Consecutive forwards are append-only with hints on (`tests/test_hints.py`), and a hint costs at most its own tokens in the cache model.

## Replay safety

Replay emits a call with no model call and no human in the loop only if `policy.is_readonly` says it is read-only. Otherwise the edge needs operator approval (`treejit approve`). A commit point (`policy.commit_reason` is non-empty: `git push`, `curl`, `send_*`, an interpreter, a script, ...) needs approval **and** `promote_runs + 1` passing runs, and a contested one goes to the model (below). For shell commands:

- **One view of the program.** `shellwords.unwrap` skips `VAR=val` assignments and transparent wrappers with their own options (`env`, `sudo`, `time`, `nohup`, `exec`, `command`, `nice`, `timeout N`, `stdbuf`). Edge shapes, the read-only check and commit-point detection all use it, so `env git push origin main` has the shape `git push` and is a commit point. An unrecognised wrapper option makes the command not read-only; `sudo` never counts as read-only.
- **Allowlist, not denylist.** Every simple command must run an allowlisted program (`ls`, `cat`, `grep`, `head`, `wc`, `jq`, `diff`, ...), named literally (no `$VAR`, globs, brace expansion or `$'\…'`), as a bare name or from a standard bin directory. Assignments are limited to harmless variables (`LC_*`, `LANG`, `TZ`, `NO_COLOR`, `PAGER=cat`, ...), which keeps out `LD_PRELOAD`, `PATH`, `GIT_EXTERNAL_DIFF`.
- **Per-program argument checks** accept only options they know, with literal arguments. They keep out `sort -o`, `uniq IN OUT`, `date -s`, `tree -o`, `yq -i`, `fd -x`, `rg --pre`, `bat --pager`, `find -exec*`/`-delete`/`-fprint*`, and so on. `git` global options are limited (no `-c`, `--config-env`, `--exec-path`); the subcommand must be a read with no `--output`; `branch`/`tag` only when listing, `config` only when reading, `remote` only bare or `show`/`get-url`, `stash` only `list`/`show`.
- **sed and awk**: only a conservative subset is read-only (sed without `w W r R e a i c y`, labels, `-i`, `-f`; awk with only `-F`/`-v` and no `system`, `getline`, pipes or output redirection). Anything else needs approval.
- **Redirections** may only read, duplicate a descriptor, or write to `/dev/null`, `/dev/stdout`, `/dev/stderr`. Command and process substitution make a line not read-only.
- **Excluded by default:** `sudo`, `less`/`more`, `xargs`, `tee`, shells, `eval`, shell keywords. `extra_readonly_commands` opts a program in (with any arguments; the path, assignment and redirection rules still apply).
- **The tokenizer cooks words as bash does.** `$'...'` is decoded in full and `$"..."` cooks like `"..."`, so `$'\147it' push` has the shape `git push`. Leading reserved words are skipped when locating the program.
- **Commit points** are matched only where a program runs: each simple command's program (after reserved words and wrappers), the command that `xargs`, `parallel`, `watch`, `strace`, `flock`, ... run, `find -exec*`, `python -m MODULE`, and nested scripts (`$(...)`, backticks, `sh -c`, `eval`, `env -S`, quoted text a shell in the line may run). Quoted text elsewhere is data (`git commit -m "then git push"` is not a commit point; `apt install curl`, `git stash push` aren't either). In a command that isn't read-only, commit points are:
  - **configured patterns** (`commit_commands`), matched from the program; git patterns from git's subcommand, found by parsing its global options (`git -C repo --no-pager push`);
  - a **non-literal program or git subcommand** (`$CMD x`, `git $x`, `{git,push}`, `git${IFS}push`);
  - **git** `push`, `send-pack`, `http-push`, `svn`, `p4`, `send-email`, ...; any non-builtin subcommand (aliases, `git-*` extensions); `-c`/`--config-env`/`--exec-path` except harmless keys; `rebase --exec`, `bisect run`, `submodule foreach`, `filter-branch`, `hook`;
  - **opaque executors**: interpreters (except `-m` with a local module such as `pytest`, and `--version`), shells and `source`, script paths (except tool bins such as `.venv/bin/pytest`), task runners and package scripts (`make`, `just`, `gradle`, `npm run`, `npx`, `cargo run`, `go run`, ...) unless every target is a local word (`test`, `lint`, `build`, ...); `uv run`/`poetry run`/`bundle exec CMD` classify CMD;
  - **remote writers**: `gh` except read verbs and `gh api` GETs; `curl`/`wget` except a plain GET/HEAD of a loopback URL; cloud and cluster CLIs unless a read verb and no write verb (`kubectl get pods`); HTTP, mail and socket clients.
  - **Deliberate exceptions** (still commit points): dry runs, local `rsync`, `npm run <name>` with a non-local name. Known-local runners (`pytest`, `tox`, `npm test`, `cargo test`/`build`, `go test`, `make test|lint|build`) are not commit points, though not read-only either.
- **The reason is stored.** The builder stores `commit_reason` per (node, edge); `materialize` re-renders every call it emits and rejects it when the reason differs, so a T3 value can't change a call's commit status.
- **Contested commit points go to the model** (decision 21). A commit point where the model has also chosen something else after the same history (a sibling edge, or ending the episode, at any context of the step) never replays on a majority (T0). A decision-list rule (T1) may pick it only if the rule separated the choices in all the evidence (`leak` 0), has no excess failed replays, has `task_rule_support` supporting examples, and the task resembles those examples. Otherwise the step goes to T2, which shows the task and the known choices (option 0 is "something else"), or to T4.
- **`treejit approve EDGE --not-commit`** declares an opaque edge the operator knows is local (`./run_checks.sh`): it approves the edge and drops the extra-run requirement. It is per edge and never implied by `approve '*'`.
- **Shell tool arguments.** For shell tools, `env` entries are treated like `K=V` before the command; variables that run programs or load code (`GIT_*`, `*EDITOR`, `*PAGER`, `LD_*`, `BASH_ENV`, `PATH`, …) are commit points, also in string commands (`GIT_EXTERNAL_DIFF=x git diff`). Escalation flags (`with_escalated_permissions`, `dangerouslyDisableSandbox`, …) are never read-only, and unknown arguments are not read-only; `workdir`, `timeout`, `description` and similar are neutral. A T3 fill can't add a dangerous variable.
- **Repository config.** git reads (`status`, `diff`, `show`, `blame`, `log -p`) run programs named by the repository's config and attributes (`core.fsmonitor`, `diff.external`, textconv and clean filters, an embedded bare repository; `repro/S4_output.txt`). The proxy can't see the disk, so treejit **assumes a trusted checkout**. With `trust_repo_config = false`, git reads need approval like any write. Either way, a run is *tainted* once a call writes `.git/*`, `.gitattributes`, `.gitmodules` or `.gitconfig`, sets a git config key not known to be harmless, sets `GIT_*` variables, runs `git clone`/`submodule`/`init --bare`, or extracts an archive; for the rest of that run git reads are not read-only (reason `repo_tainted`). Also set `safe.bareRepository=explicit` in the agent machine's global git config.

## Compatibility with current Claude models

Claude Fable 5.1, Mythos 5.1 and Opus 5.5 changed three things treejit depends on, and left one question open (PLAN X1, X2). Nothing in this table has been run against the real API.

| Change | What treejit does | Verified against |
|---|---|---|
| **Forced tool use is a 400** (`tool_choice` `tool`/`any`: "not supported for this model") | T2/T3 subcalls use structured outputs (`output_config.format`) and never send `tool_choice` or `thinking`; effort `low` where the model takes it. | Documentation, plus fakes: request shapes per model (`tests/test_subcalls_compat.py`), T2/T3 end to end against a fake always-thinking model that 400s on a forced `tool_choice`. |
| **Preserved thinking**: a thinking block is bound to the exact prefix that produced it; accounts created on or after 2026-08-31 get a 400 when an earlier part of the history changes | Everything treejit changes in a forwarded body is append-only: sticky hints and first-sight compaction. `compact_mode = "window"` and `"epoch"` are **incompatible** with the check (both rewrite tool results earlier forwards sent in full). | Documentation, plus unit tests that consecutive forwards are byte-identical up to the previous request, hints included (`tests/test_hints.py`, `tests/test_compaction.py`). |
| **Thinking can't be disabled** (`{"type": "disabled"}` is a 400; leaving `thinking` out means adaptive) | The "drop `thinking` after replayed turns" fallback is skipped on models that think by default; the body is forwarded as the harness sent it. | Documentation and unit tests. |
| **Open question**: does the API accept treejit's replayed assistant turns (tool calls without thinking blocks) inside a tool loop whose other turns carry signed thinking blocks? | Nothing yet; if it doesn't, replay on these models needs a different shape. | **Unverified.** |

`repro/X2_live_check.py` settles all four once credentials exist (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile). It runs (A) a direct probe: a real turn, a replay-shaped turn without thinking, another real turn; and (B) a short session through the proxy with a pre-trained tree and `hints = "always"`, with `thinking.block_binding.prefix_mismatch_behavior: "error"` and the `thinking-binding-controls-2026-08-01` beta, and prints every 400 and `input_transformations`. Without credentials it prints SKIP; `--offline` checks its plumbing against a scripted stand-in.

## Decisions

The original design didn't specify these. Each came from a failure seen in the benchmarks. PLAN issue IDs are given where one applies.

**Learning**

1. **What the model would choose is learned only from steps the model chose.** Replayed steps count toward success and failure, but not toward branch purity or decision lists; otherwise replay reinforces its own guesses.
2. **Side exits teach.** A replayed step that breaks its postcondition is a miss against the context that chose it. If the model recovers and the run passes, the model's choice is the correct label there (DAgger-style).
3. **Back-off needs agreement.** When the root path has too little evidence, a less specific macro context may decide, but only if it proposes something the more specific contexts have seen the model do.
4. **Decision lists have no catch-all default.** An input no rule fires on goes to the model. Rules are penalized for firing on other labels' examples.
5. **Bindings may abstain.** A rule that is never wrong and correct on a clear majority is kept; where it can't produce a value, the step goes to the model (or T3).
6. **Stateless replay bookkeeping.** Replayed tool-call ids encode the deciding node and its confidence (`toolu_tj_<node>_<conf><rand>`), so side exits and the confidence budget need no session state.
7. **Extended thinking.** Replayed assistant turns carry no signed thinking blocks, so on models where leaving `thinking` out turns it off, a frontier call after replayed turns is sent without `thinking`. On models that think by default the body is sent as is.

**T2 and T3**

8. **A T2 pick is a model choice.** It is logged as not replayed and feeds purity and decision lists, so T1 can learn the branch and stop asking. Checkpoints and T3 steps are logged as replayed, since the tree proposed their structure.
9. **Value back-off.** Once T3 serves a hole at a general (n-gram) context, the specific contexts stop collecting model-chosen evidence, so a hole may borrow the rule the same edge has at a more specific context. Without this, T3 replaced free T0 steps with small calls and cost more than it saved.
10. **One subcall, first step only.** Subcalls are made only for the first call of a response, and a T2/T3 step ends the batch.

**Failed replays and where the model stops** (seed-3 regression, L1, L2)

11. **Failed replays are evidence against the choice that made them.** A replayed step in a failed run is a *negative* for its edge at its contexts, lowering purity (T0) and counting as a miss for the decision-list rules that predicted it (T1). Only the *excess* counts: negatives beyond a `1 − purity` failure rate among all replays of the same choice (`features.excess_negatives`).
12. **Task-word rules must earn T1.** A rule on a task word replays only once `task_rule_support` (5) model-chosen examples support it (observation rules need 2); each excess negative adds that many again. Until then its branch goes to T2.
13. **A minority choice blocks T0 until it is outnumbered 8 to 1.** The leading child's share gets one pseudo-count against it when the model has chosen more than one child.
14. **The model's decision to stop is a choice too.** A forwarded response with no tool call records `runs.ended_after`, and the builder adds an END choice for passing runs. When END leads at the deciding context, the request goes straight to T4 (reason `end@…`); END is never replayed.
15. **A task-word rule is trusted only on tasks like those that support it** (L1). Each rule stores the task-word sets of its supporting examples (the words at least two examples share; the most recent 40). T1 replays only if the task's best Jaccard similarity to one of them is at least `task_rule_similarity` (0.5); otherwise T2 with reason `unproven_rule`, so each new kind of task costs one small call, once. Observation predicates get no such gate (except at contested commit points, decision 21): similarity over file contents or order details would call almost every input new.
16. **Failures count per kind of task, not per rule** (L2). A task-word rule also stores the word sets of inputs it replayed into failed runs (`nx`); T1 needs the task to be more similar to a supporting example than to any of them. With 20 passing replays on one kind of task, a second kind now fails once and the first keeps T1 (`test_failures_count_per_input_class_not_per_rule`).

**Replay safety** (S1–S4, B1)

17. **When unsure, say "not read-only".** A false "no" costs one model call or one approval; a false "yes" runs a write unattended. Checks accept what they understand rather than rejecting what they recognise as dangerous.
18. **Unknown effects are a commit** (S1). Interpreters, shells, scripts, task targets, package scripts and git aliases are commit points. Known-local runners are exempt; `--not-commit` is the per-edge override. The synthetic suite was unchanged task for task.
19. **Match where programs run, not where words appear** (S3). Subsequence matching flagged 28 of 51 realistic commands wrongly; program-position matching gets 44 of 51 right, the other 7 being the documented exceptions. What the shell decides at run time counts as a commit (S2).
20. **Taint instead of rewriting** (S4). Git reads trust the repository's config, but a run that edits git config or metadata loses that trust for its remaining steps; computed from the run's own calls, so it needs no state. Rewriting commands with `-c` overrides doesn't cover filters or textconv and changes edge shapes.
21. **Only the task can choose between irreversible actions** (B1). On tau-bench, `get_order_details` on a pending order is followed by a cancel in some tasks and a modify in others; only the task says which. A rule like `status == pending → cancel` was right in 4 of 5 examples, enough for T0/T1, and replayed irreversible cancels into tasks that wanted a modify. A commit point is now replayed without the model only when the evidence shows no alternative (pooled over every context of the step) or a rule separated the alternatives perfectly with task-rule support on a similar task; every other case gets a T2 call that shows the task. A higher confidence threshold was rejected (the misrouting rule and correct cancel rules have about the same confidence), as was always using T2 for commit points (a small call even where the model has only ever done one thing).

**Runs and episodes** (T2, M1, E4)

22. **User turns are context, never choices.** A user reply is an edge the tree can condition on, but never replayed; the question before it becomes END evidence.
23. **The boundary is decided by the agent's last turn, not the user's words.** Whether the agent asked something is visible and is what makes a reply a reply. The heuristic is stateless; the header or `episode_mode` overrides it.
24. **Identify the conversation, then the task in it.** The first call id alone merged conversations whose backend numbers calls (`repro/T2_out.txt`). Session id or first turn, then episode index, then task hash, with the continuation check and forks as a backstop.

## Known limits

- **No real-model runs.** All results come from a simulated model or a tau-bench oracle. `--agent claude` and `repro/X2_live_check.py` are ready but have not run (no API key). Whether current Claude models accept replayed turns without thinking blocks is open.
- **Tool execution is invisible.** treejit sees the model API, not tool execution. It backtracks its policy, not the world; a tool with side effects outside the call's arguments is judged by its call alone.
- **The first alternative at an uncontested commit point can be misrouted once.** Decision 21 needs evidence of an alternative. After `promote_runs + 1` passing runs in which the model only ever cancelled after a lookup (at every context), the first task that wants a modify there gets a replayed cancel. `approve '*'` accepts that risk; approving commit edges one by one doesn't. The T2 call at a contested commit point also trusts the model to read the task.
- **Observation rules have no similarity gate** (decision 15), except at contested commit points. A chance observation predicate (a line of file content only one kind of task has shown so far) needs 2 supporting examples and is refuted only by the pooled negatives of decision 11.
- **Kinds of tasks are told apart by their words.** A new kind of task that uses a known kind's words (`delete P carefully` next to `delete P`) still gets one misroute from a task-word rule before its failed run blocks it (decision 16). The similarity threshold is global and unweighted.
- **Family keying is heuristic.** A volatile block the masks don't recognise that makes up more than about 10–20% of a prompt splits families, as does a large per-project `CLAUDE.md`. A harness upgrade that changes more than about 20% of the prompt starts a new family. Families migrated from the old prefix keying keep its `startswith` fallback. Concurrent writers to one family can lose one member's counts.
- **Episode boundaries are a heuristic.** A new task typed after the agent asked a question continues the episode as a `text` step; a follow-up after a turn that asked nothing starts a new episode with a fresh root path. Set `X-TreeJIT-Episode` or `episode_mode` when the harness knows. `show`/`export` draw the root path only down to a user step.
- **Run identity without a header or session id** is derived from the first user text and first assistant turn: two conversations identical up to the current step share a run until they diverge. Context editing that clears old tool results changes the salt of weak ids (a new run); `/compact` makes a new conversation to treejit. A weak-id first response whose conversation never continues keeps `run_id` NULL.
- **Responses API wire details are unverified** against a real Codex trace: the SSE event sequence was built from the public reference; replayed call items go upstream without their item `id`; `custom_tool_call`/`local_shell_call` streaming events are unverified. Stateful requests pass through unlearned.
- **Commit-point coverage is a list, not a proof.** Programs outside the categories above (`cp`, `rm`, `docker build`, `pip install`, an unknown CLI) are writes, not commit points, so an approved edge replays them after `promote_runs` passing runs. Commands built at run time from data treejit doesn't see (a function defined in an earlier call, a Makefile target that pushes) are caught only when the call itself shows an opaque executor.
- **Sticky hints and compaction decisions are shared by byte-identical prefixes** (a first message identical across conversations gets the same hint), and those tables grow with the log.
- **Weak call ids.** Conversations with weak ids (`call_0`) whose first turns are identical can't be told apart until their first observation.
- **Configs must match across processes.** Two processes on one db with different configs (e.g. `trust_repo_config`) build different trees; the generation check covers operator state, not config.
- **Repository taint sees calls, not the disk.** Taint covers every call in the conversation (all episodes), but a `git pull`/`checkout` that brings in a `.gitattributes` or a bare repository, or a `cd` in an earlier call, isn't tracked: that is the trusted-checkout assumption.
- **Rebuild cost is O(runs) per outcome.** Memoization made it cheap (P1), and the proxy builds off the event loop, but there is no incremental builder. `--rebuild-every K` in the bench trades freshness for speed.
- **Inline mode is synchronous.** `AsyncAnthropic`/`AsyncOpenAI` are rejected by `wrap()`; use the proxy. `messages.stream(output_format=...)`, `beta.messages` and `chat.completions.stream()` pass through unrecorded.
- **Compaction and the prompt cache.** First-sight compacts only steps replayed in a burst; the cache model is first-order (no 20-block lookback, no minimum cacheable length).
- **END needs a passing run and a visible final answer.** A forwarded inline stream closed before its stop reason is logged as 499 and adds no END.
- **The final answer always costs one full call.**

## CLI reference

`--db` (default `treejit.db` or `$TREEJIT_DB`) and `--config` (a `treejit.toml`) go before or after the subcommand. Ids (edges, nodes, families, runs) can be any unique prefix of at least 4 characters; the CLI prints 8.

| command | what it does |
|---|---|
| `treejit serve [--host H] [--port P]` | run the proxy (`/v1/messages`, `/v1/chat/completions`, `/v1/responses`, `POST /outcome`, `GET /health`, `GET /stats`; other paths pass through) |
| `treejit show [--family F] [--ids] [--depth N] [--no-macros]` | print the tree: tiers, bindings, guards, decision lists, macros; `--ids` adds short node/edge ids |
| `treejit runs [--limit N]` | list recent runs |
| `treejit explain <run\|latest> [--json]` | per-step timeline: who decided each step (`model`, `T0`/`T1`, `T2`, `ck`, `T3`, `user`), why, tokens; failed small calls as their own rows; a `small calls: N (M failed), T tokens` header line. Its token total equals the `requests` table's for the run. |
| `treejit outcome <run\|latest> pass\|fail\|error [--reason R]` | report a run's verifier result (rebuilds the family) |
| `treejit pending [--family F] [--json]` | promoted edges held back only by policy: example call, evidence, the approve commands |
| `treejit approve --review [--family F]` | walk the pending queue: `y` = at the listed nodes, `e` = everywhere, `n`/`s` = leave, `q` = stop |
| `treejit approve <edge\|'*'> [--node N]` | let replay cross a write edge / commit point (commit points still need `promote_runs + 1` passing runs) |
| `treejit approve <edge> --not-commit` | also declare this edge not a commit point (a local script); per edge, never implied by `'*'` |
| `treejit approve <edge> --revoke` / `treejit revoke <edge> [--node N] [--not-commit]` | undo an approval (`--not-commit`: withdraw only that declaration) |
| `treejit pin <node> [edge] [--unpin]` | force-promote a node or edge and protect it from eviction |
| `treejit prune [--days D] [--min-hits N] [--compact-days C] [--dry-run]` | evict cold nodes (defaults `evict_days`, `evict_min_hits`); with `--compact-days C`, also drop compaction decisions of runs idle longer than C days (never by default) |
| `treejit export --format html\|mermaid\|skills [--out PATH] [--family F]` | export the tree as an HTML page, a Mermaid graph, or a SKILL.md |
| `treejit build [--family F]` | rebuild trees from the trace log |
| `treejit stats [--family F]` | replay vs frontier counts, small calls |

## Configuration

Every key has a default. A `treejit.toml` (`[treejit]` table; `--config`, `$TREEJIT_CONFIG`, or `./treejit.toml`) or `TREEJIT_<KEY>` environment variables override them (lists as comma-separated values). Unknown keys are an error.

| key | default | meaning |
|---|---|---|
| `db` | "treejit.db" | SQLite file (trace log, tree, approvals) |
| `promote_runs` | 2 | N: distinct passing runs before an edge goes live (commit points need N + 1) |
| `max_depth` | 12 | D: root-anchored path depth; deeper steps use n-gram contexts only |
| `ngram` | [3, 2, 1] | last-k-edge macro contexts, most specific first |
| `purity` | 0.8 | minimum share of the evidence a child needs to be chosen (T0) |
| `task_rule_support` | 5 | model-chosen examples a task-word rule needs before T1 replays on it (decision 12) |
| `task_rule_similarity` | 0.5 | minimum Jaccard similarity of the task to a supporting example (decision 15; 0 = off) |
| `episode_mode` | "auto" | `auto` \| `conversation` \| `turn`: where a new task starts in a conversation |
| `theta` | 0.5 | confidence budget: the product of replayed edges' confidences must stay above it |
| `hard_cap` | 8 | K: max consecutive replayed steps |
| `batch` | true | collapse independent proven read-only edges into one assistant message |
| `max_batch` | 4 | max calls in one batched message |
| `t2` | true | enable T2 (choose among known children, budget checkpoints) |
| `t3` | true | enable T3 (fill holes of a known edge) |
| `small_model` | "" | model for T2/T3 subcalls (empty: the request's model) |
| `subcall_max_tokens` | 512 | `max_tokens` of a subcall (raised to 4096 on always-thinking models) |
| `subcall_format` | "auto" | Anthropic subcall shape: `auto` \| `json_schema` \| `tool_auto` \| `tool` (forced; 400s on current models) |
| `subcall_effort` | "low" | `output_config.effort` for subcalls on models that take it (empty: never) |
| `tomb_k` | 2.0 | decayed failures across distinct inputs before an edge is tombstoned |
| `tomb_prob` | 0.5 | minimum failure share for a tombstone |
| `half_life_days` | 14.0 | half-life of the decay applied to old evidence |
| `max_runs` | 2000 | most recent runs per family a rebuild reads |
| `rebuild` | "auto" | `auto` (background under `serve`, sync otherwise) \| `sync` \| `background` |
| `evict_days` | 30.0 | `prune`: evict nodes idle this long ... |
| `evict_min_hits` | 3 | ... with fewer hits than this |
| `hints` | "failures" | frontier hints on T4 calls: `off` \| `failures` (only where a child failed) \| `always` |
| `hint_max` | 5 | max hint lines |
| `compact` | false | enable frontier prefix compaction |
| `compact_keep_last` | 3 | the last N observations always go upstream in full |
| `compact_min_chars` | 400 | smaller observations are never compacted |
| `compact_mode` | "first_sight" | `first_sight` (append-only) \| `epoch` (+ re-compact when the cache is cold) \| `window` (legacy) |
| `compact_epoch_ttl` | 300.0 | `epoch`: seconds after which a conversation's cache counts as cold |
| `compact_keep_path` | false | also keep observations an earlier step's bindings read (old rule 3b) |
| `compact_retention_days` | 7.0 | epoch-mode conversation timestamps are forgotten after this long (decisions and hints are never pruned by age) |
| `replay_tools` | Read, Glob, Grep, LS, NotebookRead, … (21 entries) | non-shell tools that are read-only (globs) |
| `shell_tools` | Bash, bash, shell, run_shell_command, execute_command, … (7 entries) | tools whose `command`/`cmd` argument is a shell command |
| `commit_tools` | send_\*, cancel_\*, return_\*, exchange_\*, modify_\*, … (15 entries) | non-shell tools that are commit points (globs) |
| `commit_commands` | git push, npm publish, pnpm publish, yarn publish, cargo publish, … (31 entries) | shell command patterns that are commit points (opaque executors are added by the policy) |
| `extra_readonly_commands` |  | programs to treat as read-only with any arguments |
| `trust_repo_config` | true | git reads trust the repository's config (false: git reads need approval) |
| `host` | "127.0.0.1" | `serve` bind address |
| `port` | 8787 | `serve` port |
| `anthropic_upstream` | "https://api.anthropic.com" | where Anthropic requests are forwarded |
| `openai_upstream` | "https://api.openai.com" | where OpenAI (Chat Completions, Responses) requests are forwarded |

## Development

```bash
pip install -e '.[dev]' -e bench
pytest -q                                   # ~860 tests; tau-bench tests skip without TAUBENCH_PATH
pyflakes src bench/src tests
python -m treejit_bench --tasks 200 --out bench_out [--via-proxy] [--seed N] [--family coding|retail|mixed] \
    [--modes baseline,treejit,treejit+ok,treejit+ok+compact] [--payload none|tau|claude-code] [--cache] [--rebuild-every K]
TAUBENCH_PATH=../tau-bench python -m treejit_bench --suite taubench --tau-split test --out tau_out
```

[`PLAN.md`](PLAN.md) lists the issues closed in the last round of work, with their outcomes; [`repro/`](repro/README.md) holds the reproduction scripts.
