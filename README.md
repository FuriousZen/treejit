# treejit

An inference proxy with memory. treejit sits between an LLM agent harness and the model API. It learns a persistent execution tree from successful runs, and replays proven tool-call sequences without calling the model. The model is only called at the frontier: nodes the tree hasn't seen, branch points it can't resolve, and arguments it can't derive. For the last two, a small constrained call often suffices: it picks among known branches (T2) or fills in just the missing values (T3), instead of re-reading the whole conversation. Failed branches are recorded and avoided.

In short, it's a tracing JIT for agent trajectories, combined with a tree search whose tree persists across tasks.

```
harness ──► treejit ──► model API
               │  replayed step: tool call returned in ms, 0 tokens
               └─ frontier step: forwarded, recorded, learned from
```

## Status

This is the MVP from the handoff, plus value branching and composite argument templates (pulled forward because the benchmark needed them) and the T2/T3 escalation tiers.

| Area | State |
|---|---|
| Anthropic Messages + OpenAI Chat Completions dialects, JSON and SSE (replay and pass-through) | done |
| Trace recorder, system-prompt family keying (masked line sets), span-preserving shell tokenizer | done |
| Tree builder: anti-unification, provenance bindings, last-k-edge macros, root depth cap D | done |
| Failed replays as negative evidence, earned task-word rules, END (the model stops here) as a choice | done |
| T0 replay / T1 guarded branch, postcondition side exits, confidence budget, hard cap K, batching | done |
| Read-only allowlist, commit points + operator approval, soft tombstones, node-local T4 hints | done |
| CLI: `serve show runs explain outcome pending pin approve revoke prune export build stats`; HTML / Mermaid / SKILL.md export | done |
| Run identity (header, harness session id, derived; forks instead of extending finished runs) and multi-turn episodes (user steps) | done |
| Benchmark harness (synthetic suite, inline and real-proxy modes) + learning-curve report | done |
| T2 choose / budget checkpoint, T3 hole filling (one small forced-tool subcall, proxy and inline) | done |
| Frontier prefix compaction (verified replayed observations digested in forwarded requests) | done, **opt-in** (`compact = true`) |
| Inline-mode streaming replay (`stream=True` and Anthropic `messages.stream()`: replay, record, END) | done (sync clients) |
| Macros-as-tools, OpenAI Responses API | not yet |
| tau-bench runner (`--suite taubench`: oracle-with-noise agent, `ClaudeAgent` for a real model, tau-bench's own reward) | done; oracle runs below, real-model runs not yet (no API key here) |
| Bench cost model: virtual harness payload (`--payload`), prompt-cache model (`--cache`), `cost_tokens` | done |

The core (`src/treejit`, except `proxy.py`) uses only the standard library. Proxy mode also needs `httpx` and `uvicorn`.

## Results (synthetic suite, 200 tasks, `python -m treejit_bench --tasks 200 --modes baseline,treejit,treejit+ok,treejit+ok+compact`)

A simulated agent works a mixed stream of coding tasks (typo fix / version bump / delete module, each with a flaky-test branch) and tau-bench-style retail tasks (branching on order status). The tree starts empty. The simulated model sees only the conversation. It is stochastic (argument formatting varies, free-form commit messages), and it takes a known-bad shortcut 6% of the time. Observations are realistically sized: `Read` returns a 45–65-line file, `pytest -q` prints 150–420 tests as progress rows plus a warnings summary, and `git status` lists untracked build junk in about a third of the tasks.

*Full model calls* are T4 calls (the whole conversation). *Small calls* are T2/T3 subcalls (a short prompt and a forced tool call); their tokens are included in *tokens / task*. *Served by replay* counts tool calls that no full model call produced, T2/T3 steps included. Seed 0:

| mode | tasks | full model calls / task | small calls / task | tokens / task | served by replay | success | sim. wall-clock / task |
|---|---|---|---|---|---|---|---|
| plain agent | 151–200 | 6.02 | – | 6,196 | 0% | 96% | 10.0 s |
| treejit, read-only allowlist | 151–200 | 3.70 | 0.14 | 4,820 | 48% | 100% | 6.3 s |
| treejit, edges approved | 41–50 | 1.10 | 0.80 | 2,169 | 98% | 100% | 2.5 s |
| treejit, edges approved | 151–200 | **1.06** | 0.90 | 2,038 | **99%** | 100% | 2.6 s |
| treejit, edges approved + compaction | 151–200 | **1.06** | 0.90 | **1,893** | **99%** | 100% | 2.6 s |

Seeds 0–5. Success is over all 200 tasks; the other columns are tasks 151–200 (plain agent / allowlist / approved / approved + compaction). Seed 3 was re-measured after decisions 16–17; the other rows predate them:

| seed | success (of 200) | full calls / task | small calls / task | tokens / task | served |
|---|---|---|---|---|---|
| 0 | 193 / 198 / 200 / 200 | 6.02 / 3.70 / 1.06 / 1.06 | – / 0.14 / 0.90 / 0.90 | 6,196 / 4,820 / 2,038 / 1,893 | 0 / 48 / 99 / 99% |
| 1 | 189 / 198 / 200 / 200 | 6.36 / 4.20 / 1.02 / 1.02 | – / 0.02 / 0.28 / 0.28 | 6,684 / 5,311 / 1,637 / 1,445 | 0 / 41 / 100 / 100% |
| 2 | 192 / 198 / 200 / 200 | 6.32 / 4.12 / 1.00 / 1.00 | – / 0.00 / 0.78 / 0.78 | 6,711 / 5,279 / 1,931 / 1,734 | 0 / 39 / 100 / 100% |
| 3 | 189 / 198 / 200 / 200 | 6.26 / 4.20 / 1.04 / 1.04 | – / 0.10 / 0.94 / 0.94 | 6,590 / 5,406 / 2,026 / 1,943 | 0 / 42 / 99 / 99% |
| 4 | 185 / 198 / 199 / 199 | 6.28 / 4.08 / 1.02 / 1.02 | – / 0.10 / 1.20 / 1.20 | 6,744 / 5,513 / 2,315 / 2,248 | 0 / 45 / 100 / 100% |
| 5 | 192 / 198 / 200 / 200 | 6.66 / 4.58 / 1.00 / 1.00 | – / 0.00 / 0.02 / 0.02 | 7,401 / 6,014 / 1,452 / 1,407 | 0 / 37 / 100 / 100% |

- With edges approved (`treejit approve '*'`, which simulates operator review of write steps and commit points), almost every tool call is served from about task 40 on; the only full call left is usually the final answer. With the default read-only allowlist, only read steps replay, and T2/T3 rarely apply (their options must be replayable too).
- treejit never does worse than the plain agent on these seeds. The remaining failures are the simulated model's own shortcuts at steps it still decides (allowlist mode), Seed 3 no longer misroutes (below).
- **Seed 3 used to regress** (171/200 with edges approved, 86% success in tasks 151–200). The node after `git status` had learned the decision-list rule `task~src → git rm` from 2 delete-module tasks and 1 typo task in `README.md`. From task 13 on, it replayed `git rm` into every typo task whose file is under `src/`: 28 failed runs. The failures never reached the rule. Blame goes to edges, and `git rm` was right at that node for other tasks. Replayed steps never become examples, so the inputs the rule misrouted stopped producing evidence. Decisions 12–13 below fix this. One misroute was left (task 13): by then the rule had 5 supporting delete tasks and no counterexample. Decision 16 removes it: the typo task resembles none of those delete tasks, so it gets a T2 call instead, and seed 3 reaches 200/200.
- Small calls are mostly T3 fills (the free-form commit message, the `Edit` strings) and budget checkpoints. Few of them fall back to T4 ("something else", `not_this_step`): 8 of 127 at seed 0.
- Running the same stream through the real ASGI proxy with SSE streaming (`--via-proxy`) gives identical numbers.
- The floor is one full model call per task, because the final answer is always generated.
- In the simulation a small call costs about 45% of a full call's tokens (~450 vs ~1,080), because the simulated system prompt, tools and conversation are tiny. A real harness sends far more per call (Claude Code: tens of thousands of tokens), so the token column understates what T2/T3 save. `--payload` and `--cache` re-price the same trajectories (below).
- **Caveats:** the model and its token counts are simulated (tokens ≈ prompt chars / 4, latency = 600 ms + 15 ms/output token), so treat the absolute numbers as illustrative. tau-bench results with an oracle agent are [below](#tau-bench); real-model traffic hasn't been run yet.

Report: [`docs/learning_curve.html`](docs/learning_curve.html) (model calls, tokens, replay share and small calls vs task index, with a table view).

### Cost model: harness payload and prompt caching

`--payload none|tau|claude-code` adds a virtual harness system prompt plus tool schemas to every **full** call's input (0 / ~5k / ~24k tokens). T2/T3 subcalls are built by treejit and never carry it. `--cache` models Anthropic prompt caching: breakpoints after the static prefix, after the system prompt, and at the last message of each request (automatic caching). A request reads the longest prefix an earlier request wrote and writes the rest. With either flag the CSV gains `cache_read`, `cache_write`, `cost_tokens` and `small_cost`. `cost_tokens` is in input-token equivalents: uncached input 1×, cache write 1.25×, cache read 0.1×, output 5× (`--cost-weights W,R,O`). The default output is unchanged: the seed-0 CSV is identical task for task, apart from the timing columns. The report gains a cost panel.

Seed 0, 200 tasks, tasks 151–200:

| scenario | plain agent cost / task | treejit, edges approved | cut | small-call cost / full-call cost |
|---|---|---|---|---|
| sim as-is (no payload) | – | – | – | ≈0.5 (`repro/E2_out_seed0.txt`) |
| `--payload tau` | 37,989 | 7,711 | −80% | 0.136 |
| `--payload claude-code` | 152,369 | 27,851 | −82% | **0.035** |
| `--payload claude-code --cache` | 18,032 | 4,721 | −74% | 0.239 |

With a payload the size of Claude Code's, a small call costs about 3.5% of a full call. With caching, full calls get about 8× cheaper, because most of the prompt is a cache read. The small calls' relative cost then rises to about a quarter of a full call. T2/T3 still pay off, but by less.

## tau-bench

`bench/src/treejit_bench/taubench.py` runs [tau-bench](https://github.com/sierra-research/tau-bench) tasks through treejit inline mode. It scores them with tau-bench's own `Env.calculate_reward`: the database hash after the episode must equal the hash after the ground-truth actions, and every expected output must appear in a reply. tau-bench isn't on PyPI. Its environments need only `pydantic`. When `litellm` isn't installed, a stub module is inserted; the user simulator is never called.

```bash
git clone --depth 1 https://github.com/sierra-research/tau-bench && pip install pydantic
export TAUBENCH_PATH=$PWD/tau-bench        # tests/test_taubench.py skips without it
python -m treejit_bench --suite taubench --tau-env retail --tau-split test --modes baseline,treejit,treejit+ok --out tau_out
#   --tasks N --tau-start I     a slice (default: the whole split; retail test 115, train 500, dev 20; airline test 50)
#   --noise P                   the oracle's per-write slip probability (default 0.05)
#   --agent claude [--claude-model claude-opus-5] [--tau-user confirm]    a real model (needs ANTHROPIC_API_KEY and anthropic)
#   --rebuild-every K           rebuild the tree after every K-th outcome (faster; changes learning dynamics, see below)
```

- **Tools and prompt.** tau-bench's `tools_info` (OpenAI function specs) become Anthropic tools, the policy wiki is the system prompt, and the task instruction is the first user message. The environment's data is serialized to JSON once and restored before each task and for the reward's replay.
- **OracleAgent.** A deterministic agent. It makes a canonical read prefix, then the task's ground-truth actions, then a final answer containing the expected outputs. The retail prefix is `find_user_id_by_email` (or `…_by_name_zip`), `get_user_details`, `get_order_details` per order, and `get_product_details` for new items. The airline prefix is `get_user_details` and `get_reservation_details`. With probability `--noise` per write step, the oracle *slips*: a wrong reason, a dropped item, a wrong payment method, and so on. Slips depend only on (seed, env, split, task index), so every mode sees the same model mistakes. The oracle answers T2/T3 subcalls with the same policy. With noise 0 it scores reward 1.0 on all 115 retail test tasks, all 500 retail train tasks and all 50 airline test tasks.
- **ClaudeAgent** (`--agent claude`) sends the same bodies to a real model through the Anthropic SDK. In treejit modes it runs through `jit.wrap(agent)`. It sets a cache breakpoint on the system prompt plus automatic caching, and accounts `cache_read_input_tokens` / `cache_creation_input_tokens`. Real models ask for confirmation before writes (the wiki requires it), so use `--tau-user confirm`. treejit currently splits such multi-turn episodes (PLAN M1). T2/T3 subcalls use a forced `tool_choice`. Some newer models reject that with a 400, and the subcall then falls back to T4. `claude-opus-5` accepts it. This path is tested only against a fake SDK client, since there is no API key here.
- **ScriptedUser.** Single-turn by default: the episode ends at the agent's first text reply. `confirm` answers up to 4 agent questions with "Yes, I confirm."

**Results.** Oracle, seed 0, noise 0.05, retail. There is no payload: the real wiki and 16 tool schemas (~5k tokens) are in every body. Cost is in input-token equivalents, without the cache model.

| split | mode | tasks | full calls / task | small calls / task | tokens / task | cost / task | served by replay | reward |
|---|---|---|---|---|---|---|---|---|
| test | plain agent | 1–115 | 7.75 | – | 41,627 | 43,938 | 0% | 0.939 |
| test | treejit, read-only allowlist | 1–115 | 4.13 | 3.21 | 29,633 | 31,420 | 65% | 0.939 |
| test | treejit, edges approved | 1–115 | 3.76 | 3.63 | 28,084 | 29,828 | 69% | 0.939 |
| test | treejit, edges approved + compaction | 1–115 | 3.76 | 3.63 | 27,445 | 29,189 | 69% | 0.939 |
| test | treejit, edges approved, `--rebuild-every 10` | 1–115 | 4.31 | 3.57 | 29,244 | 31,144 | 52% | 0.939 |
| train | plain agent | 1–300 | 6.94 | – | 36,684 | 38,737 | 0% | 0.940 |
| train | treejit, read-only allowlist | 1–300 | 3.50 | 1.08 | 23,125 | 24,341 | 62% | 0.940 |
| train | treejit, edges approved | 1–300 | 2.44 | 3.27 | 19,313 | 20,534 | 78% | 0.940 |
| train | treejit, edges approved | 251–300 | 2.02 | 3.16 | 16,170 | 17,233 | 83% | 0.920 (plain agent 0.920) |
| airline test | plain agent / allowlist / edges approved | 1–50 | 5.20 / 3.52 / 3.30 | – / 1.12 / 1.88 | 25,409 / 19,456 / 19,443 | | 0 / 40 / 45% | 0.98 / 0.98 / 0.98 |

- On the retail test split, every treejit mode has exactly the plain agent's reward: the 7 failing tasks are the oracle's slips, task for task. The read-only allowlist mode also matches the plain agent on train and airline.
- **With edges approved, treejit used to fail retail train tasks that the plain agent passes** (149, 191, 197, 238, 243, 245 and 298 in the first runs, then 160 after the multi-turn episode changes, with reward 0.920–0.937), and airline test task 29. In every case T0/T1 (sometimes with a T3 fill) replayed an irreversible cancel that the task didn't want: `cancel_pending_order` on a pending order the task wants *modified*, or `cancel_reservation` in a read-only task. The id was the one just looked up, so the binding was right; the *choice* was wrong. `treejit explain tau-retail-train-0-238` showed `T1@r6 cancel_pending_order(...) conf=0.70`. The rule behind it was `json.status == "pending" → cancel_pending_order` at that node: 5 model-chosen examples, 4 cancels and 1 that went on to modify (purity exactly 0.8, support 4, one leak). An observation rule needed only 2 supporting examples, and nothing else looked at the task. Status "pending" is where both cancel and modify happen; only the task text tells them apart. Decision 24 fixes this: all of these tasks now pass, and retail train 300, retail test 115 and airline test 50 fail exactly the plain agent's tasks (the oracle's slips), task for task. On train, 14 commit-point writes were replayed by T0/T1 and 12 had their structure picked by T0/T1 with a T3 fill; after the fix, 3 and 1 are. The others go to T2 and cost 0.03 small calls per task (3.24 → 3.27). Full calls per task (2.44) and the served share (78%) don't change, because T2 answers with the values in the same small call that a T3 fill used to spend. With the default allowlist these steps always went to the model, so nothing was lost there.
- Small calls are frequent (2–3.6 per task). Most retail steps have their structure decided, but no binding produces a value this input needs (order id, item ids, reason), so a T3 fill asks for it. A small call costs 0.19–0.21 of a full call here, because every full call carries the ~5k-token wiki and tools.
- **Rebuild cost.** The tree is rebuilt after each outcome, and on tau-bench data that dominates the run time. Each treejit mode took 6–13 minutes on test 115 and about 35 minutes on train 300 on this machine (4 runs in parallel), against 40 s and 100 s for the plain agent (PLAN P1). `--rebuild-every K` rebuilds only after every K-th outcome and keeps serving the stale tree in between. That changes the learning dynamics, because evidence from the last K−1 runs isn't visible yet: K=10 served 52% instead of 69% on test, in 93 s. The default is K=1.
- The numbers differ slightly from the prototype's (`repro/E1_tau_proto.py`). The prototype drew slips per instruction text, and some of its slips had no effect. The runner keys slips on the task index and always perturbs an argument.

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

**Outcomes.** Only runs reported as passing ever promote an edge.

```bash
curl -s localhost:8787/outcome -d '{"run_id": "latest", "outcome": "pass"}'
treejit outcome <run_id|latest> fail --reason "tests failed in CI"
```

In Claude Code, a `Stop` hook that runs your verifier and then posts `/outcome` closes the loop. Outcome `error` (timeouts, 429s) is recorded but never counted as evidence.

**Rebuilds.** Each outcome rebuilds its family's tree. Under `treejit serve` the rebuild runs on a background worker thread (`rebuild = "auto"`, the default), so an outcome never stalls other requests: they keep the previous tree until the new one is swapped in. Outcomes that arrive during a build are coalesced into one more build of that family. `/outcome` still answers only once the tree is rebuilt, so the next request sees it; post `"wait": false` to get the answer as soon as the outcome is recorded. Inline mode and the CLI rebuild synchronously, before `jit.outcome()` returns, which keeps benchmark numbers deterministic. `rebuild = "sync"` or `"background"` (`TREEJIT_REBUILD`) forces either mode; `jit.wait_rebuilds()` waits for pending background builds. A running proxy also notices a rebuild made by another process (`treejit outcome ...`, a second instance on the same db): each request compares the family's `built_at` with its cached view's and reloads the view when they differ.

**OpenAI-compatible harnesses:** point `OPENAI_BASE_URL` at `http://127.0.0.1:8787/v1`.

**Inline mode** (harnesses you own, tests):

```python
from treejit import TreeJIT
jit = TreeJIT("treejit.db")
client = jit.wrap(anthropic.Anthropic())             # or an OpenAI client, or a callable body -> dict
client.messages.create(..., extra_headers={"X-TreeJIT-Run": "task-17"})
with client.messages.stream(..., extra_headers={"X-TreeJIT-Run": "task-17"}) as s:  # streaming works too
    msg = s.get_final_message()
jit.outcome("task-17", "pass")
```

Streaming (`create(stream=True)`) takes the same path as JSON. A replay returns a `ReplayStream`: the replay's events, served locally, as SDK event models (`RawMessageStreamEvent`, `ChatCompletionChunk`) when the SDK is installed and as dicts otherwise. A forward returns a `TeeStream`: the upstream events, unchanged, recorded (usage, run, END) when the stream is exhausted. Closing it before the stop reason arrives records status 499 and no END. Anthropic's `messages.stream()` is the SDK's own `MessageStreamManager` fed by those streams (`text_stream`, `get_final_message()`, derived `text`/`input_json` events); without the SDK a minimal shim with the same methods is used. `X-TreeJIT-Run` never goes upstream. Everything else on the client (`messages.count_tokens`, `messages.batches`, `beta`, `with_raw_response`, ...) is the real client's and is not recorded.

**Inspect and operate**

```bash
treejit show --ids          # tree with tiers, bindings, guards, decision lists, macros; short node/edge ids
treejit runs; treejit stats
treejit explain <run|latest> [--json]   # per-step timeline: who decided each step (model/T0/T1/T2/ck/T3/user), why, tokens
treejit pending [--family F] [--json]   # promoted edges held back only by policy: example call, evidence, approve commands
treejit approve --review                # walk the queue: y = at the listed nodes, e = everywhere, n/s = leave, q = stop
treejit approve <edge|'*'> [--node N]   # let replay cross a write edge / commit point
treejit approve <edge> --not-commit     # ...and declare an opaque edge (./run_checks.sh) not a commit point; never implied by '*'
treejit revoke <edge> [--node N] [--not-commit]   # same as approve --revoke (--not-commit: withdraw only that)
treejit pin <node> [edge]               # force-promote, protect from eviction
treejit prune --days 30 --min-hits 3
treejit export --format html|mermaid|skills --out ...
```

Ids (edges, nodes, families, runs) can be given as any unique prefix of at least 4 characters; the CLI prints 8. `--db` and `--config` go before or after the subcommand.

`explain` accounts for every logged request of the run, so its token total equals the `requests` table's for that run. The header line counts small calls (`small calls: N (M failed), T tokens`). A step is labelled by what produced it: `model`, `T0`/`T1` (replay), `T2` (the model picked it among known children), `ck` (the model confirmed it at a budget checkpoint) or `T3` (the model filled its holes), read from the call id's suffix. A small call that failed shows as its own row (`(T2 small call failed: chose_new → model)`), followed by the model's step. A text answer the conversation went on after reads `replied to the user`, and later user turns show as `user` rows between the steps.

Each tree edge has a tier: **hot** (replayable), **live** (promoted but blocked), **warm** (one passing run), **cold**, or **tomb**. `show` prints why a live edge doesn't replay: `LIVE:holes` (an argument has no binding), `LIVE:needs_approval` (not read-only, or a commit point), or `LIVE:commit_point_needs_evidence` (approved, but a commit point also needs `promote_runs + 1` passing runs). Only the last two appear in `pending`: approval can't fix holes or tombstones.

## How it works

| module | job |
|---|---|
| `dialects.py` | Parse requests into an *episode* (task + (call, observation) steps, with later user turns as user steps; see [Runs and episodes](#runs-and-episodes)) and read harness session ids. Build replay responses as JSON or SSE. Accumulate upstream SSE for usage. Inject hints. |
| `families.py` | Tree key = tool schemas + the stable lines of the system prompt. Lines are masked (dates, times, absolute paths, hashes, uuids, URLs, numbers; git status and commit-log lines collapse to one placeholder), hashed and counted per family, weighted by length. A line is stable when ≥80% of the family's distinct prompts have it (every line, for a 1-member family). A prompt joins a family when it has ≥90% of the family's stable characters and the stable lines make up ≥80% of its own; ties go to the best coverage, then the oldest family. Volatile lines anywhere (an early `<env>`, a large `gitStatus`) don't split a family; a different agent sharing a short preamble doesn't merge into one. Ids never change. Families from the old prefix keying are seeded from their prefix and keep its `startswith` rule. `families.prefix` shows the stable (masked) lines. |
| `shellwords.py` | Span-preserving shell tokenizer (quotes, `$(...)`, heredocs, operators). Replayed commands are spliced into the original text, so quoting survives. |
| `policy.py` | What replay may emit unattended: the read-only allowlist and commit points (see [Replay safety](#replay-safety)). |
| `templates.py` | Edge = tool + arg keys + per-segment command heads (`git commit`, `npm test`). Calls with one shape are anti-unified per token into constants and variables. |
| `bindings.py` | Provenance search. Each variable binds to `$task` / `$obs[-k]` / `$arg[-k]` via extractors (JSON path, `key: value`, regex types, after-word, line, token), `fmt` templates (`Bump version to {$task.version}`) or `case` (value chosen by predicates). A variable with no rule is a hole. |
| `features.py` | Predicate set: error/exit code, empty, JSON field equals, substring, task keyword. Guards are conjunctions of stable features; branches are decision lists. |
| `builder.py` | Deterministic rebuild from the trace log on every outcome (bounded to the last `max_runs`): promotion, bindings, guards, postconditions, credit assignment, tombstones, tiers, priority score. |
| `compaction.py` | Opt-in frontier prefix compaction: digests of verified replayed observations in forwarded T4 requests (below). |
| `tree.py`, `replay.py` | Stateless recognition (root path ≤ D, then last-k n-grams). Climbs T0 → T1 → T3/T2 → T4, bounded by budget, cap, allowlist and commit points. Returns T2/T3 opportunities as `plan.sub` and never calls a model itself. |
| `subcalls.py` | Builds the T2/T3 subcall (short prompt, forced `treejit_fill` / `treejit_choose` tool, per dialect) and parses its answer. |
| `proxy.py`, `inline.py`, `cli.py`, `export.py`, `operate.py` | The two entry points, CLI, views, and operator tools (approval queue, run timelines, short ids). |

### Decisions made during implementation

The handoff didn't specify these. Each one came from a failure seen in the benchmark:

1. **What the model would choose is learned only from steps the model chose.** Replayed steps still count toward success and failure, but not toward branch purity or decision lists. Otherwise replay reinforces its own guesses. The benchmark showed a misrouted-but-harmless read becoming "certain".
2. **Side exits teach.** A replayed step that breaks its postcondition is a miss against the context that chose it. If the model then recovers and the run passes, the model's choice is recorded as the correct label at that context (DAgger-style).
3. **Back-off needs agreement.** When the root path has too little evidence, a less specific macro context may decide, but only if it proposes something the more specific contexts have seen the model do.
4. **Decision lists have no catch-all default.** An input no rule fires on goes to the model. Rules are scored with a penalty for firing on other labels' examples.
5. **Bindings may abstain.** A rule that is never wrong and correct on a clear majority is kept; where it can't produce a value, that step goes to the model. A node that mixes two task types therefore keeps working for the majority instead of becoming a hole.
6. **Stateless replay bookkeeping.** Replayed tool-call ids encode the deciding node and its confidence (`toolu_tj_<node>_<conf><rand>`), so side exits and the confidence budget need no server-side session state.
7. **Extended thinking.** Replayed assistant turns carry no signed thinking blocks, so a frontier call that follows replayed turns in the same episode is sent without `thinking`.

### T2 and T3

`TreeJIT.handle()` may return `Result(kind="subcall")`: a small request body in the client's dialect. The transport (proxy or inline) sends it upstream non-streaming with the client's auth headers, then calls `jit.resume(result, response_json, status)`. That returns a normal replay (built as SSE if the client streamed) or the T4 forward. There is at most one subcall per incoming request. Any failure falls back to exactly the T4 forward the request would have had, hints included: an HTTP error, no tool call, missing or empty values, "something else", or a value that fails the safety re-checks.

- **T3 (fill).** The structure is decided: T0/T1 picks an edge that is live, not tombstoned, and allowed by the allowlist or approvals. But some variables have no rule, or their rule abstains for this input. The subcall shows the task, all calls so far (arguments truncated to 200 chars), the last 3 tool results (2,000 chars each), and the next call with `<placeholders>`. The model answers one string per hole (with an earlier run's value as an example), or `not_this_step`. Values are spliced in as data (one quoted shell word), then the call is re-rendered and re-checked. A read-only call must stay read-only, its commit reason (`policy.commit_reason`) must be the one stored for the edge (so `sh -c 'echo a'` can't be filled into `sh -c 'git push'`, nor `make test` into `make deploy`), and the call must still match the same edge shape.
- **T2 (choose).** Used in two cases. The first is a node with enough evidence that is ambiguous: at least 2 children the model has chosen, and the replayable ones account for at least 50% of those choices. The second is a *checkpoint*: the confidence budget runs out on an otherwise confident step, so the proposed step becomes option 1 and its siblings follow. The model answers an option number, or 0 for "something else" (→ T4). If the chosen option has holes, the same call asks for their values (`o<N>_<name>` fields). A T2 step restarts the confidence budget. The hard cap K still counts it as a replayed step.
- Subcalls use `small_model` if set (else the request's model) and `subcall_max_tokens`, and `t2` / `t3` switch them off. All four are config keys (or `TREEJIT_*` variables). Subcalls are logged in `requests` with tier `T2` / `T3` and their own usage, and `treejit stats` shows them. Edge savings are still measured against T4 calls only.
- Subcall steps are marked in the call id (`..._t3`, `..._t2`, `..._ck`), so recognition, side exits and the budget stay stateless.

Decisions made while adding them:

8. **A T2 pick is a model choice.** The model chose among known children, so the step is logged as not replayed and feeds purity and decision lists. T1 can then learn the branch and stop asking. Budget checkpoints and T3 steps are logged as replayed, because the tree proposed their structure.
9. **Value back-off.** Once T3 serves a hole at a general (n-gram) context, the more specific contexts stop collecting model-chosen evidence, so they never become the deciding context. Their value rules are still learned from every passing instance, so a hole may borrow the rule the same edge has at a more specific context. Without this, T3 replaced free T0 steps with small calls (the version-bump commit message) and cost more tokens than it saved: 1,439 tokens/task against 1,373 before T2/T3.
10. **One subcall, first step only.** Subcalls are only made for the first call of a response, and a T2/T3 step ends the batch.

### Replay safety

Replay emits a call with no model call and no human in the loop only if `policy.is_readonly` says the call is read-only. Otherwise the edge needs operator approval (`treejit approve`). A commit point (`policy.commit_reason` is non-empty: `git push`, `curl`, `send_*`, an interpreter, a script, ...) needs approval and `promote_runs + 1` passing runs. For shell commands (`policy.py`):

- **One view of the program.** `shellwords.unwrap` skips `VAR=val` assignments and transparent wrappers with their own options: `env` (`-i`, `-u NAME`, `-C DIR`, `VAR=val`), `sudo`, `time`, `nohup`, `exec`, `command`, `nice`, `timeout N`, `stdbuf`. Edge shapes (`command_heads`), the read-only check and commit-point detection all use it, so `env git push origin main` has the shape `git push`, is a commit point, and is not read-only. An unrecognised wrapper option (such as `env -S`) makes the command not read-only. `sudo` is located but never counts as read-only. A bare `env`/`printenv` (with no command) is read-only.
- **Allowlist, not denylist.** Every simple command in the line must run an allowlisted program (`ls`, `cat`, `grep`, `head`, `wc`, `jq`, `diff`, ...). The program must be a bare name or live in a standard bin directory (`./ls` doesn't count), and its name must be literal: no `$VAR`, globs, brace expansion or `$'\…'` escapes. Assignments are limited to harmless variables (`LC_*`, `LANG`, `TZ`, `TERM`, `NO_COLOR`, `PAGER=cat`, ...), which keeps out `LD_PRELOAD`, `PATH`, `GIT_EXTERNAL_DIFF` and similar.
- **Per-program argument checks** accept only options they know. Their arguments must also be literal, since a glob can expand to a file named `--pre=x`. This keeps out:
  - `sort -o`/`--compress-program`, `uniq IN OUT`, `date -s`/`date MMDDhhmm`, `hostname NAME`, `tree -o`/`-R`, `file -C`;
  - `yq -i`/`-s`, `fd -x`/`-X`, `rg --pre`/`--hostname-bin`, `ag --pager`, `bat --pager`/`bat cache`;
  - `find`'s exec/write actions (`-exec*`, `-ok*`, `-delete`, `-fprint*`, `-fls`).

  `git` gets a separate check. Global options are limited to `-C`, `--git-dir`, `--work-tree`, `--no-pager` and similar (no `-c`, `--config-env` or `--exec-path`). The subcommand must be a read (`status`, `diff`, `log`, `show`, `grep`, ...) with no `--output` (or an abbreviation of it) and no `grep -O`. `branch`/`tag` are allowed only when listing, `config` only when reading a dotted key or with `--get*`/`--list`, `remote` only bare or with `show`/`get-url`, `stash` only with `list`/`show`, and `reflog` not with `expire`/`delete`/`drop`.
- **sed and awk** have languages that can write files and run commands, so only a conservative subset counts as read-only:
  - sed: flags `-n -E -r -s -u -z -e -l`; scripts made of addresses, `!`, `{}` and `p P d D n N g G h H x z = q Q l`, or `s///` with flags `g p i I m M N`. No `w`, `W`, `r`, `R`, `e`, `a`/`i`/`c`, `y` or labels, and no `-i`/`-f`.
  - awk: only `-F` and `-v` options, and no `system`, `getline`, `extension`, `@`, `|` (except `||`) or `>` (except `>=`) anywhere in the program.

  Anything outside the subset needs approval. It is cheaper to spend one model call than to classify a language by substring.
- **Redirections** may only read (`<`, `<<<`, heredocs), duplicate a descriptor (`2>&1`, `>&2`), or write to `/dev/null`, `/dev/stdout` or `/dev/stderr`. This covers `>`, `>>`, `>|`, `&>`, `N>`, `>&FILE`, `<>` and `>(...)`. Any command substitution makes the line not read-only, and so does process substitution.
- **Excluded by default:** `sudo`, `less`/`more` (`LESSOPEN`, `-o` log files), `xargs`, `tee`, shells, `eval`, and shell keywords (`for`, `if`, `{ }`, functions). To opt a program in, add it to `extra_readonly_commands` (`TREEJIT_EXTRA_READONLY_COMMANDS=less,xargs`). This trusts the program with any arguments, but the program-path, assignment and redirection rules still apply.
- **The tokenizer cooks words as bash does.** `$'...'` is decoded in full (`\NNN`, `\xHH`, `\uHHHH`, `\cX`, ...), and `$"..."` cooks like `"..."`, so `$'\147it' push` has the shape `git push`. Leading reserved words (`if`, `then`, `do`, `{`, `!`, `function f`) are skipped when locating the program.
- **Commit points** (`policy.commit_reason`, which says *why*; `is_commit_point` is `bool` of it) are matched only where a program runs: the program of each simple command after reserved words and wrappers (`env git push`, `timeout 60 git push`, `if git push; then`), the command that `xargs`, `parallel`, `watch`, `strace`, `flock`, `chroot` and similar runners run, `find -exec*`/`-ok*`, and `python -m MODULE`. Nested scripts count too: `$(...)`, backticks, `sh -c`, `eval`, `env -S`, and quoted text when a shell, `watch` or `parallel` in the line may run it (`echo 'git push' | sh`). Quoted text elsewhere is data, so `git commit -m "then git push"` is not a commit point, and `apt install curl`, `mkdir ssh`, `git stash push` are not either. A multi-word pattern (`git push`) is also caught anywhere in a command that isn't read-only, for wrappers we don't model (`mywrap git push`). In a command that isn't read-only, commit points are:
  - **Configured patterns** (`commit_commands`), matched loosely from the program (`docker compose push`, `npm --registry x publish`). Patterns headed by `git` match from git's subcommand, found by parsing its global options (`git -C repo --no-pager push`).
  - **Non-literal program or git subcommand**: `$CMD x`, `git $x`, `git "$@"`, `{git,push}`, `git${IFS}push`, `$'\147it' push`. The shell decides what runs, so the policy can't.
  - **git**: `push`, `send-pack`, `http-push`, `svn`, `p4`, `send-email`, `imap-send`, `cvsexportcommit`; any subcommand that isn't a builtin (an alias can be `!sh -c ...`, an extension is any `git-*` program); `-c KEY=...` / `--config-env` / `--exec-path=` except for harmless keys (`user.*`, `color.*`, `core.pager=cat`, ...); `rebase --exec`, `bisect run`, `submodule foreach`, `filter-branch`, `hook`.
  - **Opaque executors**, whose effects the call doesn't show: interpreters (`python`, `node`, `perl`, `ruby`, `awk` outside the read-only subset, ...) except `-m` with a local module (`pytest`, `unittest`, `pip`, `mypy`, ...) and `--version`; shells (`bash x.sh`, `bash <<< '...'`, `... | sh`), `source`/`.`; script paths (`./deploy.sh`, `scripts/x.py`, `x.sh`), except tool bins (`.venv/bin/pytest`, `node_modules/.bin/jest`); task runners (`make`, `just`, `rake`, `nox`, `gradle`/`./gradlew`, `mvn`, ...) unless every target is made of local words (`test`, `lint`, `build`, `test:unit`, `typecheck`, ...), with no `VAR=val` override; package scripts and runners (`npm run release`, `yarn release`, `npm start`, `npx vercel`, `uv tool run`, `cargo run`, `go run`/`generate`, `deno run`, `dotnet run`, unknown cargo extensions), with the same local-name exemption and `npx jest`-style exemptions for known local tools; `uv run`/`poetry run`/`bundle exec CMD` classify CMD.
  - **Remote writers**: `gh` except read verbs (`view`, `list`, `diff`, `checks`, `status`, `download`, `checkout`, `clone`, top-level `search`/`status`/`browse`) and `gh api` GETs (a `-f` field turns it into a POST); `curl`/`wget` except a plain GET/HEAD of loopback URLs with options that send nothing (`curl -sSf http://127.0.0.1:3000/ -o /dev/null`); cloud and cluster CLIs (`kubectl`, `helm`, `terraform`, `aws`, `gcloud`, `az`, `fly`, `heroku`, `vercel`, ...) unless the positional words include a read verb and no write verb (`kubectl get pods`, `aws ec2 describe-instances`); HTTP, mail and socket clients (`http`, `xh`, `sendmail`, `nc`, ...).
  - **Deliberate exceptions** (still commit points): dry runs (`git push --dry-run`, `npm publish --dry-run`, `kubectl apply --dry-run=client`: we'd have to trust each tool's flag), local `rsync` (telling it from a remote copy means parsing its host syntax), and `npm run <name>` whose name isn't made of local words (`npm run push-docs`). Known-local runners (`pytest`, `tox`, `npm test`, `cargo test`/`build`, `go test`, `make test|lint|build`, `uv run pytest`) are not commit points, though they're not read-only either.
- **The reason is stored.** The builder stores `commit_reason` per (node, edge) from the edge's reference call. `materialize` re-renders every call it emits and rejects it when the reason differs, so a T3 value can't change a call's commit status or turn one commit into another.
- **Contested commit points go to the model** (decision 24). A commit point where the model has also chosen something else after the same history (a sibling edge, or ending the episode, at any context of the step) never replays on a majority (T0). A decision-list rule (T1) may pick it only if the rule separated the choices in all the evidence: no example of another choice where its predicate holds (`leak` 0), no excess failed replays, `task_rule_support` supporting examples whatever the predicate, and a task that resembles those examples (the similarity gate of decision 16, applied to observation predicates as well). Otherwise the step goes to a T2 call, which shows the task and the known choices (option 0 is "something else", so a task that wants nothing done gets the model), or to T4 if T2 is off or the known choices cover too little. A commit point the model has only ever chosen after that history still replays with approval plus `promote_runs + 1` passing runs.
- **`treejit approve EDGE --not-commit`** is the per-edge override for an opaque executor the operator knows is local (`./run_checks.sh`): it approves the edge and drops the `promote_runs + 1` requirement. It is stored per edge (`not_commit` table) and never implied by `approve '*'`, which approves every edge but leaves commit points needing the extra evidence.
- **Repository config.** git reads (`status`, `diff`, `show`, `blame`, `log -p`) run programs named by the repository's own config and attributes: `core.fsmonitor`, `diff.external`, textconv and clean filters, and those of a bare repository embedded in the tree (`repro/S4_output.txt`). The proxy can't see the disk, so treejit **assumes a trusted checkout**: the repository's config and history come from someone you trust. With `trust_repo_config = false` (`TREEJIT_TRUST_REPO_CONFIG=0`) git reads are not read-only and need approval like any write; everything else is unchanged. Either way, a run is *tainted* once one of its calls changes what git will read (`policy.repo_taint`): it writes `.git/*`, `.gitattributes`, `.gitmodules` or `.gitconfig` (shell or file tools), sets a git config key that isn't known to be harmless, sets `GIT_*` variables, runs `git clone`, `git submodule`, `git init --bare`, or extracts an archive (`tar x`, `unzip`, `7z`, ...). For the rest of a tainted run, replay treats git reads as not read-only: an unapproved `git status` edge goes to the model (reason `repo_tainted`). Also set `safe.bareRepository=explicit` in the global git config of the machine the agent runs on: it stops git from using a bare repository it finds while walking up from the working directory. Rewriting replayed commands with `-c core.fsmonitor=false ...` was considered and rejected: it doesn't cover filters or textconv, and it changes edge shapes.

Decision made while hardening it:

11. **When unsure, say "not read-only".** A false "no" costs one model call, or one approval. A false "yes" runs a write with no model call and nobody watching. Checks therefore accept what they understand rather than rejecting what they recognise as dangerous. The exception is `find`, whose side-effecting actions are a closed, documented set. On the synthetic suite the stricter policy changed nothing: read-only `treejit` mode served 44% of tool calls in tasks 71–120, before and after.

Decisions made while closing the commit-point gaps (S1–S4 in `PLAN.md`):

18. **Unknown effects are a commit.** A call whose effects can't be read off the call (an interpreter, a shell, a script, a task target, a package script, a git alias) is a commit point, not just a write. The cost is one extra passing run before an approved edge replays; the alternative let `./deploy.sh` or `python -c '...git push'` replay after two runs under `approve '*'` (`repro/S1_output.txt`). Operators who know better say so per edge (`--not-commit`). Known-local runners are exempt so that test and build steps, the most common opaque calls, stay ordinary approved writes. The synthetic suite is unchanged task for task (its `python -m pytest` is a local module, and `git add`/`commit`/`rm` are local).
19. **Match where programs run, not where words appear.** Matching patterns as a subsequence of any command flagged 28 of 51 realistic commands wrongly (`apt install curl`, `git stash push`, `gh pr view`). Patterns now match only at program positions, git's subcommand is found by parsing its global options, and anything the shell decides at run time (a non-literal program or git subcommand) counts as a commit instead. 44 of 51 are now right; the other 7 are the documented exceptions above.
20. **Taint instead of rewriting.** Git reads trust the repository's config (`trust_repo_config`, default true), but a run that edits git config or metadata loses that trust for its remaining steps. Taint is computed from the run's own calls at decision time (`replay.option`), so it needs no state and never outlives the run.

Decision made after the tau-bench misroutes (B1 in `PLAN.md`):

24. **Only the task can choose between irreversible actions.** On tau-bench, `get_order_details` on a pending order is followed by a cancel in some tasks and a modify in others, and in airline by a cancel or by nothing. The observation is the same; the task says which. A rule like `status == pending → cancel` is right in 4 of 5 examples, which met the purity bar (0.8), and a majority is how T0 works. For reads and ordinary writes that trade is fine: a wrong read costs a model call, and side exits and negatives learn from it. A wrong cancel can't be undone, and the negative arrives only after the damage. So a commit point is replayed without the model only when the evidence shows no alternative: the model never chose anything else after this history (pooled over every context of the step, from the root path to the last edge, so a narrow context that has seen only cancels doesn't hide a general one where modifies happened too), or a rule on the input separated the alternatives perfectly (`features.learn_decision_list` now records each rule's `leak` over every example at the node) with as much support as a task-word rule and on a task like its examples. Every other case gets a T2 call. It shows the task, and the model's pick is a labelled example like any T2 pick. The cost was measured rather than assumed. On tau-bench retail train it is +0.03 small calls per task and no extra full calls, because the T2 call carries the values that a T3 fill used to ask for. On the synthetic suite (seeds 0–5, edges approved) it is +0.02 to +0.06 full calls per task, from the rare `processed → transfer` branch, which no longer replays on 2–4 examples, and 0.4–1.3 points of served share, with success unchanged. Also considered: a higher confidence threshold for commit points. The misrouting rule had confidence 0.70 and a correct cancel rule has about the same, so any threshold that stops one stops the other. Always using T2 for commit points was also considered: it costs a small call even where the model has only ever done one thing.

### Learning from failed replays and from where the model stops

Decisions made after the seed-3 regression (see Results):

12. **Failed replays are evidence against the choice that made them.** A replayed step in a failed run is a *negative* for its edge at the contexts of that step. Negatives lower the edge's purity (T0). They also count as misses for the decision-list rules that predict that edge on that input (T1). A failed run also fails every other replayed step in it, so only the *excess* counts: negatives beyond a failure rate of `1 - purity` among all replays of the same choice, passing ones included (`features.excess_negatives`). Twenty passing replays and one failure change nothing. One failure against two passing replays does.
13. **Task-word rules must earn T1.** A decision-list rule on a task word replays only once `task_rule_support` (default 5) model-chosen examples support it. Observation rules need 2. Each excess negative adds that many again. Until then the rule's branch goes to a T2 call, or to T4 if T2 is off. The model's pick is a labelled example, so a chance rule is broken by the first input it would have misrouted, while a real one reaches its support within a few tasks. In the seed-3 scenario (`tests/test_learning.py`), the typo task under `src/` gets a T2 call instead of `git rm`, and the rule disappears.
14. **A minority choice blocks T0 until it is outnumbered 8 to 1.** When the model has chosen more than one child at a node, the leading child's share gets one pseudo-count against it. So 4 choices against 1 (80%) is not enough to replay without looking at the input, and 8 against 1 is. Without this, seed 3 replayed `git rm pyproject.toml` into a version bump at task 8.
15. **The model's decision to stop is a choice too.** When a forwarded response has no tool call (and did not stop for `max_tokens`/`length`), `complete()` records `runs.ended_after`. For passing runs, the builder adds an END choice at the contexts after the last step. END counts toward the node's evidence (`nodes.n_end`, included in `n_pass`) and can be a decision-list label. When END leads at the deciding context, the request goes straight to T4 with reason `end@…`, with no subcall and no replay. END is never replayed: the final answer always comes from the model. A starved specific context that has only seen END also vetoes a back-off proposal. In the synthetic suite, 192 of 200 final answers at seed 0 are recognised as END. The suite had no end-of-task subcall to save: no failed subcall there was followed by the final answer, before or after this change, and small calls per task are unchanged. The waste shows up when an n-gram context has a child after the last step, as in `test_t2_resolves_ambiguous_node` (3 small calls → 2) and `test_final_answer_is_recorded_as_end_and_not_proposed_again`, where it also prevents a wrong T0 replay.
16. **A task-word rule is trusted only on tasks like those that support it.** Support alone can be reached by chance: at seed 3 the first 5 delete tasks all named `src/` paths and no typo task had yet, so `task~src → git rm` was proven when the first `src/` typo task arrived. Each task-word rule now stores the task-word sets of its supporting examples (distinct, the most recent 40). T1 replays on it only if the task's best Jaccard similarity to one of them is at least `task_rule_similarity` (default 0.5; 0 turns the gate off). Otherwise the step goes to T2 with reason `unproven_rule`, and the model's pick becomes an example, so each new kind of task costs one small call, once. The stored sets keep only the words that at least two supporting examples share, while the task keeps all of its own. A word seen in one example only (a file name, a typo word, a version) says nothing about the kind of task, and with those words kept, every typo task with a new typo looked new: in read-only mode that cost 0.033 extra small calls per task instead of 0.023. Cutting the task's words the same way made the gate too lenient, and it missed seed 3 task 13, since the words that mark a new kind of task (`correct`, `ship`) are exactly the ones no example has. Observation predicates (`err==false`, `json.status=="pending"`, `obs~line`) get no such gate: they test the state the choice depends on, and similarity over the rest of an observation (file contents, order details) would call almost every input new. Over seeds 0–5 with edges approved, observation rules made 1,257 T1 replays (1,076 on features, 181 on `obs~line`), and none of them was in a failed run.
17. **Failures count per kind of task, not per rule.** A task-word rule also stores the task-word sets of the inputs it replayed into failed runs (`nx`, leaving out any set that also supports it, since the same words then both passed and failed). T1 needs the task to be more similar to a supporting example than to any of them. Before, the negatives of decision 12 were pooled per rule: after N passing replays on one kind of task, a second kind that shares the rule's word needed about N/4 failed runs before the excess showed, and the demotion then also sent the first kind to T2. Now, with N = 20, the second kind fails once and the first kind keeps T1 (`test_failures_count_per_input_class_not_per_rule`; it failed 6 times before). The pooled excess still applies, for failures that task words can't separate. `excess_negatives` also no longer returns float residue (`1 − 0.2·5` was 2.2e-16, not 0).

### Runs and episodes

An **episode** is one task: every turn of a conversation since the last task boundary (`dialects.episode_of`, the same rules for Anthropic and OpenAI chat). The task is the episode's first user text. A later user turn either continues the episode or starts a new one:

- **Continues, as a `steer` step**, when the user cut in while the agent was working: text next to tool results, an interrupt (`[Request interrupted by user…]`, the marker itself dropped), or a user message with no finished agent turn since the previous one.
- **Continues, as a `yes` / `no` / `text` step**, when the agent's turn ended by asking something: a question that isn't a generic closer ("anything else?") or a confirmation prompt (`(yes/no)`, "shall I", "do you want", "confirm"). `yes` and `no` are bare confirmations and refusals ("Yes, please proceed with the cancellation." is `yes`; "yes, but only #W2" is `text`). A bare "ok" or "no thanks" after a turn that asked nothing also continues, as `text`.
- **Starts a new episode** otherwise: a new request after the agent finished without asking. Claude Code's pattern, one conversation with several prompts, is several episodes. Consecutive user messages count as one turn, and Claude Code's local slash-command transcripts (`<command-name>`, `<local-command-stdout>`) are dropped from the text.

`episode_mode` (config) or the `X-TreeJIT-Episode` request header overrides this: `conversation` makes the whole conversation one episode (tau-bench: one task, many user turns), `turn` makes every user message a new one (the behaviour before multi-turn episodes).

A **user step** is a pseudo-call `$user:<kind>` whose observation is the user's text. Recognition gives it an edge like any call, so "after the user said yes" is a context, and the step after it is learned there. Bindings (`$obs[-1]`, an email the user typed) and features read it like any observation. The builder never makes it a choice: no user edge is ever replayable, so replay never produces a user turn. When the user answers a finished agent turn (`yes`/`no`/`text`), the agent's text reply before it is END evidence at that context: the model chose to stop and ask there, so that request goes to the model, as the final answer does. `yes` and `no` are different edges, so a write learned after "yes" is never proposed after "no". A T0 write after a `yes` still needs approval like any write, and a commit point also needs `promote_runs + 1` passing runs (`tests/test_episodes.py` replays `cancel_pending_order` after "yes" under `approve '*'`).

A **run** is one episode of one conversation. Its id is, in priority order:

1. `X-TreeJIT-Run: H`: `H`, or `H.<task hash>` for another task under the same header.
2. A harness session id: Claude Code's `metadata.user_id` (`…_session_<uuid>`, or a JSON object with `session_id`; parsed defensively), the `X-Claude-Code-Session-Id` header, or OpenAI's `prompt_cache_key`. The id is `r_` + hash(family, session, first user text, episode index, task hash), known from the episode's first request.
3. Otherwise hash(family, first user text of the conversation, the conversation's first assistant turn, episode index, task hash). The first assistant turn is its tool-call ids, or its text when it made no call. Weak ids (`call_0`, `toolu_01`, short or counter-like, `model.weak_call_id`) are salted with the first observation. So a conversation's first request gets its run id when its response arrives, or, with weak ids, one request later, when treejit fills in the rows that waited. OpenAI's `user` is mixed in but never names a run alone: it identifies a person, not a conversation.

Two rules keep conversations from merging. A run whose recorded steps the conversation doesn't continue (different call ids at its first or last step, or more steps than the request has) is someone else's, so the next id is tried (`<id>.2`, `.3`, …). And **a run with an outcome is never extended.** A conversation that goes on after its outcome (a Stop hook reported after the agent asked a question, then the user said "yes") continues in a fork `<id>.2`. The fork holds the whole episode, but the steps it copied (`runs.inherited`) are context only: they were counted in the run they came from. An outcome posted for `H` also reaches its forks that have none yet.

Decisions made while adding them (T2, M1 and E4 in `PLAN.md`):

21. **User turns are context, never choices.** Making the user's reply an edge lets the tree condition on it with the machinery it already has (root paths, n-grams, guards, bindings). Keeping it out of the choices keeps replay from speaking for the user, and turns the question before it into the END evidence it is.
22. **The boundary is decided by the agent's last turn, not by the user's words.** Whether the agent asked something is visible, and it is what makes a reply a reply. Short-reply heuristics alone would have merged "now write tests" into the previous task, and a question heuristic alone would have split "yes" after "I can also commit this." The heuristic is stateless (every request re-parses the whole history the same way), and the header or `episode_mode` overrides it where the harness knows better.
23. **Identify the conversation, then the task in it.** The first call id alone merged conversations whose backend numbers calls (`repro/T2_out.txt`: three conversations and a later `rm -rf build` in one run marked pass). Session id or first turn, then episode index, then task hash, with the continuation check and forks as a backstop, keeps each id stable across the requests of one episode and distinct across conversations.

### Frontier prefix compaction (opt-in)

With `compact = true` (`TREEJIT_COMPACT=1`), a request that goes to the model (T4) is forwarded with the raw observations of *verified replayed steps* replaced by a short deterministic digest. Only the forwarded copy changes. The harness's own history, and the trace treejit records, keep the full text. The code is in `compaction.py`, called from the forward path of `engine.handle`.

```
[treejit: replayed & verified step — Bash(python -m pytest -q) → ok, 21 lines, 1001 chars; first line: "…… [ 34%]"; last line: "233 passed, 5 warnings in 6.93s"]
[output elided by treejit; call the tool again if you need it]
```

For a JSON observation, the digest shows the top-level keys with small scalar values instead of a first line, for example `json {"order_id": "#W1", "status": "pending", "items": [3]}`.

Rules:

1. **Eligible.** A step is eligible only if all of these hold:
   - its call id carries the replay marker;
   - its edge is known at the node encoded in the id;
   - its observation satisfies that edge's (non-empty) learned postcondition;
   - its features show no error.

   Model-chosen, side-exited and errored steps are always sent in full.
2. **Keep the last few.** The last `compact_keep_last` (default 3) observations of a request are sent in full. Observations under `compact_min_chars` (default 400) are left alone.
3. **Keep what the next decision reads.** An observation is kept when a binding rule (`["x", ["obs", k], …]`, including those nested in `fmt`, and `case`), a guard, or a decision list of any child of any current frontier context (root path and n-gram) reads it. Binding rules reach back at most 3 observations, so this only matters when `compact_keep_last` is below 3. The older "path" rule (keep what an earlier step's bindings read) is off by default, `compact_keep_path = true` restores it: decisions are made on the harness's uncompacted body, so it protected nothing, and dropping it changed no trajectory on seeds 0–5 (PLAN C2).
4. **Append-only (`compact_mode = "first_sight"`, the default).** A step may be compacted only in the *first* forwarded request that contains it. Once it went upstream in full it stays full, and once compacted it stays compacted, even after a tree rebuild or a restart. Consecutive forwards of one conversation are therefore byte-identical up to the previous request's last message (`fwd[i].messages[:len(fwd[i-1].messages)-1] == fwd[i-1].messages[:-1]`), and the provider's prompt cache keeps the whole prefix. In practice this compacts the steps of a *burst*: more than `compact_keep_last` steps replayed between two frontier calls, which is how treejit is used once edges are approved.
   - Every decision is stored in `compactions`, keyed by call id and observation hash; a NULL digest means "sent in full". Digests are pure functions of the call and its observation (sorted-key labels, source-order JSON keys, no time or random ids).
   - Without a row (pruned, another instance, compaction just switched on) the decision is reconstructed from the conversation: the first forward that contained step *i* is the request just before the first model-chosen step after *i* (or the current one), and step *i* was in its keep-last window iff *i* ≥ *f* − `compact_keep_last`.
   - Steps of earlier episodes of the same conversation (before the last user text message) keep their stored digests.
5. **Epoch re-compaction (`compact_mode = "epoch"`, opt-in).** As first-sight, but when the previous forward of this conversation is older than `compact_epoch_ttl` (default 300 s, the 5-minute cache TTL; set 3600 for the 1-hour TTL), the cache entry has expired and the next request rewrites the whole prefix anyway. That request is compacted like `window` below, and the result becomes the new sticky state. A conversation is identified by its first tool call id, and its last forward time is kept in `compact_convs`.
6. **`compact_mode = "window"`** is the behaviour before this fix: the keep-last window moves on every request and the step that leaves it is compacted. It sends the fewest raw tokens, but it changes an earlier message on every forward, so it only suits providers without prompt caching.

Each forwarded request records `compacted N obs/C chars` (plus `(epoch)` for an epoch re-compaction) in its note and the characters saved in `requests.compacted_chars`. `treejit prune` also drops the decisions of runs idle longer than `compact_retention_days` (default 7; `--compact-days N`, 0 keeps them), and the forward path does the same at most once an hour. A pruned conversation that comes back is forwarded with the same bytes, because the decisions are reconstructed.

**Why first-sight, and the prompt cache.** Before this fix the README claimed the cache survived compaction. It didn't: in the moving window, the step that leaves the window changes from full to compacted in every request, so the previous request's cache entry never matches past that step. `tests/cache_model.py` bills request sequences the way the Anthropic prompt cache does (reads 0.1×, writes 1.25×, a breakpoint after `system` and one at the end of the conversation, 5-minute TTL), driving the real engine. The output is in `repro/C1_after_small.txt` and `repro/C1_after_bigsys.txt` (the latter with a 62.5k-character, Claude Code-sized system prompt). Billed input tokens against compaction off:

| pattern (15 steps, ~3.3k-char observations) | window | first_sight | epoch |
|---|---|---|---|
| dense: forwarded after every replayed step | +119% (+43% big system) | ±0 | ±0 |
| interleaved: every other step is the model's | +76% (+20%) | ±0 | ±0 |
| bursty: 4 replayed steps between frontier calls | +9% (+3%) | **−20%** (−8%) | −20% (−8%) |
| interleaved, a 10-minute pause before forward 7 | +15% | ±0 | **−27%** |
| bursty, a 10-minute pause before forward 3 | −28% | −20% | **−45%** |

First-sight never bills more than compaction off: the bodies are the same sequence with some observations shorter, and every prefix is preserved. `tests/test_compaction.py` checks this for all three patterns and both breakpoint placements, and checks the append-only invariant over consecutive forwards. Chunking the window (advancing it every 3 or 5 steps) still cost +33% / +10% on the dense pattern in the original repro (`repro/C1_out_small.txt`), so it was not kept.

On the simulated bench (`repro/C1_bench_billed.py`: each task's full calls billed as one conversation, subcalls left out), the edges-approved traffic is bursty, and first-sight is also the cheapest mode:

| seed | billed / task: off | first_sight | window | compacted chars / task (first_sight / window) |
|---|---|---|---|---|
| 0 | 1,574 | 1,367 (−13.1%) | 1,391 (−11.6%) | 611 / 687 |
| 1 | 1,557 | 1,360 (−12.6%) | 1,385 (−11.0%) | 584 / 665 |
| 2 | 1,528 | 1,316 (−13.9%) | 1,331 (−12.9%) | 601 / 674 |
| 3 | 1,555 | 1,351 (−13.1%) | 1,362 (−12.4%) | 595 / 627 |
| 4 | 1,577 | 1,382 (−12.4%) | 1,413 (−10.4%) | 575 / 680 |
| 5 | 1,591 | 1,366 (−14.2%) | 1,394 (−12.4%) | 655 / 740 |

With a Claude Code-sized system prompt (`--bigsys`, 62.5k chars, written once per task) the savings shrink to −0.8% to −1.0% for first-sight and −0.7% to −0.9% for window. Trajectories are identical in all three modes.

**What else breaks the prefix** (measured with the same model, `repro/C1_after_*.txt`, last table):

- **Frontier hints** are appended to the last user message of a forwarded request and are not in the next one (the harness never saw them). With a breakpoint the harness placed itself on its last block (Claude Code does this), the hint sits after it and costs only its own tokens (+0.3% to +2.4% with `hints = always`). With top-level *automatic* caching the breakpoint lands on the hint, so no hinted request's cache entry is ever read again: +156% to +247% billed with `hints = always` (+58% to +65% with the big system prompt). The default `hints = failures` only hints at nodes with failed edges. On models that enforce preserved thinking, deleting the hint is also a history edit. Not fixed here (`dialects.inject_hint`): the fix is to pin an explicit breakpoint on the last harness block before appending the hint, or to keep hints in later requests.
- **The thinking drop.** `prepare_forward` removes `thinking` once the episode has a replayed step. Toggling thinking invalidates the messages cache on every model, but it happens at most twice per episode (on → off at the first forward after a replay, off → on when a new episode starts), and the prefix it invalidates is usually short: +0.1% in the model. On models where thinking can't be disabled (omitting `thinking` means adaptive), the drop disables nothing.

**Results.** Averaged over all 200 tasks in `treejit+ok+compact` (first_sight) vs `treejit+ok`:

| seed | tokens / task (compaction off) | tokens / task (compaction on) | change | compacted chars / task | success (off → on) |
|---|---|---|---|---|---|
| 0 | 2,113 | 1,948 | −7.8% | 611 | 200 → 200 |
| 1 | 1,892 | 1,733 | −8.4% | 584 | 200 → 200 |
| 2 | 2,229 | 2,061 | −7.5% | 601 | 200 → 200 |
| 3 | 2,369 | 2,208 | −6.8% | 595 | 199 → 199 |
| 4 | 2,323 | 2,167 | −6.7% | 575 | 199 → 199 |
| 5 | 1,877 | 1,699 | −9.5% | 655 | 200 → 200 |

- Trajectories (success, calls, tiers) are identical, task for task, with compaction on and off.
- Compared with the moving window with the path rule (the previous default: 230–674 chars/task, −3.2% to −8.5%), first-sight without the path rule compacts more, not less: dropping rule 3b more than makes up for keeping steps that were first sent inside the window.
- Retail is unaffected: its observations are under 400 characters.
- Compaction only acts on frontier calls. With edges approved there are few of them, often just the final answer.

## Known limits

- **Tool execution.** treejit sees the model API, not tool execution. It backtracks its policy, not the world.
- **Rebuild cost.** The tree is rebuilt in full for a family on each outcome, from its last `max_runs` runs. Pure per-step work is memoized across rebuilds: tokenization, shapes, observation features, node ids, the policy verdicts on reference calls, and each shape's template (a left fold that resumes when the instances only grew). Binding search parses and indexes each observation once instead of once per candidate, and decision lists skip feature values no two examples share. On the sim coding family, a rebuild after one more outcome takes about 90 ms at 100 runs, 0.4 s at 1000 and 0.85 s at 2000 (it was 0.18, 1.0 and 2.2 s). On JSON-heavy observations (tau-bench-like retail, about 800 characters each) it takes 0.2 s at 115 runs and 0.6 s at 300 (it was 1.2 and 6.2 s; `repro/P1_timing.out`). Under the proxy the build runs off the event loop, and `/health` answers within about 10 ms during a 2000-run rebuild. The build is still O(runs) per outcome: an incremental builder that rebuilds only the nodes an outcome touches isn't implemented, because blame decay, `max_runs` windows and per-node instance caps make node rows depend on more than their own runs.
- **Family keying is heuristic.** A volatile block the masks don't recognise (free text other than git output) that makes up more than about 10–20% of a prompt splits families. So does a large per-project `CLAUDE.md`: two projects on the same harness then get separate trees, which is usually what you want. A harness upgrade that changes more than about 20% of the prompt starts a new family. Counts are halved every 64 distinct prompts, so the stable set follows slow drift. Families migrated from the old prefix keying keep its `startswith(prefix)` fallback, and with it the old over-merging of agents that share that prefix. If two processes add members to the same family at the same moment, one member's counts can be lost (last write wins).
- **Kinds of tasks are told apart by their words.** A new kind of task that uses the words of a known one (`delete P carefully` next to `delete P`) still gets one misroute from a task-word rule before its failed run blocks it (decision 17). The similarity threshold is global, and it doesn't weigh words.
- **Observation rules have no similarity gate** (decision 16), except where they choose a contested commit point (decision 24). A chance observation predicate, such as a line of file content that only one kind of task has seen so far, needs 2 supporting examples and is refuted only by the pooled negatives of decision 12.
- **The first alternative at a commit point can still be misrouted once.** Decision 24 needs evidence of an alternative. After `promote_runs + 1` passing runs in which the model only ever cancelled after a lookup (at every context, general ones included), the first task that wants a modify there gets a T0 cancel. `approve '*'` accepts that risk; approving commit edges one by one, or waiting for more runs, doesn't. The T2 call for a contested commit point also trusts the model to read the task. A real model that picks the wrong option there acts as if it had made the call itself.
- **END needs a passing run and a visible final answer.** A forwarded inline stream records where the model stopped only if the caller reads it to the stop reason; one closed earlier is logged as 499. Runs without an outcome never add END evidence.
- **Inline mode is synchronous.** `AsyncAnthropic` / `AsyncOpenAI` are rejected by `wrap()`; use the proxy for async harnesses. `messages.stream(output_format=...)` (structured outputs), `beta.messages`, `with_raw_response` and `chat.completions.stream()` pass through to the real client unrecorded (the run header is still stripped from `messages.stream`). A `ReplayStream` has `.response = None`, so `request_id` on a replayed `MessageStream` is unavailable.
- **Compaction and the prompt cache.** The default first-sight mode never changes a message already sent, so it compacts only steps replayed in a burst before a frontier call; a step first sent inside the keep-last window stays full until an epoch (`compact_mode = "epoch"`, only after the cache went cold). The cost model is first-order: no 20-block lookback, no minimum cacheable length, one breakpoint layout. Frontier hints still break the cache under top-level automatic caching (see above).
- **Commit-point coverage is a list, not a proof.** Programs outside the categories in [Replay safety](#replay-safety) (`cp`, `rm`, `docker build`, `pip install`, an unknown CLI) are writes, not commit points, so an approved edge replays them after `promote_runs` passing runs. Commands the shell builds at run time from data we don't see (a function or alias defined in an earlier call, `$(cat cmd.txt)` inside a script file, a Makefile target that pushes) are only caught when the call itself shows an opaque executor. Tools with their own exec hooks (`sed` `e`, `vim -c`, `git difftool`, repo-configured `pre-commit`) are writes, not commit points.
- **Repository taint sees calls, not the disk.** It catches writes to git metadata and config that a call names. A `git pull`/`checkout`/`apply` that brings in a `.gitattributes` or an embedded bare repository, a `cp -r` of a bare repository under another name, or a file tool writing a bare repository's `config` doesn't taint the run: that is the trusted-checkout assumption (see `trust_repo_config`). A `cd` in an earlier call isn't tracked either.
- **Run identity.** Without a header or a session id, a run is named from the conversation's first user text and first assistant turn. Two conversations that are identical up to the current step (same text, same call ids, same observations) share a run until they diverge, and then the later one continues in a fork. That takes weak call ids or a text-only first answer. A harness that clears old tool results (Claude Code's context editing) changes the salt of weak ids, which starts a new run. `/compact` replaces the history, so the conversation after it is a new conversation to treejit, with a new root path. A session id alone doesn't survive it either, since the key includes the first user text. A request row whose conversation never sends another request after a weak-id first response keeps `run_id` NULL. The deferral is kept in memory, so a restart between the two requests has the same effect.
- **Episode boundaries are a heuristic.** A new task typed after the agent asked a question ("Want me to open a PR?" / "No. Now fix the login bug") continues the episode as a `text` step, and a follow-up request after a turn that asked nothing ("now also handle the error case") starts a new episode, whose root path starts fresh. Set `X-TreeJIT-Episode` or `episode_mode` when the harness knows where tasks begin. The agent's own text isn't part of the context, so an acknowledgement "yes" (after a turn that asked nothing) is recorded as `text`, not `yes`. `show` and `export` draw the root path only down to a user step; the nodes after it appear under the macros.

## Development

```bash
pip install -e '.[dev]' -e bench
pytest -q                                   # ~745 tests (mostly the policy tables); tau-bench tests skip without TAUBENCH_PATH
python -m treejit_bench --tasks 200 --out bench_out [--via-proxy] [--seed N] [--family coding|retail|mixed] \
    [--modes baseline,treejit,treejit+ok,treejit+ok+compact] [--payload none|tau|claude-code] [--cache] [--rebuild-every K]
TAUBENCH_PATH=../tau-bench python -m treejit_bench --suite taubench --tau-split test --out tau_out
```
