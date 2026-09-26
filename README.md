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
| Trace recorder, stable system-prompt prefix learning, span-preserving shell tokenizer | done |
| Tree builder: anti-unification, provenance bindings, last-k-edge macros, root depth cap D | done |
| Failed replays as negative evidence, earned task-word rules, END (the model stops here) as a choice | done |
| T0 replay / T1 guarded branch, postcondition side exits, confidence budget, hard cap K, batching | done |
| Read-only allowlist, commit points + operator approval, soft tombstones, node-local T4 hints | done |
| CLI: `serve show runs explain outcome pending pin approve revoke prune export build stats`; HTML / Mermaid / SKILL.md export | done |
| Benchmark harness (synthetic suite, inline and real-proxy modes) + learning-curve report | done |
| T2 choose / budget checkpoint, T3 hole filling (one small forced-tool subcall, proxy and inline) | done |
| Frontier prefix compaction (verified replayed observations digested in forwarded requests) | done, **opt-in** (`compact = true`) |
| Macros-as-tools, OpenAI Responses API, inline-mode streaming replay | not yet |
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
| treejit, edges approved + compaction | 151–200 | **1.06** | 0.90 | **1,981** | **99%** | 100% | 2.6 s |

Seeds 0–5. Success is over all 200 tasks; the other columns are tasks 151–200 (plain agent / allowlist / approved / approved + compaction):

| seed | success (of 200) | full calls / task | small calls / task | tokens / task | served |
|---|---|---|---|---|---|
| 0 | 193 / 198 / 200 / 200 | 6.02 / 3.70 / 1.06 / 1.06 | – / 0.14 / 0.90 / 0.90 | 6,196 / 4,820 / 2,038 / 1,981 | 0 / 48 / 99 / 99% |
| 1 | 189 / 198 / 200 / 200 | 6.36 / 4.20 / 1.02 / 1.02 | – / 0.02 / 0.28 / 0.28 | 6,684 / 5,311 / 1,637 / 1,560 | 0 / 41 / 100 / 100% |
| 2 | 192 / 198 / 200 / 200 | 6.32 / 4.12 / 1.00 / 1.00 | – / 0.00 / 0.78 / 0.78 | 6,711 / 5,279 / 1,931 / 1,734 | 0 / 39 / 100 / 100% |
| 3 | 189 / 198 / 199 / 199 | 6.26 / 4.18 / 1.04 / 1.04 | – / 0.08 / 1.08 / 1.08 | 6,590 / 5,527 / 2,234 / 2,127 | 0 / 43 / 99 / 99% |
| 4 | 185 / 198 / 199 / 199 | 6.28 / 4.08 / 1.02 / 1.02 | – / 0.10 / 1.20 / 1.20 | 6,744 / 5,513 / 2,315 / 2,248 | 0 / 45 / 100 / 100% |
| 5 | 192 / 198 / 200 / 200 | 6.66 / 4.58 / 1.00 / 1.00 | – / 0.00 / 0.02 / 0.02 | 7,401 / 6,014 / 1,452 / 1,407 | 0 / 37 / 100 / 100% |

- With edges approved (`treejit approve '*'`, which simulates operator review of write steps and commit points), almost every tool call is served from about task 40 on; the only full call left is usually the final answer. With the default read-only allowlist, only read steps replay, and T2/T3 rarely apply (their options must be replayable too).
- treejit never does worse than the plain agent on these seeds. The remaining failures are the simulated model's own shortcuts at steps it still decides (allowlist mode), and one misroute at seed 3 (task 13, below).
- **Seed 3 used to regress** (171/200 with edges approved, 86% success in tasks 151–200). The node after `git status` had learned the decision-list rule `task~src → git rm` from 2 delete-module tasks and 1 typo task in `README.md`. From task 13 on, it replayed `git rm` into every typo task whose file is under `src/`: 28 failed runs. The failures never reached the rule. Blame goes to edges, and `git rm` was right at that node for other tasks. Replayed steps never become examples, so the inputs the rule misrouted stopped producing evidence. Decisions 11–13 below fix this. One misroute is left (task 13): by then the rule had 5 supporting delete tasks and no counterexample, and that single failure demotes it.
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
| train | treejit, edges approved | 1–300 | 2.50 | 2.13 | 19,109 | 20,176 | 77% | **0.920** |
| train | treejit, edges approved | 251–300 | 2.04 | 2.04 | 15,677 | 16,573 | 83% | 0.900 (plain agent 0.920) |
| airline test | plain agent / allowlist / edges approved | 1–50 | 5.20 / 3.52 / 3.24 | – / 1.12 / 1.88 | 25,409 / 19,456 / 19,184 | | 0 / 40 / 47% | 0.98 / 0.98 / **0.96** |

- On the retail test split, every treejit mode has exactly the plain agent's reward: the 7 failing tasks are the oracle's slips, task for task. The read-only allowlist mode also matches the plain agent on train and airline.
- **With edges approved, treejit fails 7 retail train tasks that the plain agent passes** (149, 191, 197, 238, 243, 245 and 298), and one airline test task (29). It also avoids one slip (retail train 69, where a replayed cancel used the correct reason). This is PLAN B1, with a sharper diagnosis. In every retail case, T1 or T3 replays the irreversible `cancel_pending_order` on a pending order that the task wants *modified*, right after `get_order_details` on it. The following `modify_pending_order_*` then fails with "non-pending order cannot be modified". For example, `treejit explain tau-retail-train-0-238` shows step 6, `T1@r6 cancel_pending_order(order_id=$order_id, reason=$reason) conf=0.70`, on `#W5995614`, which the ground truth modifies. In airline task 29, a read-only task, `T1@r2 cancel_reservation(reservation_id=$reservation_id) conf=0.89` cancels the first reservation looked up. The id is the one just looked up, so the binding is consistent. What goes wrong is the *choice* of a write at a node where cancel and modify (or nothing) both follow a lookup and only the task text says which is meant. With the default allowlist these steps go to the model, so nothing is lost.
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
export ANTHROPIC_CUSTOM_HEADERS="X-TreeJIT-Run: $(uuidgen)"   # optional; run ids are derived otherwise
claude
```

**Outcomes.** Only runs reported as passing ever promote an edge.

```bash
curl -s localhost:8787/outcome -d '{"run_id": "latest", "outcome": "pass"}'
treejit outcome <run_id|latest> fail --reason "tests failed in CI"
```

In Claude Code, a `Stop` hook that runs your verifier and then posts `/outcome` closes the loop. Outcome `error` (timeouts, 429s) is recorded but never counted as evidence.

**OpenAI-compatible harnesses:** point `OPENAI_BASE_URL` at `http://127.0.0.1:8787/v1`.

**Inline mode** (harnesses you own, tests):

```python
from treejit import TreeJIT
jit = TreeJIT("treejit.db")
client = jit.wrap(anthropic.Anthropic())             # or an OpenAI client, or a callable body -> dict
client.messages.create(..., extra_headers={"X-TreeJIT-Run": "task-17"})
jit.outcome("task-17", "pass")
```

**Inspect and operate**

```bash
treejit show --ids          # tree with tiers, bindings, guards, decision lists, macros; short node/edge ids
treejit runs; treejit stats
treejit explain <run|latest> [--json]   # per-step timeline: who decided (model/T0/T1/...), why, tokens, first obs line
treejit pending [--family F] [--json]   # promoted edges held back only by policy: example call, evidence, approve commands
treejit approve --review                # walk the queue: y = at the listed nodes, e = everywhere, n/s = leave, q = stop
treejit approve <edge|'*'> [--node N]   # let replay cross a write edge / commit point
treejit revoke <edge> [--node N]        # same as approve --revoke
treejit pin <node> [edge]               # force-promote, protect from eviction
treejit prune --days 30 --min-hits 3
treejit export --format html|mermaid|skills --out ...
```

Ids (edges, nodes, families, runs) can be given as any unique prefix of at least 4 characters; the CLI prints 8.

Each tree edge has a tier: **hot** (replayable), **live** (promoted but blocked), **warm** (one passing run), **cold**, or **tomb**. `show` prints why a live edge doesn't replay: `LIVE:holes` (an argument has no binding), `LIVE:needs_approval` (not read-only, or a commit point), or `LIVE:commit_point_needs_evidence` (approved, but a commit point also needs `promote_runs + 1` passing runs). Only the last two appear in `pending`: approval can't fix holes or tombstones.

## How it works

| module | job |
|---|---|
| `dialects.py` | Parse requests into an *episode* (task + (call, observation) steps). Build replay responses as JSON or SSE. Accumulate upstream SSE for usage. Inject hints. |
| `families.py` | Tree key = tool schemas + the learned stable prefix of the system prompt (longest common prefix, trimmed to a line). |
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

- **T3 (fill).** The structure is decided: T0/T1 picks an edge that is live, not tombstoned, and allowed by the allowlist or approvals. But some variables have no rule, or their rule abstains for this input. The subcall shows the task, all calls so far (arguments truncated to 200 chars), the last 3 tool results (2,000 chars each), and the next call with `<placeholders>`. The model answers one string per hole (with an earlier run's value as an example), or `not_this_step`. Values are spliced in as data (one quoted shell word), then the call is re-rendered and re-checked. A read-only call must stay read-only, no new commit point may appear, and the call must still match the same edge shape.
- **T2 (choose).** Used in two cases. The first is a node with enough evidence that is ambiguous: at least 2 children the model has chosen, and the replayable ones account for at least 50% of those choices. The second is a *checkpoint*: the confidence budget runs out on an otherwise confident step, so the proposed step becomes option 1 and its siblings follow. The model answers an option number, or 0 for "something else" (→ T4). If the chosen option has holes, the same call asks for their values (`o<N>_<name>` fields). A T2 step restarts the confidence budget. The hard cap K still counts it as a replayed step.
- Subcalls use `small_model` if set (else the request's model) and `subcall_max_tokens`, and `t2` / `t3` switch them off. All four are config keys (or `TREEJIT_*` variables). Subcalls are logged in `requests` with tier `T2` / `T3` and their own usage, and `treejit stats` shows them. Edge savings are still measured against T4 calls only.
- Subcall steps are marked in the call id (`..._t3`, `..._t2`, `..._ck`), so recognition, side exits and the budget stay stateless.

Decisions made while adding them:

8. **A T2 pick is a model choice.** The model chose among known children, so the step is logged as not replayed and feeds purity and decision lists. T1 can then learn the branch and stop asking. Budget checkpoints and T3 steps are logged as replayed, because the tree proposed their structure.
9. **Value back-off.** Once T3 serves a hole at a general (n-gram) context, the more specific contexts stop collecting model-chosen evidence, so they never become the deciding context. Their value rules are still learned from every passing instance, so a hole may borrow the rule the same edge has at a more specific context. Without this, T3 replaced free T0 steps with small calls (the version-bump commit message) and cost more tokens than it saved: 1,439 tokens/task against 1,373 before T2/T3.
10. **One subcall, first step only.** Subcalls are only made for the first call of a response, and a T2/T3 step ends the batch.

### Replay safety

Replay emits a call with no model call and no human in the loop only if `policy.is_readonly` says the call is read-only. Otherwise the edge needs operator approval (`treejit approve`). A commit point (`is_commit_point`: `git push`, `curl`, `send_*`, ...) needs approval and `promote_runs + 1` passing runs. For shell commands (`policy.py`):

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
- **Commit points** are checked at the unwrapped program (`env git push`, `timeout 60 git push`). For a command that isn't read-only, they are also checked anywhere in it (`git -C repo push`, `xargs git push`, `find -exec git push`). Nested scripts count too: `$(...)`, backticks, `sh -c`, `eval`, `env -S`, and quoted text when a shell or `watch` in the line may run it (`echo 'git push' | sh`). Quoted text elsewhere is data, so `git commit -m "then git push"` is not a commit point.

Decision made while hardening it:

11. **When unsure, say "not read-only".** A false "no" costs one model call, or one approval. A false "yes" runs a write with no model call and nobody watching. Checks therefore accept what they understand rather than rejecting what they recognise as dangerous. The exception is `find`, whose side-effecting actions are a closed, documented set. On the synthetic suite the stricter policy changed nothing: read-only `treejit` mode served 44% of tool calls in tasks 71–120, before and after.

### Learning from failed replays and from where the model stops

Decisions made after the seed-3 regression (see Results):

12. **Failed replays are evidence against the choice that made them.** A replayed step in a failed run is a *negative* for its edge at the contexts of that step. Negatives lower the edge's purity (T0). They also count as misses for the decision-list rules that predict that edge on that input (T1). A failed run also fails every other replayed step in it, so only the *excess* counts: negatives beyond a failure rate of `1 - purity` among all replays of the same choice, passing ones included (`features.excess_negatives`). Twenty passing replays and one failure change nothing. One failure against two passing replays does.
13. **Task-word rules must earn T1.** A decision-list rule on a task word replays only once `task_rule_support` (default 5) model-chosen examples support it. Observation rules need 2. Each excess negative adds that many again. Until then the rule's branch goes to a T2 call, or to T4 if T2 is off. The model's pick is a labelled example, so a chance rule is broken by the first input it would have misrouted, while a real one reaches its support within a few tasks. In the seed-3 scenario (`tests/test_learning.py`), the typo task under `src/` gets a T2 call instead of `git rm`, and the rule disappears.
14. **A minority choice blocks T0 until it is outnumbered 8 to 1.** When the model has chosen more than one child at a node, the leading child's share gets one pseudo-count against it. So 4 choices against 1 (80%) is not enough to replay without looking at the input, and 8 against 1 is. Without this, seed 3 replayed `git rm pyproject.toml` into a version bump at task 8.
15. **The model's decision to stop is a choice too.** When a forwarded response has no tool call (and did not stop for `max_tokens`/`length`), `complete()` records `runs.ended_after`. For passing runs, the builder adds an END choice at the contexts after the last step. END counts toward the node's evidence (`nodes.n_end`, included in `n_pass`) and can be a decision-list label. When END leads at the deciding context, the request goes straight to T4 with reason `end@…`, with no subcall and no replay. END is never replayed: the final answer always comes from the model. A starved specific context that has only seen END also vetoes a back-off proposal. In the synthetic suite, 192 of 200 final answers at seed 0 are recognised as END. The suite had no end-of-task subcall to save: no failed subcall there was followed by the final answer, before or after this change, and small calls per task are unchanged. The waste shows up when an n-gram context has a child after the last step, as in `test_t2_resolves_ambiguous_node` (3 small calls → 2) and `test_final_answer_is_recorded_as_end_and_not_proposed_again`, where it also prevents a wrong T0 replay.

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
2. **Keep the last few.** The last `compact_keep_last` (default 3) observations are always sent in full. Observations under `compact_min_chars` (default 400) are left alone.
3. **Keep what decisions read.** An observation is kept when either of these reads it:
   - **current:** a binding rule (`["x", ["obs", k], …]`, including those nested in `fmt`, and `case`), a guard, or a decision list of any child of any current frontier context (root path and n-gram);
   - **path:** a binding rule of the edge any earlier step took, at any of that step's contexts or at the replayed node.

   `arg` rules read call arguments, which are never compacted. Binding rules reach back at most 3 observations, so with the default `compact_keep_last` everything the next decision can read is in the kept window anyway.
4. **Deterministic and monotonic.** The digest is a pure function of the call and its observation: sorted-key labels, source-order JSON keys, no time or random ids. Decisions are *sticky*: the first time a call id is compacted, its digest is stored in `compactions` (keyed by call id + observation hash) and reused verbatim on every later request, even after a tree rebuild. A step therefore never flips back from compacted to full, and the provider's prompt-cache prefix survives up to the newest step that left the keep-last window. The only override is rule 2 or 3 "current", which applies if the harness rewinds a conversation.

Each forwarded request records `compacted N obs/C chars` in its note and the characters saved in `requests.compacted_chars`. Older DBs get the column through a guarded `ALTER TABLE`.

**Results.** Averaged over all 200 tasks in `treejit+ok+compact` vs `treejit+ok`:

| seed | tokens / task (compaction off) | tokens / task (compaction on) | change | success (off → on) |
|---|---|---|---|---|
| 0 | 2,113 | 2,043 | −3.3% | 200 → 200 |
| 1 | 1,892 | 1,832 | −3.2% | 200 → 200 |
| 2 | 2,229 | 2,040 | −8.5% | 200 → 200 |
| 3 | 2,369 | 2,243 | −5.3% | 199 → 199 |
| 4 | 2,323 | 2,247 | −3.3% | 199 → 199 |
| 5 | 1,877 | 1,813 | −3.4% | 200 → 200 |

- Trajectories are identical, task for task, with compaction on and off. Compaction elides 230–674 observation characters per task.
- Retail is unaffected: its observations are under 400 characters.
- Rule 3 "path" is what limits the savings: the typo and bump `Edit`s bind `old_string` from the `Read` output, so the file is kept for the rest of the episode. Dropping that rule (an experiment, not an option) compacted 2.8× more (797 vs 288 chars/task at seed 0; tokens/task 1,894 vs 2,034) with unchanged success in this suite. That was measured before decisions 11–14, which changed the seed-0 trajectories slightly.
- Compaction only acts on frontier calls. With edges approved there are few of them, often just the final answer.

## Known limits

- **Tool execution.** treejit sees the model API, not tool execution. It backtracks its policy, not the world.
- **Rebuild cost.** The tree is rebuilt in full for a family on each outcome (tens of ms at a few hundred runs, capped by `max_runs`). An incremental builder is future work.
- **Stable-prefix learning.** A dynamic block early in the system prompt shrinks the learned prefix to whatever precedes it.
- **One misroute before a chance rule is caught.** A task-word rule whose first `task_rule_support` examples all agree by chance still replays once into the input that breaks it; the failed run then demotes it (seed 3, task 13). A higher `task_rule_support` trades small calls for fewer of these.
- **Negatives are counted per rule, not per input class.** A rule with many passing replays that starts misrouting a new kind of task needs several failures before the excess over the tolerated rate (`1 - purity`) shows. Only then is it demoted.
- **END needs a passing run and a visible final answer.** Streams consumed outside the engine (inline-mode streaming) don't record where the model stopped, and runs without an outcome never add END evidence.
- **Compaction and the prompt cache.** The keep-last window moves as the conversation grows. The step that leaves it changes from full to compacted once, which invalidates the cache from that message on. A chunked boundary (advancing the window only every few steps) would trade a little compaction for longer cache hits; it isn't implemented.
- **Run identity.** Without an `X-TreeJIT-Run` header, run ids are derived from the task and first tool-call id. Resuming the same task text in a new conversation starts a new run.

## Development

```bash
pip install -e '.[dev]' -e bench
pytest -q                                   # ~470 tests (mostly the policy tables); tau-bench tests skip without TAUBENCH_PATH
python -m treejit_bench --tasks 200 --out bench_out [--via-proxy] [--seed N] [--family coding|retail|mixed] \
    [--modes baseline,treejit,treejit+ok,treejit+ok+compact] [--payload none|tau|claude-code] [--cache] [--rebuild-every K]
TAUBENCH_PATH=../tau-bench python -m treejit_bench --suite taubench --tau-split test --out tau_out
```
