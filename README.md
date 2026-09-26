# treejit

An inference proxy with memory. treejit sits between an LLM agent harness and the model API. It learns a persistent execution tree from successful runs, and replays proven tool-call sequences without calling the model. The model is only called at the frontier: nodes the tree hasn't seen, branch points it can't resolve, and arguments it can't derive. Failed branches are recorded and avoided.

In short, it's a tracing JIT for agent trajectories, combined with a tree search whose tree persists across tasks.

```
harness ──► treejit ──► model API
               │  replayed step: tool call returned in ms, 0 tokens
               └─ frontier step: forwarded, recorded, learned from
```

## Status

This is the MVP from the handoff, plus two pieces pulled forward because the benchmark needed them: value branching and composite argument templates.

| Area | State |
|---|---|
| Anthropic Messages + OpenAI Chat Completions dialects, JSON and SSE (replay and pass-through) | done |
| Trace recorder, stable system-prompt prefix learning, span-preserving shell tokenizer | done |
| Tree builder: anti-unification, provenance bindings, last-k-edge macros, root depth cap D | done |
| T0 replay / T1 guarded branch, postcondition side exits, confidence budget, hard cap K, batching | done |
| Read-only allowlist, commit points + operator approval, soft tombstones, node-local T4 hints | done |
| CLI: `serve show runs outcome pin approve prune export build stats`; HTML / Mermaid / SKILL.md export | done |
| Benchmark harness (synthetic suite, inline and real-proxy modes) + learning-curve report | done |
| T2 checkpoints, T3 hole filling | **not yet**: those cases go to the model (T4) |
| Frontier prefix compaction (verified replayed observations digested in forwarded requests) | done, **opt-in** (`compact = true`) |
| Macros-as-tools, OpenAI Responses API, inline-mode streaming replay | not yet |
| tau-bench runner | not yet: only the synthetic suite has been run |

The core (`src/treejit`, except `proxy.py`) uses only the standard library. Proxy mode also needs `httpx` and `uvicorn`.

## Results (synthetic suite, 200 tasks, `python -m treejit_bench --tasks 200 --modes baseline,treejit,treejit+ok,treejit+ok+compact`)

A simulated agent works a mixed stream of coding tasks (typo fix / version bump / delete module, each with a flaky-test branch) and tau-bench-style retail tasks (branching on order status). The tree starts empty. The simulated model sees only the conversation. It is stochastic (argument formatting varies, free-form commit messages), and it takes a known-bad shortcut 6% of the time. Observations are realistically sized: `Read` returns a 45–65-line file, `pytest -q` prints 150–420 tests as progress rows plus a warnings summary, and `git status` lists untracked build junk in about a third of the tasks.

| mode | tasks | model calls / task | tokens / task | tool calls served by replay | success | sim. wall-clock / task |
|---|---|---|---|---|---|---|
| plain agent | 151–200 | 6.02 | 6,196 | 0% | 96% | 10.0 s |
| treejit, read-only allowlist | 151–200 | 3.70 | 4,811 | 49% | 100% | 6.1 s |
| treejit, edges approved | 41–50 | 1.90 | 2,692 | 85% | 90% | 2.9 s |
| treejit, edges approved | 151–200 | **1.24** | 1,647 | **95%** | 100% | 1.8 s |
| treejit, edges approved + compaction | 151–200 | **1.24** | **1,575** | **95%** | 100% | 1.8 s |

- The handoff target was >80% of steps served without a frontier call after ~50 tasks. The suite reaches 85% on tasks 41–50 once edges are approved (`treejit approve '*'`, which simulates operator review of write steps and commit points). With the default read-only allowlist, only read steps replay.
- Seeds 0–3 land at 80–95% served and 1.2–2.1 model calls/task over the last 50 tasks. Success over all 200 tasks is 199, 200, 200 and 185 for treejit (edges approved) against 193, 189, 192 and 189 for the plain agent. Seed 3 is the exception: early on, a T1 decision list at the node after `git status` learned the task-word predicate `task~src` from a handful of runs and replayed `git rm` into four typo tasks. That's a learner weakness (tiny-sample task-word stumps), not related to compaction.
- Running the same stream through the real ASGI proxy with SSE streaming (`--via-proxy`) gives identical numbers.
- The floor is about one model call per task, because the final answer is always generated. The remainder is holes like free-form commit messages (T3's job).
- **Caveats:** the model and its token counts are simulated (tokens ≈ prompt chars / 4, latency = 600 ms + 15 ms/output token), so treat the absolute numbers as illustrative. The real test is tau-bench or Claude Code traffic, which hasn't been run yet.

Report: [`docs/learning_curve.html`](docs/learning_curve.html) (model calls, tokens and replay share vs task index, with a table view).

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
treejit show --ids          # tree with tiers, bindings, guards, decision lists, macros
treejit runs; treejit stats
treejit approve <edge|'*'> [--node N]   # let replay cross a write edge / commit point
treejit pin <node> [edge]               # force-promote, protect from eviction
treejit prune --days 30 --min-hits 3
treejit export --format html|mermaid|skills --out ...
```

Each tree edge has a tier: **hot** (replayable), **live** (promoted but has holes or needs approval), **warm** (one passing run), **cold**, or **tomb**.

## How it works

| module | job |
|---|---|
| `dialects.py` | Parse requests into an *episode* (task + (call, observation) steps). Build replay responses as JSON or SSE. Accumulate upstream SSE for usage. Inject hints. |
| `families.py` | Tree key = tool schemas + the learned stable prefix of the system prompt (longest common prefix, trimmed to a line). |
| `shellwords.py` | Span-preserving shell tokenizer (quotes, `$(...)`, heredocs, operators). Replayed commands are spliced into the original text, so quoting survives. |
| `templates.py` | Edge = tool + arg keys + per-segment command heads (`git commit`, `npm test`). Calls with one shape are anti-unified per token into constants and variables. |
| `bindings.py` | Provenance search. Each variable binds to `$task` / `$obs[-k]` / `$arg[-k]` via extractors (JSON path, `key: value`, regex types, after-word, line, token), `fmt` templates (`Bump version to {$task.version}`) or `case` (value chosen by predicates). A variable with no rule is a hole. |
| `features.py` | Predicate set: error/exit code, empty, JSON field equals, substring, task keyword. Guards are conjunctions of stable features; branches are decision lists. |
| `builder.py` | Deterministic rebuild from the trace log on every outcome (bounded to the last `max_runs`): promotion, bindings, guards, postconditions, credit assignment, tombstones, tiers, priority score. |
| `compaction.py` | Opt-in frontier prefix compaction: digests of verified replayed observations in forwarded T4 requests (below). |
| `tree.py`, `replay.py` | Stateless recognition (root path ≤ D, then last-k n-grams). Climbs T0 → T1 → T4, bounded by budget, cap, allowlist and commit points. |
| `proxy.py`, `inline.py`, `cli.py`, `export.py` | The two entry points, CLI, and views. |

### Decisions made during implementation

The handoff didn't specify these. Each one came from a failure seen in the benchmark:

1. **What the model would choose is learned only from steps the model chose.** Replayed steps still count toward success and failure, but not toward branch purity or decision lists. Otherwise replay reinforces its own guesses. The benchmark showed a misrouted-but-harmless read becoming "certain".
2. **Side exits teach.** A replayed step that breaks its postcondition is a miss against the context that chose it. If the model then recovers and the run passes, the model's choice is recorded as the correct label at that context (DAgger-style).
3. **Back-off needs agreement.** When the root path has too little evidence, a less specific macro context may decide, but only if it proposes something the more specific contexts have seen the model do.
4. **Decision lists have no catch-all default.** An input no rule fires on goes to the model. Rules are scored with a penalty for firing on other labels' examples.
5. **Bindings may abstain.** A rule that is never wrong and correct on a clear majority is kept; where it can't produce a value, that step goes to the model. A node that mixes two task types therefore keeps working for the majority instead of becoming a hole.
6. **Stateless replay bookkeeping.** Replayed tool-call ids encode the deciding node and its confidence (`toolu_tj_<node>_<conf><rand>`), so side exits and the confidence budget need no server-side session state.
7. **Extended thinking.** Replayed assistant turns carry no signed thinking blocks, so a frontier call that follows replayed turns in the same episode is sent without `thinking`.

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
| 0 | 2,109 | 2,034 | −3.6% | 199 → 199 |
| 1 | 2,044 | 1,995 | −2.4% | 200 → 200 |
| 2 | 2,653 | 2,340 | −11.8% | 200 → 200 |
| 3 | 2,849 | 2,666 | −6.4% | 185 → 185 |

- Trajectories are identical, task for task, with compaction on and off.
- Retail is unaffected: its observations are under 400 characters.
- Rule 3 "path" is what limits the savings: the typo and bump `Edit`s bind `old_string` from the `Read` output, so the file is kept for the rest of the episode. Dropping that rule (an experiment, not an option) compacts 2.8× more (797 vs 288 chars/task at seed 0; tokens/task 1,894 vs 2,034) with unchanged success in this suite.
- Compaction only acts on frontier calls. With edges approved there are few of them, often just the final answer.

## Known limits

- **Tool execution.** treejit sees the model API, not tool execution. It backtracks its policy, not the world.
- **Rebuild cost.** The tree is rebuilt in full for a family on each outcome (tens of ms at a few hundred runs, capped by `max_runs`). An incremental builder is future work.
- **Stable-prefix learning.** A dynamic block early in the system prompt shrinks the learned prefix to whatever precedes it.
- **Compaction and the prompt cache.** The keep-last window moves as the conversation grows. The step that leaves it changes from full to compacted once, which invalidates the cache from that message on. A chunked boundary (advancing the window only every few steps) would trade a little compaction for longer cache hits; it isn't implemented.
- **Run identity.** Without an `X-TreeJIT-Run` header, run ids are derived from the task and first tool-call id. Resuming the same task text in a new conversation starts a new run.

## Development

```bash
pip install -e '.[dev]' -e bench
pytest -q                                   # 46 tests, ~2 s
python -m treejit_bench --tasks 200 --out bench_out [--via-proxy] [--seed N] [--family coding|retail|mixed] \
    [--modes baseline,treejit,treejit+ok,treejit+ok+compact]
```
