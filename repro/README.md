# repro/

Reproduction scripts and captured outputs for the issues in [`PLAN.md`](../PLAN.md). File names start with the issue ID.

**Which commit they reproduce.** Unless a line says otherwise, a script demonstrates the issue *as it was on the plan's base commit* `19a53f5`, and its `*_out*` / `*.out` / `*_output.txt` file is the output captured there. On the current code most of them now show the fixed behaviour (or no longer apply). Files named `*after*` were captured after the fix. Prototypes (`*_proto*`, `*_patches.py`, `*_variants.py`, `P1_memo.py`) monkeypatch internals of the base commit and are kept only as evidence for the design choice; they are not expected to run against HEAD.

**Paths.** Many scripts put `/home/user/treejit/src` (or `tests`) on `sys.path`, the checkout they were written in. Run them from the repo root with the package installed (`pip install -e . -e bench`) or with `PYTHONPATH=$PWD/src:$PWD/bench/src:$PWD/tests`. Scripts that need a tau-bench clone look for `repro/E1_tau-bench` (clone `https://github.com/sierra-research/tau-bench` there).

**Subcall detection.** Since W8 (X1), Anthropic T2/T3 subcalls use structured outputs (`output_config.format`) instead of a forced `tool_choice`. The scripts with fake models (`E1_tau_proto.py`, `E2_measure.py`, `E4_first_step_subcall.py`, `C1_bench_billed.py`) detect and answer subcalls with `treejit.subcalls.subcall_tool` / `answer_content`, as the bench does, so they run on the current code.

| ID | Script | What it shows | Output |
|---|---|---|---|
| B1 | `E1_tau_proto.py --split train --n 300` | treejit+ok-only failures on tau-bench retail train (wrong irreversible write replayed). Superseded by `python -m treejit_bench --suite taubench` | `E1_out_train300.txt` (base) |
| S1 | `S1_opaque_executors.py` | 40 opaque executors (interpreters, scripts, task runners, aliases, `gh api`) and whether they are commit points; end-to-end under `approve '*'` | `S1_output.txt` (base) |
| S1 | `S1_bench.py [proto]` | sim bench with the current policy vs the prototype: unchanged | `S1_bench_output.txt` |
| S1–S3 | `S_proto_policy.py`, `S_proto_plugin.py`, `S_proto_eval.py` | prototype of the commit-point redesign, evaluated on the S1/S2/S3 sets | `S_proto_eval_output.txt`, `S_proto_tests_output.txt` |
| S2 | `S2_bash_parity.py` | 23 commands where the tokenizer and real bash disagree (logging `git` stub) | `S2_output.txt` (base) |
| S2 | `S2_ro_fuzz.py` | fuzz of the read-only side: ~42.6k commands, 0 bypasses | `S2_ro_fuzz_output.txt` |
| S3 | `S3_false_positives.py` | 51 realistic commands vs commit-point detection (28 wrong on base) | `S3_output.txt` (base) |
| S4 | `S4_git_config_exec.sh` | "read-only" git reads running repo-configured programs (fsmonitor, diff.external, textconv, filters, embedded bare repo) | `S4_output.txt` |
| T2 | `T2_run_identity.py` | deterministic call ids merge three conversations into one run | `T2_out.txt` (base) |
| M1 | (unit tests in `tests/test_episodes.py`) | `task='yes'` episodes after a confirmation | E1 notes |
| L1 | `L1_repro.py` | a chance task-word rule replays once into an unseen input class (A = misroute, B = guard) | `L1_repro.out` (base) |
| L1 | `L1_state_before.py` | the seed-3 decision list right before task 13 | `L1_state_before.out` |
| L1 | `L1_variants.py` | prototype fixes measured on the bench | `L1_variants_repro.out`, `L1_bench_summary.out` |
| L2 | `L2_repro.py` | negatives pooled per rule: failures needed to demote vs N passing replays | `L2_repro.out` (base) |
| L2 | `L2_patches.py`, `L2_bench.py`, `L2_compare.py` | kNN prototype and its bench comparison | `L2_compare.out`, `L2_compare_knn0.out` |
| P1 | `P1_gen.py`, `P1_json_gen.py` | generate coding / JSON-heavy trace logs and time the synchronous rebuild | `P1_gen_*.out`, `P1_json_gen300.out` |
| P1 | `P1_profile.py`, `P1_cumprofile.py` | cProfile of `build_family` | `P1_profile.out`, `P1_cumprofile_2000.out` |
| P1 | `P1_memo.py` | estimated win from memoizing per-step data (prototype) | `P1_memo.out` |
| P1 | `P1_latency.py` | `/outcome` blocking the proxy's event loop | `P1_latency.out` (base), `P1_latency_after.out` |
| P1 | `P1_stale_view.py` | a running proxy never reloads a tree rebuilt by another process | `P1_stale_view.out` (base), `P1_stale_view_after.out` |
| P1 | `P1_timing.py` | `build_family` timings as a running process sees them | `P1_timing.out` |
| C1 | `C1_cache.py` | moving-window compaction vs the prompt cache (+123% billed on dense traffic) | `C1_out_small.txt`, `C1_out_bigsys.txt` (base) |
| C1, H1 | `C1_after.py` | the same patterns after first-sight compaction and sticky hints | `C1_after_small.txt`, `C1_after_bigsys.txt` (regenerated at `6e1177f`, with sticky hints) |
| C1 | `C1_bench_billed.py --seed N [--bigsys]` | sim bench billed with the cache model: off / first_sight / window | `C1_bench_billed_out.txt` (regenerated at `6e1177f`; seeds 0–5, without and with `--bigsys`) |
| C2 | `C2_nopath.py`, `C2_compare.py` | dropping compaction rule 3b changes no trajectory on seeds 0–5 | `C2_nopath_s*.txt`, `C2_bench_s*.txt`, `C2_compare_out.txt` |
| C3 | `C3_growth.py` | the `compactions` table grows forever; `prune` doesn't touch it | `C3_out.txt` (base) |
| K1 | `K1_families.py` | early `<env>` block splits families; a shared preamble merges agents (includes the prototype) | `K1_out.txt` (base) |
| T1 | `T1_inline_stream.py` | inline streaming bypasses treejit and leaks `X-TreeJIT-Run` | `T1_out.txt` (base) |
| T1 | `T1_wrapped_surface.py` | the wrapped client's `.messages` loses `stream`/`count_tokens` | `T1b_out.txt` (base) |
| E1 | `E1_tau_proto.py`, `E1_patch.py` | the tau-bench prototype (oracle-with-noise agent, tau-bench's own reward) | `E1_out_test5.txt`, `E1_out_test115.txt`, `E1_out_train300.txt` (base) |
| E2 | `E2_measure.py` | prompt composition of full vs small calls in the sim; re-priced under harness payloads and caching | `E2_out_seed0.txt` (base), `E2_out_seed0_after.txt` (`6e1177f`) |
| E4 | `E4_explain_proxy.py` | `explain` mislabels steps on header-less proxy traffic | `E4_out_s0.txt` (base) |
| E4 | `E4_first_step_subcall.py` | a failed first-step subcall kept `run_id NULL` | `E4b_out.txt` (base), `E4b_after_out.txt` (HEAD: every subcall row has a run id, explain total = table total) |
| E5 | `E5_macro_headroom.py` | headroom for macros-as-tools (≤0.08 replayable steps after a full call): no-go | `E5_out.txt` |
| R1 | `R1_probe.py`, `R1_responses_design.md` | `/v1/responses` passed through unlearned; argv shell args never read-only; the Responses design note | `R1_out.txt` (base) |
| X2 | `X2_live_check.py [--offline]` | preserved thinking vs replayed turns and sticky hints, against a real model; prints SKIP without `ANTHROPIC_API_KEY` | none yet (no key) |
