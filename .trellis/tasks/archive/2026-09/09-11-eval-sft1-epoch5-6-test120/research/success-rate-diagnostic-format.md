# Research: Success rate alongside diagnostic format validation

- Query: Minimal existing-entrypoint change permitting actual success evaluation when sampled format rate is low, preserving invalid-output behavior.
- Scope: internal; checkout `.worktree/fix-sft1-termination-retrain`; no remote actions.
- Date: 2026-09-12

## Findings

### Files and source patterns
- `src/nimloth/training/sft/evaluation/early.py:65-99`: two blocking branches. Completed rollout currently raises if saved static format diagnostic is below threshold (lines 74-77). New rollout returns 2 at lines 91-96 on below-threshold diagnostic. Change both to explicit warnings; retain actual gate generation, evidence validation and corruption errors. No new CLI needed. Reuse returned generator for environment episodes.
- `src/nimloth/training/sft/evaluation/format_gate.py:284-300`: 32 sample numerator, denominator and passed threshold stored independently. Keep threshold outcome truthful even when it ceases to block environment measurement. Existing output schema/function can remain; annotate orchestration contract that format policy is diagnostic so new semantics are traceable. `allow_generate=False` on completed replay should still require original evidence, not manufacture it.
- `src/nimloth/environment/navigation/early_evaluation.py:57-78`: Stage1 uses raw-token termination and strict parser; failures set `action_index=None`, `service_response=''`, reason preserved. No repair, forced token or semantic fallback.
- Same file lines 99-127: every invalid generation is sent as empty service string, original model text remains assistant history, turn JSON includes parser, raw generation, service text, reward and actual environment info. Invalid turn consumes one of outer `range(config.max_steps)` turns. It does NOT immediately fail the entire episode; subsequent valid actions can still succeed. Success accumulates OR over explicit `metrics.traj_metrics.success` booleans (lines 49-53, 115).
- Same file lines 128-147: terminal extra generation is saved with `executed=False`; do not include this in executed-action format denominator. Completed record saved only after terminal generation, so crash there leaves episode incomplete and resume reruns it.
- `src/nimloth/rollout/early_records.py:15-36`: summary counts ALL completed episode records, including episodes containing invalid turns. Rate is successes/completed, never successes/format-passing subset. When complete Base60 + CommonSense60, denominator is120; partial output explicitly requested/completed/complete. Empty rate is null. Standard identity check requires seeds1..60 each set.
- `external/VAGEN/vagen/envs/navigation/navigation_env.py:245-280`: locally available source executes only actions with parser format correct; empty string causes no physical action. `_step_count` only increments for executed actions, while caller's outer turn cap still terminates after20 generations. Environment success set during valid execution, not fabricated from parsing. This local submodule source may differ from remote pinned844378c; remotely verify fixed source before claiming live behavior beyond client's no-op contract.
- `src/nimloth/training/sft/evaluation/README.md:19-27,41-51`: old blocking31/32 contract needs update with diagnostic semantics. No-op contract is already documented. README also still says weight8 at line26; current checkpoint code already accepts other valid weights, so that sentence is stale.
- `tests/training/sft/evaluation/test_early_flow.py:10-72`: existing parameterized test expects gate failure to suppress episodes; update expectation to warnings + episodes both cases. Add completed-replay below-threshold test. Keep corrupt diagnostic rejection and summarize-only no allocation tests.

### Minimal recommendation and tradeoffs
Continue measuring static32 format rate using same unconstrained sampler, but treat below31/32 as diagnostic warning. Run all120 episodes through existing strict/no-op policy. Report static sample format rate separately from environment executed-turn format rate, and success rate across all completed episodes; do not filter invalid episodes. This measures deployable policy robustness while exposing format defects. It does not claim model meets prior format acceptance criterion. Keep training's FORMAT_UNMET stopping logic unchanged unless user explicitly changes it.

No-op invalid turns are already the implemented execution policy; switching to immediate episode failure, extracting an otherwise-valid action from malformed text, retrying until valid, or truncating at action_end would change the metric and is not the minimal authorized path.

Add explicit `format_validation_policy=diagnostic` metadata or equivalent to distinguish runs. New output directory avoids immutable-contract collisions with prior blocked runs; do not silently relabel historical runs as valid.

## Related specs
- `.trellis/workflow.md`: durable task research, source-backed implementation/check workflow.
- `.trellis/spec/experiments/outputs-checkpoints-and-evidence.md`: metric unit/split/checkpoint/sample count, partial-aware evidence, checkpoint/input identity and output ownership.
- Module README strict raw-response/termination policy above; user request for actual success measurement supersedes old gate-blocking orchestration, not strict action parser.

## External references
No internet needed; pinned VAGEN source is repository-local reference. Remote service implementation must be matched to documented844378c before launch.

## Caveats / Not Found
No GPU/remote/resource validation done. No speed claim or runtime estimate established. This research concerns correctness of actual success denominator and minimal gate orchestration only. Parent has pending a1001-vs-a1002 scheduling decision.
