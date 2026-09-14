# Execution Plan

1. Validate both adapter identities, clean source commits, VAGEN Batch API import, free GPUs/ports, disk/memory and unused output roots.
2. Export epoch 5 and epoch 6 sequentially through the formal checkpoint exporter; run the Stage 1 full-HF acceptance loader on each.
3. Prepare two isolated AI2-THOR homes from the verified release cache and start separate BatchEnvironmentServer instances; verify health and empty-batch API.
4. Launch the two unified Stage 1 evaluations in parallel with fixed parameters and recorded PIDs/logs.
5. Monitor GPU/process/log/output health until both finish or fail; do not auto-retry.
6. Use the same unified entry with `--resume --summarize-only` to verify final summaries, then report per-set and overall success rate with evidence limits.


## Current continuation implementation

1. Inspect committed epoch7 metadata and official workflow/continuation interfaces.
2. Implement boundary/EOS weight and objective identity end to end, with compatibility and exact gradient tests.
3. Implement only the necessary explicit objective-change continuation path if not already available; document fresh optimizer/convergence versus preserved epoch/data lineage.
4. Independently review code/tests; main agent updates and shows minimal spec pseudocode diff.
5. Commit task changes, synchronize to isolated remote worktree, run CPU preflight and fresh resource/identity checks, then launch approved training from epoch7 and monitor actual optimizer progress.

## Latest user correction: existing resume only

User explicitly forbids a new entry and requests replacing weight-change rejection with warnings. This supersedes the prior separate-run/explicit-continuation design above. Use original workflow --resume with CLI action16 and boundary16 in the original run, loading epoch7 and preserving optimizer/scheduler/RNG/data cursor AND unweighted-LM convergence state. Only recognized loss-weight changes are warnings; stage/data/model/other hyperparameter mismatches remain errors. No new continuation option, phase offset, optimizer reset, new model initialization, or edited training_state. New checkpoints record actual weight identity; existing epoch checkpoints stay immutable. Workflow records the accepted parameter transition.

## 2026-09-12 sampling format validation correction (user requested)
Training Stage1 format validation currently greedy128 versus production sampling temperature0.7/top_p0.95/max512. Change existing format-validation generation/config plumbing to sampling defaults aligned with production, explicitly disable HF top_k, fixed diagnostic seed0 with training RNG restoration; retain raw outputs and generation settings. Preserve Stage2 defaults, loss and stopping policy. No new entrypoint. Backend equality is not guaranteed by equal seed; matched-checkpoint HF/vLLM sampling comparison remains a proposed subsequent experiment, not completed evidence. User requested analysis, not automatic new training or stopping-rule change.

## 2026-09-12 confirmed weighted convergence continuation
User confirmed proposed rule after choosing resume epoch12 over backend comparison: monitor validation_weighted_lm_loss; patience2 and relative1%, min2 retained; at plateau require sampled format>=31/32 for readiness, otherwise stop with explicit unmet-format status. Existing train.py --resume, keep action/boundary/EOS16, optimizer/scheduler/RNG, original data/cache. Reset only metric-specific convergence and best selection on explicit unweighted->weighted transition. Recompute weighted epoch12 validation baseline with resumed model; no old unweighted CSV fallback. Preserve historical terminal evidence, distinguish current active/terminal state. Stage1 sampling .7/.95/top_k0/512/seed0 already deployed. No hard maxepoch, no automatic eval. User authorizes implementation/check/commit/deploy and continuation, no further confirmation of same rule needed.

## 2026-09-12 validation acceleration
User requests speedup; retain32 samples, sampling.7/.95/top_k0/max512/seed0 and strict outputs. Stage1-only inference temporarily recursively materializes real full parameters and bypasses nested FSDP module edges to avoid token allgather, restoring exact references before reshard. Explicit KVcache and root eval; restore modes/RNG and optimizer identity. No new training/eval entrypoint or approximation. Actual GPUs40GB. Verify CPU graph/exception contracts, then <10min2rank real FSDP unwrapped-forward/train/save mechanics probe (15min total deadline), then resume productionstep250 and time full validation/peak memory. Live workflow replacement caused parent signal-selection error documented in progress; no input/checkpoint edits. Latest step250 replays2 updates through252. Continue same authorized weights/metric; no autoevaluate policy.

## 2026-09-12 success rate alongside format (current user request)
User requires success rate as well as format rate. Existing official evaluation blocks all environment episodes when static format<31/32; change low rate to explicit diagnostic warning and continue actual120 episodes, retaining required compatible raw format evidence and strict malformed-output handling. No output repair/no denominator filtering/no new entrypoint. Existing empty-noop invalidaction behavior and environment success contract retained. Training format readiness31/32 unchanged. Resource decision pending user: a1002 alongside training versus a1001 between training; use latest completedepoch14, existing sampling.7/.95 testBase60+CommonSense60. No remote eval launch until resource answer.
