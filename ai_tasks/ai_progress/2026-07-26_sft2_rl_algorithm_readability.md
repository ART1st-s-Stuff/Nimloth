# SFT2/RL algorithm readability and shared value objective

## Goal

- Make SFT2 variable and objective names state their actual semantics.
- Remove the generic SFT2 auxiliary-loss layer whose only implementation is DINO-grid.
- Make RL algorithm control flow readable without deleting real artifact/runtime checks.
- Share the executed-action Monte Carlo ValueHead objective between SFT2 and RL.

## Confirmed semantics

- SFT2 already receives `action_value_target` values computed by
  `discounted_action_value_targets()` over the complete episode before sampling history
  contexts.
- Its old private `_value_loss()` already selected the executed action and applied MSE to
  that Monte Carlo return. Adding another MSE would double-count the same objective.
- The optional ranking term is separate from the Monte Carlo regression and remains
  configurable.

## Implemented

- Added `training/common/value.py` with `ActionValueLoss` and `action_value_loss()`.
  SFT2, RL complete-episode MC, RL direct-policy training and the SFT2 trajectory
  equivalence diagnosis now call it.
- Replaced SFT2 `auxiliary_targets`, `SFT2AuxiliaryLoss` and `DINOGridLoss` with the
  explicit `dino_grid_target`, `dino_grid_weight` and `dino_grid_mse()` flow.
- Renamed `aux_device` to `world_model_device` and `load_aux_checkpoint()` to
  `load_world_model_checkpoint()`.
- Renamed the SFT2 metric `value_reg` to `value_mc_mse`.
- Split RL policy replay and reference KL work into named helpers. The main training step
  now directly shows state projection, WM/MC value, SIGReg, policy and total-loss order.
- Removed checks duplicated by `RLConfig`, `EpisodeTrainingBatch`,
  `TemporalDifferenceStep` and `PolicyReplayInput`. Checks protecting cached-state
  provenance, replay capabilities, reference-token alignment and PPO tensor alignment
  remain at those boundaries.

## Commits

- `94d6b015`: shared objective and main SFT2/RL refactor.
- `05376c1e38d8c1ef1be8a926801f36baf7487d09`: follow-up naming cleanup.

Both commits are pushed to `origin/dev`. The pre-existing `external/le-wm` worktree state
was not staged or modified.

## Validation

- Passed: Python `compileall` for affected source and tests.
- Passed: `git diff --check`.
- Passed: static call-site scans for removed auxiliary names and old checkpoint/device
  names.
- Added tests for executed-action-only MC MSE, ranking behavior and ValueHead/input-state
  gradients.
- Pending: pytest execution. The local Python environments contain neither torch nor
  pytest. Two superpod SSH attempts were immediately closed by the proxy with
  `Connection closed by UNKNOWN port 65535`; this was not a command/test failure.

## Next step

When SSH connectivity is restored, update the existing server worktree to exact commit
`05376c1e38d8c1ef1be8a926801f36baf7487d09` and run the focused common/SFT2/RL tests,
then the expanded SFT2/RL/WM regression. Do not claim completion before those tests pass.

## 2026-07-27 correction to the projector conclusion

- Human clarified that SFT1 DINO supervision of `SharedSlotProjector` is acceptable. The
  previous conclusion that the projector must belong only to SFT2 was wrong.
- The actual defect is the SFT2 double path: freezing the SFT1 projector, adding another
  online grid encoder, adding a WM EMA target for that encoder, and decoding the resulting
  state back to DINO space.
- The corrected contract reuses the SFT1 projector as the trainable SFT2 StateProjector;
  its output is the WM state and predicted state receives direct DINO MSE. WM uses no EMA.
- ID33/ID45/ID46 and derived RL checkpoints are retained historical artifacts but are
  invalid as evidence for the corrected state semantics. No compatibility conversion is
  provided.

### Implemented and validated

- Removed the second grid encoder, WM EMA target encoder, DINO decoder, checkpoint extras,
  retired config fields, and the ID33 warm-start path.
- SFT2 loads the SFT1 `SharedSlotProjector` in FP32 and trains it directly. Its output is
  the grid state; predicted next state receives direct cached-DINO MSE. The target
  Backbone remains frozen/vision-EMA-controlled while the shared target-side projector
  keeps the standard SFT2 gradient path.
- RL training and planning loaders reconstruct the direct projector from the new
  `state_proj.pt`; a focused test proves the retired nested state dict is rejected.
- Focused CPU tests passed: `21 passed in 3.51s`. Affected-source `compileall`, three
  launcher `bash -n` checks, and `git diff --check` passed. No GPU, distributed, or full
  repository test was run.

## Follow-up: explicit data migration and source contracts

- Added `nimloth.rollout.migration` and `nimloth_trajectory_v1`. Historical JSONL is
  converted offline with declared action-space/reward/planner semantics and a SHA256
  manifest; current training readers no longer parse legacy field aliases or defaults.
- Made RL state ownership explicit through `gradient.state_source`. Planner training reads
  rollout-saved Qwen hidden states; non-planner recompute mode runs Qwen. Missing data does
  not trigger an automatic source switch.
- Removed action-only Qwen replay fallback and duplicate completed `messages` storage.
- Restricted SFT2 training to compact cache v2. Historical v1 caches must be rebuilt from
  migrated JSONL because they lack the current terminal next-state encoding.
- Removed the SFT2 legacy cache builder/collator/CLI selector and the deprecated SFT2 latent
  query mask CLI alias.

### Follow-up validation

- Affected focused tests: `106 passed`.
- Nimloth tests that can be collected in this local environment: `384 passed, 1 skipped`
  across the main suite, loopback Gloo tests, and outer-runner fault injection.
- `compileall`, migration CLI help, SFT2 pipeline `bash -n`, and `git diff --check` passed.
- Full repository collection remains unavailable because external VAGEN/VERL and the local
  optional vLLM/PEFT/pandas tests lack dependencies. No GPU experiment was run.
- Changes remain uncommitted as requested; the unrelated `external/le-wm` state was untouched.

## Follow-up: direct RL algorithm API, module-level DDP, and Chinese comments

- Added detailed Chinese documentation around retained history, autoregressive segment
  replay, endpoint target detachment, Qwen action distillation, and per-update TD loss
  normalization.
- Removed `RLTrainingStepModule`, `RLTrainingSteps`, and all private request objects. The
  loop directly calls `RLAlgorithm.temporal_difference_step()`, `monte_carlo_step()`, and
  `sequence_step()` with the shared `RLModelRuntime`.
- Moved synchronization to the actual parameter owners: multi-device Qwen uses official
  DDP with `device_ids=None`; StateProjector, WM predictor, ValueHead, and optional
  TokenValueHead use their existing single-device DDP boundaries. A two-rank Gloo probe
  verified multiple forwards plus a no-grad target forward before one backward and matched
  gradients across ranks.
- Renamed the non-planner algorithm entry from generic `training_step()` to
  `sequence_step()` without a compatibility alias. Updated the advertised synthetic smoke
  script to the current `RLModelRuntime` and `RLBatch` contract.
- Translated the remaining English-only docstrings and explanatory comments under
  `src/nimloth/training/rl` to Chinese.
- Validation: complete `tests/training/rl` passed (`122 passed, 1 warning`), including a
  two-rank Gloo TD-to-MC optimizer-step parameter consistency test. No GPU run, commit, or
  push was performed.
