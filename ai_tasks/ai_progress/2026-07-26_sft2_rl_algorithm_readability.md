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
  state branch freezes both the Backbone and shared projector; the projector trains when
  that state appears as the current state.
- RL training and planning loaders reconstruct the direct projector from the new
  `state_proj.pt`; there is no old-format compatibility or dedicated rejection branch,
  so ordinary key/shape errors surface from strict loading.
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

## 2026-07-27 follow-up: RL restores the shared DINO WM objective

- Human corrected the stale conclusion that RL should omit DINO loss. Grid-WM training in
  SFT2 and RL now calls `training/common/world_model.py` for the same weighted state MSE and
  predicted-state DINO-grid MSE.
- SFT2 keeps its validated offline DINO cache. RL aligns each prediction with the real next
  observation image from the trajectory and uses the pinned frozen DINOv2-large teacher;
  computed targets are cached by image path.
- Greedy H=2 grid configs explicitly set `lambda_wm=1.0` and `lambda_dino=0.5`. Planner TD
  target paths use Qwen anchor endpoints; non-planner sequence targets use every next image
  in batch/time order.
- Focused CPU tests passed: `56 passed in 6.94s`, including a two-rank loopback Gloo unit
  test. No GPU or full-scale training was run. These follow-up changes are not committed yet.

### Fixed WM target correction

- Removed `WorldModel.project_target_state()`, which had become a misleading alias after
  removal of the WM EMA target encoder. There is no target projector or target WM.
- The target-side shared-projector gradient was an older SFT2 behavior, not introduced by
  this follow-up. It was incorrect because the WM supervision value could move toward the
  current prediction even though the same state trains the projector when used as current.
- SFT2 and both RL WM entry points now keep the complete next-state supervision value fixed.
  Planner TD reads the actual endpoint anchor state saved by rollout instead of reprojecting
  endpoint Qwen hidden.
- Isolated tests set the predictor Jacobian to current state to zero and verify that the
  fixed supervision branch produces no StateProjector gradient.
- Focused CPU regression passed: `78 passed in 6.71s`, including the two-rank loopback
  Gloo/DDP update test. No GPU or full-scale training was run. Changes remain uncommitted.

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
- Documented `RLTrainingLoop._run_iteration` in place: the comments now separate rollout
  collection, planner/non-planner training-unit construction, single-update gradient
  accumulation, fresh-rollout recovery, and post-update checkpoint commit.
- Validation: complete `tests/training/rl` passed (`122 passed, 1 warning`), including a
  two-rank Gloo TD-to-MC optimizer-step parameter consistency test. No GPU run, commit, or
  push was performed.
- The loop-only follow-up passed `tests/training/rl/test_loop.py` (`6 passed`), Python
  syntax compilation, and `git diff --check`; it did not change the loop control flow.

## Follow-up: explicit DINO target assembly

- Removed image-path and frozen-DINO I/O from `RLAlgorithm`. Planner training now loads
  all endpoint targets once before fresh-rollout consumption, keeps the batch on CPU, and
  aligns one target to each flattened TD step by order. Sequence training puts its full
  `(B,H,...)` target tensor into `RLBatch` before calling the algorithm.
- `TemporalDifferenceStep.end_image_path` names the exact planner supervision image; the
  old all-anchor load, `anchors.index(end_step) - 1`, and per-TD whole-batch transfers are
  gone. Different episodes may share the same numerical end step without key collisions.
- SFT2 was inspected rather than mechanically rewritten: its assembler loads exactly the
  current batch's `B` next images and all `B` targets enter the same loss. A recording
  provider test now guards this contract.
- Focused CPU regression passed (`39 passed in 15.13s`), including the two-process Gloo
  TD-to-MC DDP update test. Affected `compileall` and `git diff --check` passed; no GPU test,
  commit, or push was performed.
