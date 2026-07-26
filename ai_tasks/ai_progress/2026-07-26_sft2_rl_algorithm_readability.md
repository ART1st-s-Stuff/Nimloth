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
