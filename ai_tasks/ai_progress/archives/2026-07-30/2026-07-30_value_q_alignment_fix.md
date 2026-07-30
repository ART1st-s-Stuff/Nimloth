# SFT2/RL ValueHead standard Q alignment fix

## Task

Correct the temporal alignment bug that trained and evaluated an executed action
against the successor it produced.  ValueHead must represent the standard
outgoing quantity `Q(s_t, a_t)`.

## Implemented contract

- For a T-step SFT2 rollout, WM/DINO supervise
  `[hat{s}_{t+1}, ..., hat{s}_{t+T}]`.
- Value loss supervises `[s_t, hat{s}_{t+1}, ..., hat{s}_{t+T-1}]` paired with
  `[a_t, ..., a_{t+T-1}]` and the matching MC returns.
- A K-action planning path is scored on its final edge as
  `Q(hat{s}_{t+K-1}, a_{t+K-1})`; only the earliest/root action is executed.
- SFT2 and planning-RL objective identifiers were versioned, and loaders/resume
  paths fail closed on the old incoming-action checkpoint semantics.

## Modified areas

- Agent/WorldModel forward and multi-step rollout outputs.
- SFT2 and RL value-loss call sites.
- Greedy, exhaustive, beam, and MCTS final-edge scoring.
- SFT2 evaluation and RL planning checkpoint semantic guards.
- Regression tests and module documentation.

## Verification

- Focused regression suite: `51 passed`.
- Expanded WM/Agent/SFT2/RL suite: `283 passed, 1 skipped`; its three Gloo
  cases were rerun outside the network-restricted sandbox and passed
  (`4 passed, 1 skipped`).
- One unrelated VAGEN schema test remains unavailable in the selected Nix Python
  because the external VAGEN package imports optional dependency `gym`; the other
  30 tests in that focused rerun passed.
- `python -m compileall -q src tests` and `git diff --check` passed.

## Consequence

The completed epoch-1/epoch-2 SFT2 checkpoints with objective
`predicted_rollout_executed_action_mc_v2` have incompatible ValueHead semantics.
They are rejected by corrected rollout/RL loaders and must not be used to measure
the fixed planner; SFT2 must be retrained for a corrected evaluation.
