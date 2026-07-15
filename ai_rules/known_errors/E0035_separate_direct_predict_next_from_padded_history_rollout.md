# E0035: Separate direct `predict_next` from padded-history rollout

## Error

A SFT2 `LatentWMPredictor` report labeled autoregressive horizon1 as the same
"one-step" metric used during training.

Training calls the predictor on one `(State, action)` pair. `rollout_states`
uses `history_size=4` and pads the first prediction with three historical State
copies and zero actions. A predictor can therefore have nearly identical
direct validation loss but very different horizon1 rollout behavior.

## Required practice

- Report direct `predict_next` and autoregressive `rollout_states` separately.
- State the history padding used by rollout.
- Apply shuffled-action controls to both paths.
- Do not infer deployment rollout quality from the training one-step loss.
- Preserve incorrectly labeled output when correcting an experiment, then
  write explicit metric-mode metadata in the corrected artifact.

## Observed evidence

In frozen-State dynamics-dimension experiment ID20, direct MSE was
`.167086/.167503` for full8192/factorized2048. Padded-history horizon1 was
`.197503/.174901`. Full horizon1 correct-action MSE was slightly worse than its
shuffled control, while its direct path remained action-sensitive.

The original metrics were preserved as
`dynamics_metrics_pre_direct_mode_fix.json`; corrected metrics record
`direct_predict_next` and `autoregressive_rollout` modes.
