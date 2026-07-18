# E0047 — WM rollout context must match training

## Error

A `LatentWMPredictor` trained only through `predict_next_emb()` received one state/action token (`T=1`), but evaluation called `rollout_states()` with `history_size=4`. That path supplied four tokens, used positional embeddings 1–3 that were never trained, and padded missing history with repeated states plus action index 0 (`move_forward`) without a padding mask.

## Consequence

The resulting predicted states are out of the trained inference protocol. Their reconstruction quality cannot be used to judge the learned one-step WM or autoregressive dynamics. Immediate horizon-1 degradation and non-monotonic horizon metrics can be artifacts of this mismatch.

## Rule

Before evaluating or deploying WM rollouts, verify the exact training-time context length, positional embeddings, padding semantics, and action alignment. If training used `T=1`, free-run with repeated calls to `predict_next_emb()` unless the model is retrained with real multi-step contexts. Never encode missing history as a valid environment action without an explicit trained padding/mask protocol.

## Required diagnostic

Report separately:

1. teacher-forced one-step prediction from each actual state;
2. training-consistent recursive free-run prediction;
3. multi-history prediction only if that history protocol was trained.
