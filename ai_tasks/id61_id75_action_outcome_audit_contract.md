# ID61 — ID75 action-outcome audit contract

## Question

Does ID75 fail on `move_left` because successful lateral moves and blocked/no-op
executions are mixed, the frozen state cannot discriminate them, and the
train/external outcome distribution shifts?

## Authorization and scope

The human explicitly requested validation of this hypothesis. ID61 is frozen,
read-only inference on existing pre-RL artifacts. It performs no optimizer
update and writes no checkpoint. It does not authorize projector/WM retraining,
T2/T4, ValueHead, MCTS or RL.

## Sources

- Exact ID52 pre-RL train/validation archives used by ID60/ID75.
- Hash-pinned canonical ID60 float32 state cache.
- Hash-pinned ID75 retry1 Residual-T1 predictor.
- Per-step execution labels come from the exact archived environment feedback in
  `observation_texts[t+1]`: `Last action is [not] executed successfully.` This is
  direct archived evidence, not trajectory-level success, reward inference, or
  an unchanged-image proxy.

## Required analysis

1. Fail closed unless every action has exactly one authoritative feedback label.
2. Report all-step and early step0–3 success/failure counts per action.
3. Report early train versus decontaminated external failure-rate shift.
4. Run frozen ID75 inference once and report copy-relative metrics separately for
   successful and failed execution outcomes.
5. For `move_left`, bootstrap transition-level copy-minus-prediction MSE for each
   outcome and report actual/predicted state-change magnitude.
6. Use predicted delta magnitude only as a diagnostic success-discrimination
   score and report tie-aware ROC AUC; do not call it a trained success head.
7. Report exact-image-unchanged rates as a secondary proxy only.

## Runtime

- Slurm `normal`, one H800, at most 15 minutes.
- W&B project `nimloth-recon`, run ID
  `nimloth-recon-id61-id75-action-outcome-audit`, resume forbidden.
- Fresh output `outputs/experiments/evaluation/state_alignment/2026-08-24/61_id75_action_outcome_audit`.
