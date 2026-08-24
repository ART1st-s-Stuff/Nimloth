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
- Attempt0 Job `529539` completed the calculation but violated E0145 by initializing W&B under `flower`; it is diagnostic evidence only, not the formal result.
- Formal retry1 W&B project `nimloth-recon`, run ID `nimloth-recon-id61-id75-action-outcome-audit-retry1`, resume forbidden.
- Fresh formal output `outputs/experiments/evaluation/state_alignment/2026-08-24/61_id75_action_outcome_audit_retry1`.

## Actual result

Formal retry1 Job `529546` completed `0:0` in `00:00:37` on one H800 (`normal/dgx-10`); W&B `nimloth-recon` finished.

- Early `move_left` failure rate shifted from train `386/1702=22.68%` to decontaminated external `75/193=38.86%` (`+16.18pp`). Failures were not the majority.
- Successful subset: copy/prediction RMSE `0.25315/0.23156`, skill `+0.16330`, copy-minus-prediction MSE bootstrap 95% CI `[+0.00175,+0.01941]`.
- Failed subset: copy/prediction RMSE `0.05526/0.18255`, skill `-9.91418`, bootstrap 95% CI `[-0.04258,-0.01959]`.
- Failed external images were `100%` exact unchanged; successful images were `0%` unchanged. Actual state-change success AUC was `0.99977`, while predicted-change success AUC was only `0.61650`.
- Conclusion: the core mixture/outcome-prediction hypothesis is supported, but the claim that most training `move_left` actions fail is false. ID75 improves successful lateral moves and hallucinates movement on blocked/no-op transitions; weak outcome discrimination plus train/external outcome shift makes aggregate action3 skill negative.
- Result SHA256: `bace6fcbc5ec85fdeed59e6ba30ff61b58bbe382f88af51f9dd591a8105a28e4`.
