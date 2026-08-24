# ID71 — Frozen-state action-outcome predictability probe

## Authorization

The human approved the proposed low-capacity outcome probe. ID71 trains only
diagnostic readouts on existing frozen pre-RL caches. It does not authorize
projector/WM calibration, T2/T4, ValueHead, MCTS or RL.

## Question

Does the current ID176 + SFT1-projector K16 state contain linearly accessible
information that predicts whether movement actions will execute successfully,
or did the state interface discard collision-relevant geometry?

## Data and labels

- Exact ID52 pre-RL early step0–3 transitions and immutable ID60 float32 cache.
- Labels are exact archived environment feedback after each action, not
  trajectory success, reward inference, or image-change proxy.
- Movement actions with both outcomes: move_forward, move_right, move_left.
- External validation uses ID60 exact initial/current/next-image decontamination.
- Inner selection keeps each exact initial-image group wholly in fit or selection.
- Row-level source task identity remains unavailable; no formal task-generalization claim.

## Matched probes

For each movement action independently:

- flatten full K16 `[16,1024]` state or matched DINO grid;
- standardize each dimension using fit rows only;
- train one binary linear readout with identical parameter count and optimizer;
- select epoch using only inner-selection ROC-AUC;
- retrain fresh on all archive train rows for the selected epoch count;
- evaluate the external split once.

Report ROC-AUC, PR-AUC, balanced accuracy, NLL, Brier, ECE, constant train-rate
baseline, paired state-minus-DINO bootstrap interval, and ID75 predicted-change
AUC. Diagnostic weights cannot be used downstream.

## Runtime and identity

- Slurm `normal`, one H800, at most 10 minutes.
- W&B project `nimloth-recon`, run ID `nimloth-recon-id71-action-outcome-probe`, resume forbidden.
- Fresh output `outputs/experiments/evaluation/state_alignment/2026-08-24/71_frozen_state_action_outcome_probe`.
- Shared `.env` is sourced before locked `RUN_WANDB_*` values are exported; the entrypoint fails closed on effective/initialized identity mismatch.
