# ID58 SFT1 / ID74 checkpoint state matrix contract

Status: human authorized the read-only stage-0 diagnostic on `normal 1×H800`, with total execution capped below two hours. No training is authorized. Attempt0 Job`528778` was cancelled with human confirmation after a W&B project override; retry1 is the only valid result candidate.

## Purpose

Before any diagnostic retraining, isolate:

1. SFT1-to-ID74 backbone/vision drift;
2. SFT1-to-ID74 projector drift;
3. ID74 online-versus-vision-EMA installation drift;
4. compatibility of each resulting state space with the frozen ID74 one-step WM and ValueHead.

## Data and split

- Source: ID74's migrated pre-RL validation JSONL:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data/val_terminal_cot_migrated.jsonl`.
- Split semantics were verified from the original dataset/loader records: seeds 1081--1200 are task-disjoint validation rows from the train-scene datasets; this is not the heldout test scene split.
- Deterministic selection: 32 trajectories each from Base/Common/Long Horizon; candidate transitions are limited to steps 0--3 and consumed round-robin by executed action after stable record-ID hash ordering; at most one transition is selected per trajectory and every selected transition has an exact following decision state/action.
- Initial preflight found all 355 validation trajectories execute action0 at step0. Restricting the matrix to step0 would therefore make the WM/Q audit single-action. The final contract uses steps 0--3 to preserve bounded context while adding real action diversity.
- Current and next states use exact archived observations and actual corresponding CoT; no state or CoT replay is allowed.
- Original-observation frozen DINO target comes from the exact pre-RL cached DINO grid used by SFT2. No decoder resize or ID189 data is used.

## Checkpoint matrix

Backbone/vision modes:

- SFT1 merged backbone and vision;
- ID74 online backbone and vision;
- ID74 backbone with the archived ID74 vision EMA installed.

Each is crossed with:

- SFT1 `SharedSlotProjector`;
- ID74 `state_proj.pt`.

This yields six `[16,1024]` state combinations. Frozen ID74 `TemporalSpatialGridPredictor` and `ValueHead` are evaluated on every cell. On non-ID74 cells, those results mean cross-component compatibility only; they do not represent an SFT1-trained WM or Q head.

## Metrics

- actual current/current-DINO and actual next/next-DINO RMSE, cosine and token-centered cosine;
- state mean/std/RMS/slot-deviation RMS;
- current-to-next copy RMSE;
- frozen ID74 one-step predicted-to-actual RMSE and copy-relative skill;
- predicted-next/next-DINO metrics and copy-relative skill;
- frozen ID74 executed-action ValueHead return RMSE at current, actual next and predicted next;
- selected pairwise state RMSE isolating backbone, projector and EMA changes;
- overall, by-source and by-first-action summaries.

No goal probe is run because this archive has no validated goal labels or real matched same-observation/different-goal pairs. The evaluator must state this missing capability rather than creating heuristic labels.

## Freeze, output, and resume boundaries

- All backbone, vision, projector, DINO, WM and ValueHead parameters are frozen under inference mode.
- Forbidden: optimizer construction, backward, generation/replay of CoT, parameter updates, new model checkpoint, and resume.
- Attempt0 output (failed, immutable):
  `/project/peilab/atst/nimloth/outputs/experiments/evaluation/state_alignment/2026-08-23/58_sft1_id74_checkpoint_state_matrix_val96_early4`.
- Retry1 formal output is fresh and unique:
  `/project/peilab/atst/nimloth/outputs/experiments/evaluation/state_alignment/2026-08-23/58_sft1_id74_checkpoint_state_matrix_val96_early4_retry1`.
- Runner creates `RUN_OUT` exactly once and fails closed if it already exists.
- Outputs: `README.md`, `result.json`, `summary.html`, float32 `matrix_states.npz`, hashes and W&B metadata.
- Retry after any failure requires a fresh output directory and fresh W&B identity; failed output is never overwritten.

## Runtime identity and resources

- Entrypoint: `python -m nimloth.eval.sft_checkpoint_state_matrix` through `experiments/training/sft2/run_id58_sft_checkpoint_state_matrix.sh`.
- Python: `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- W&B project: `nimloth-recon`.
- Attempt0 incorrectly initialized in project `flower` because the shared env overwrote `WANDB_PROJECT`; Job`528778` was cancelled after `00:03:13`, before metrics, and is invalid.
- Retry1 W&B project/name/ID: `nimloth-recon` / `58_sft1id74_state_matrix_val96_early4_k16_retry1` / `nimloth-recon-id58-sft1id74-state-matrix-retry1`. Locked `RUN_WANDB_*` values are reasserted after sourcing credentials.
- Slurm: `normal`, one node, one GPU (H800), 16 CPUs, 96 GiB RAM, hard walltime `01:45:00`; excludes `dgx-09,dgx-13,dgx-32,dgx-51`.
- Source commit: recorded in the run-owned README from `EXPECTED_COMMIT`; production worktree must be clean and exactly match it.
