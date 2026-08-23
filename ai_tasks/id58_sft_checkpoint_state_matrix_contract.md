# ID58 SFT1 / ID74 checkpoint state matrix contract

Status: complete. Retry2 Job`528812` is the only valid result; no training was authorized or performed. Attempt0 Job`528778` was cancelled after a W&B project override; retry1 Job`528804` failed on invalid EMA constructor decay before metrics.

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
- Retry1 output (failed, immutable):
  `/project/peilab/atst/nimloth/outputs/experiments/evaluation/state_alignment/2026-08-23/58_sft1_id74_checkpoint_state_matrix_val96_early4_retry1`.
- Retry2 formal output is fresh and unique:
  `/project/peilab/atst/nimloth/outputs/experiments/evaluation/state_alignment/2026-08-23/58_sft1_id74_checkpoint_state_matrix_val96_early4_retry2`.
- Runner creates `RUN_OUT` exactly once and fails closed if it already exists.
- Outputs: `README.md`, `result.json`, `summary.html`, float32 `matrix_states.npz`, hashes and W&B metadata.
- Retry after any failure requires a fresh output directory and fresh W&B identity; failed output is never overwritten.

## Runtime identity and resources

- Entrypoint: `python -m nimloth.eval.sft_checkpoint_state_matrix` through `experiments/training/sft2/run_id58_sft_checkpoint_state_matrix.sh`.
- Python: `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- W&B project: `nimloth-recon`.
- Attempt0 incorrectly initialized in project `flower` because the shared env overwrote `WANDB_PROJECT`; Job`528778` was cancelled after `00:03:13`, before metrics, and is invalid.
- Retry1 correctly used `nimloth-recon`, but Job`528804` failed after `00:02:43` because `decay=0.0` was rejected before the EMA checkpoint load; it produced no matrix metrics or audit payload.
- Retry2 W&B project/name/ID: `nimloth-recon` / `58_sft1id74_state_matrix_val96_early4_k16_retry2` / `nimloth-recon-id58-sft1id74-state-matrix-retry2`. Locked `RUN_WANDB_*` values are reasserted after sourcing credentials; EMA construction uses valid checkpoint decay `0.999`.
- Slurm: `normal`, one node, one GPU (H800), 16 CPUs, 96 GiB RAM, hard walltime `01:45:00`; excludes `dgx-09,dgx-13,dgx-32,dgx-51`.
- Source commit: recorded in the run-owned README from `EXPECTED_COMMIT`; production worktree must be clean and exactly match it.

## Result

- Job`528812`: `COMPLETED 0:0`, normal/dgx-27, elapsed`00:02:56`.
- W&B: `nimloth-recon/58_sft1id74_state_matrix_val96_early4_k16_retry2`, finished, one history row, status`passed`.
- Selection: 96 unique trajectories; sources32/32/32; action0/2/3/4/5=`25/23/23/22/3`; step0/1/2/3=`25/31/19/21`.
- SFT1→ID74-online backbone drift with fixed SFT1 projector: RMSE`0.020806`.
- Projector drift: RMSE`0.892000--0.892725`.
- Current-DINO RMSE/cosine: SFT1 projector`0.8372--0.8376/0.6560--0.6565`; ID74 projector`1.1363--1.1371/0.3810--0.3821`.
- Canonical ID74 behavior copy/predicted RMSE=`0.703430/0.640073`, aggregate skill`+0.172027`; action0/2/3/4/5 skill=`-0.410761/-0.715589/-0.268277/+0.314698/+0.122861`.
- Canonical actual-next/predicted-next DINO RMSE=`1.128445/0.962218`; state std=`0.984237/0.866125`.
- Canonical ValueHead executed-action return RMSE on actual-next/predicted-next=`0.460140/0.472018`.
- ID74 online and vision-EMA projected states are identical on this sample. Direct weight audit: 282/390 BF16 vision tensors differ, parameter RMSE`1.3149e-6`, max abs`2.1267e-4`; the change is below the observed BF16 state-output resolution.
- Decision: visual degradation is dominated by SFT2 projector drift. SFT1 projector is a visual anchor, but goal health remains unproven. Do not retrain SFT1 yet; prioritize separately authorized SFT2 projector/interface repair and per-action-gated frozen-projector T1 residual-WM canary.
- Result SHA256: `result.json`=`de61930358e92a686a93a738732571af2db8416905349875acfa90af3d2b547b`; `matrix_states.npz`=`c3067f21668089d2953bb5acc16ea1c5f6bacfe1cfa2bee28e4ff6c4aaac6ad1`; `summary.html`=`912e9713f8b3ac284c079ac1feb5ebe7b8037a575118bb8b1ae546e4cb0eefb2`.
