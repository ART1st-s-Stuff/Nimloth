# ID188 pre-RL step0 rollout browser comparison contract

Date: 2026-08-21
Status: full Base60+Common60 Job 530662 submitted on normal 4x2 and pending Priority

## Purpose

Run the same semantic Base test seed2 browser canary directly from the corrected post-SFT2 actor and ID74 planning sidecar, before any joint PPO update. Compare this step0 evidence with ID187 source20 while validating that the browser also works on the no-resume bootstrap path.

This is a stochastic cross-run comparison. The environment transport URL is part of the sticky `rollout_sample_id`, so ID187 and ID188 must have distinct transport-bound identities. They are paired only by stable semantic fields (`data_source`, eval set, seed, and task text); identities or policy draws must not be rewritten to force equality.

## Code and entrypoints

- Parent state-process implementation commit: `4f76c926d1f41c62bb309d59f8d2a561a8cb5053`.
- VAGEN: `af06b9288a504ccc739749a0e643dd01a28effcc`.
- VERL: `494f264494b2525f2c13595f63ac4912963e6d2f`.
- Slurm: `experiments/training/rl/id188_step0_rollout_browser_canary.slurm`.
- Allocation launcher: `experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh` with `ID185_VIS_SOURCE_BOUNDARY=0`.
- Runner: `experiments/training/rl/run_vagen_k4_id188_step0_browser_canary.sh`.
- VAGEN config/gate: `external/VAGEN/vagen/configs/joint_id188_step0_visualize_one.yaml`, implementation `id188_k4_step0_browser_v1`.
- Python: `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.

## Data and task

- VAGEN Navigation heldout Base test split.
- `data_source=navigation_base_test_id188`; semantic Base seed `2`.
- The runtime verifies all five heldout assets, explicit seeds 1--60, and zero train-scene overlap before selecting one row.
- Maximum actions `20`; one newly generated rollout.

## Step0 initialization and frozen state

- Actor/model root: corrected ID176 post-SFT2 checkpoint at `outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint`.
- Planning root: ID74 epoch1 `SharedSlotProjector`, WM predictor, and `ValueHead` at `outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- Frozen-Q owner bootstraps source step `776` directly from these weights.
- `resume_mode=disable`, loaded global step `0`, `val_only=true`.
- Actor, vision tower, projector, WM, ValueHead/critic, and reference behavior are frozen. No optimizer update or checkpoint may occur.

## Policy and planner

- Scheme-B `alpha=1`, `beta=85.78297006578457`, prior temperature `1`.
- K4 history `1`, horizon `4`, `100` UCT simulations, exploration `1.0`.
- Sampling temperature `0.7`, top-p `0.95`.
- Rollout TP8/DP1 with `mm_encoder_tp_mode=data`; actor initialization DP8.

## Identity and outputs

- W&B project: `vagen`.
- Run: `188_smoke_rollout_browser_k4_dp8_tp8_step0_base_seed2_t20_s100_retry3`.
- Run ID: `nimloth-id188-smoke-rollout-browser-k4-step0-seed2-r3`; `resume=never`.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-21/188_smoke_rollout_browser_k4_dp8_tp8_step0_base_seed2_t20_s100_retry3`.
- Browser: `evaluation_browser/global_step_0/index.html`.
- A failed retry requires a new output/W&B identity; partial evidence cannot be combined or resumed.

## Slurm

- Partition `preempt`; heterogeneous component0=`1 node × 6 H800, 48 CPU, 192 GiB`, component1=`1 node × 2 H800, 16 CPU, 64 GiB`; 8 GPUs/64 CPU/256 GiB total; 5-hour limit.
- Expected healthy runtime around 10--15 minutes after allocation.
- Allocation exclusions: `dgx-09,dgx-13,dgx-32,dgx-51`.
- Navigation/Ray head exclusions and dynamic 150-second FloorPlan1 qualification are identical to ID187.
- ID187 and ID188 are independent and have no Slurm dependency; each may start whenever one valid exact4×2 allocation is available.

## Completion and comparison gates

1. Log contains `ID188_K4_STEP0_BOOTSTRAP_OK global_step=0` and no restore marker.
2. Exactly one validation row and one globally unique identity are published.
3. Every turn reports `snapshot_source_step=776` and one consistent SHA256 snapshot identity.
4. Journal, manifest, rollout audit, PNGs, HTML, per-turn float32 state `.npz`, and complete marker pass tensor metadata/finite and hash gates.
5. Browser and legacy audit agree on identity, outcome, turn count, action evidence, and full MCTS capability; browser additionally contains same-generation latent/current state, every unique predicted node state, and all 100 chronological UCT/leaf/backup traces.
6. `validation/0.jsonl` exists; no `global_step_*` checkpoint exists.
7. W&B contains only evaluation step0 and finishes.
8. Cleanup leaves no runtime-owned process.
9. After both runs finish, compare task text, success/reward/turn count, executed actions, priors, direct Q, backed-up MCTS root values/visits, and snapshot/source identity. Clearly disclose that stochastic draws are not paired.

## Full Base60+Common60 migration

The user requested the previously planned full pre-RL evaluation and declined
additional policy/split questions. The adopted comparison is the direct
apples-to-apples default: ID176 actor + frozen ID74 planner, evaluated with the
same held-out Base60+Common60, t20, K4/100-simulation protocol as ID189.

- Parent runtime commit: `19ea311badf8e76b2680ba73df42a6daee5ea2d0`; VAGEN feature commit: `9f1e89eb8c9839a406b6e62aa75703494a79e5b5`.
- Config: `joint_id188_step0_base_common120.yaml`, phase
  `step0_base_common120`.
- Runner: `run_vagen_k4_id188_step0_base_common120_normal4x2.sh`.
- Slurm: `id188_step0_base_common120_normal4x2.slurm`, normal 4x2 H800,
  64 CPU/256 GiB, five-hour limit; expected 1--3 hours and 40--60 GiB output.
- Data: exact held-out Base/Common assets, seeds1..60 each, zero train-scene
  overlap; 120 unique rows.
- Initialization: ID176 actor and ID74 planning source step776;
  `resume_mode=disable`, global step0. Every module is frozen and no checkpoint
  may be written.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-24/188_eval_rollout_browser_k4_dp8_tp8_step0_base_common120_t20_s100_normal_4x2`.
- W&B project/name/ID: `vagen` /
  `188_eval_rollout_browser_k4_dp8_tp8_step0_base_common120_t20_s100_normal_4x2` /
  `nimloth-id188-eval-rollout-browser-k4-step0-base-common120-normal-4x2`;
  resume never.
- Browser target: `evaluation_browser/global_step_0/index.html`; 120 rollouts,
  full float32 state archives and all 100 chronological MCTS simulations per
  turn. Ray must observe browser packing workers=`8` on every batch.
- A failed formal identity cannot resume or be overwritten.

VPN recovered. Remote Parent/VAGEN regressions passed `14/11`, all five production trees are clean, and checkpoint/asset/split/output/W&B/hash preflight passed. Job `530662` requests normal 4x2 H800 and is pending Priority; the cluster query showed no currently free GPUs and Slurm estimated start `2026-08-25T17:25:32`. No output or W&B run exists before allocation.
