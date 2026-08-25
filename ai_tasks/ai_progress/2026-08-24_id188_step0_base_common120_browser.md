# ID188 pre-RL step0 Base+Common120 Rollout Browser

## Goal

Evaluate the corrected post-SFT2, pre-joint-RL policy with the same held-out
Base60 + Common Sense60 K4/100-simulation protocol as ID189, then publish the
complete interactive Rollout Browser.

## Fixed contract

- Actor: ID176 corrected post-SFT2 checkpoint
  `outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint`.
- Planner: frozen ID74 epoch1 projector, predictor and ValueHead; direct source
  step `776` bootstrap.
- Policy: Scheme-B alpha1/beta85.78297006578457, K4 horizon4, 100 UCT
  simulations, exploration1.0, temperature0.7/top-p0.95.
- Data: held-out Navigation Base seeds1..60 plus Common Sense seeds1..60; 120
  rows total. Asset hashes and train-scene disjointness are checked before model
  startup.
- Execution: actor initialization DP8; rollout TP8/DP1,
  `mm_encoder_tp_mode=data`; normal 4 nodes x 2 H800.
- Frozen/read-only: actor, vision tower, projector, WM, ValueHead/critic and
  reference behavior. `val_only=true`, `resume_mode=disable`; no backward,
  optimizer update, snapshot publication or checkpoint write.
- Canonical output:
  `outputs/experiments/training/rl/2026-08-24/188_eval_rollout_browser_k4_dp8_tp8_step0_base_common120_t20_s100_normal_4x2`.
- Browser: `evaluation_browser/global_step_0/index.html`.
- W&B: project `vagen`, name
  `188_eval_rollout_browser_k4_dp8_tp8_step0_base_common120_t20_s100_normal_4x2`,
  ID `nimloth-id188-eval-rollout-browser-k4-step0-base-common120-normal-4x2`,
  resume never.
- Resources: normal 4x2 H800, 64 CPU, 256 GiB, five-hour hard limit; expected
  runtime 1--3 hours and output roughly 40--60 GiB.

## Implementation

- Parent runtime commit: `19ea311badf8e76b2680ba73df42a6daee5ea2d0`.
- VAGEN feature commit: `9f1e89eb8c9839a406b6e62aa75703494a79e5b5` on the ARTI5T feature branch; true upstream
  main is unchanged.
- Added full step0 config/gate, four-node runner, Slurm entrypoint and TDD
  contract tests.
- Ray launcher now propagates
  `VAGEN_ROLLOUT_BROWSER_PACK_WORKERS=8` into every raylet/TaskRunner and checks
  the observed value. The post-run validator requires all three browser batches
  to log effective `workers=8`.
- The output guard prepares only the parent, verifies the child is absent, then
  creates the fresh child with ordinary `mkdir`.
- Each real turn retains image, actual response/CoT, prior/action/Q/value,
  float32 latent/current/predicted states, all candidates and all 100
  chronological MCTS simulations.

## Validation and status

- VPN recovered. Remote regression passed in the isolated test worktree with bytecode/cache disabled: Parent `14 passed`; VAGEN `11 passed` (three existing warnings only). Shell syntax and all parent/VAGEN/VERL/le-wm/RCDM clean-tree gates passed.
- Clean production worktree:
  `/project/peilab/atst/nimloth/.worktree/id188-step0-bc120-prod-19ea311b`.
- Checkpoint hashes, exact heldout asset hashes and train-scene disjointness passed. The formal output is absent, W&B ID is absent, and the fixed Python/config sources are readable. NFS reported about 361 GiB available before submission.
- The first `sbatch` command used a relative script path without changing to the production worktree and was rejected before creating a Slurm job; no output or W&B identity was touched. The corrected absolute-path submission created Job `530662`.
- Job `530662` is currently `PENDING (Priority)` on `normal`, requesting four nodes, eight H800 total, 64 CPU and 256 GiB. Slurm estimated start `2026-08-25T17:25:32`; the resource query showed no currently free GPUs.
- Continue monitoring through allocation, dynamic Navigation head qualification, Ray `[2,2,2,2]`, DP8/TP8 startup, all three 40-row browser batches, final validator, W&B finish and cleanup.
