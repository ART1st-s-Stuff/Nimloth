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

- Parent implementation commit: `fa821410`.
- VAGEN feature commit: `b640182` on the ARTI5T feature branch; true upstream
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

- Local shell/Python syntax and diff checks passed.
- Remote pytest/preflight is currently **blocked before execution**: SSH through
  `superpod-csejzhang` timed out while opening the server connection, indicating
  the required VPN path is unavailable. No Slurm job, output directory or W&B
  identity has been created.
- After VPN restoration: run the specified parent/VAGEN test set in the isolated
  test worktree with bytecode/cache disabled, create a clean production
  worktree, verify checkpoint hashes/output/W&B absence and current resources,
  then submit and monitor through browser completion and read-only validation.
