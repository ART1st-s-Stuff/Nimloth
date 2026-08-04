# 2026-08-04: RL ID123 from corrected SFT2 epoch 1

## Human decision

- Restart RL directly from the corrected SFT2 checkpoint, not from any prior
  RL checkpoint, optimizer state, rollout, or consumption marker.
- Use the Slurm `normal` partition with a physical `4+4` GPU allocation:
  two nodes, four H800 GPUs per node, eight GPUs total.

## Fixed launch contract

- Nimloth branch: `fix/rl-text-stop-token-budget`.
- Topology implementation commit: `6b3cc921`.
- Config:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_44.yaml`.
- Batch-owned entrypoint: `experiments/training/rl/train_8gpu_44.slurm`.
- Initial SFT2 checkpoint:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- The checkpoint was re-read on the server: epoch 1 is complete at step 776;
  model shards, StateProjector, WM predictor, ValueHead, and training state are
  non-empty. Its invariants record `history_size=1`,
  `prediction_horizon=4`, DINO weight 0.5, and ValueHead objective
  `decision_state_executed_action_mc_v3`.
- Initialization is fresh at RL global step 0. `INITIAL_RESUME_CHECKPOINT` is
  empty, so no old RL optimizer is loaded.
- Trainable: Qwen language body, WM predictor, and ValueHead.
- Frozen: Qwen vision, StateProjector, DINO teacher, and latent query tokens.
- Planner: greedy, planning horizon 1; direct Qwen PPO and reference KL are
  disabled. WM/DINO/ValueHead transition training remains enabled.
- Schedule: 60 iterations, 16 fresh training episodes per iteration, at most
  20 environment steps per episode. Training assets are `base_train` and
  `common_sense_train` with advancing, non-overlapping seed blocks.
- Evaluation: after every 10 committed iterations, run greedy held-out
  evaluation on `base` seeds 1--60 and `common_sense` seeds 1--60, 120 episodes
  total. Evaluation never enters the optimizer.
- Token safety: decoded literal `</think>` triggers query injection and every
  formal state is capped at 16,384 processor-expanded tokens before rollout
  action execution and again before Qwen training forward.
- Distributed layout: two TP4 rollout workers; four synchronized training
  ranks with two-GPU Qwen model parallelism per rank. All eight GPUs train.
- W&B project: `nimloth-rl`. Live maximum ID was 122 and no ID123 match
  existed at preflight time.
- Planned run name:
  `123_sft2ep1_restart_rl16_eval10x120_greedyh1_k16_dino05_qwenwmvalue_iter60_ep16x20_2n4r2g_2xtp4_normal44`.
- Planned output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-04/123_sft2ep1_restart_rl16_eval10x120_greedyh1_k16_dino05_qwenwmvalue_iter60_ep16x20_2n4r2g_2xtp4_normal44`.
- Resume boundary: every committed iteration has an immutable policy input,
  complete post-update checkpoint, and atomic rollout-consumption commit. A
  preemption may resume only from the latest complete committed iteration.
- Requested cost per Slurm segment: two nodes, eight H800 GPUs total, up to
  eight hours (`<=64 GPU-hours`). The 60-iteration run may require additional
  resume segments; wall time will be measured from the first healthy update.

## Validation and current state

- Added the 16-rollout 8-GPU config without changing algorithm code or the
  existing 4+4 batch controller.
- Local `bash -n`, Python compile with an external pycache, and
  `git diff --check` passed. New config tests assert equality with the verified
  12-GPU objective/evaluation contract and assert the 4+4 routing gates.
- Current Slurm snapshot before submission had no user jobs. The `normal`
  partition had 12 free GPUs, but at most three on any one node, so the 4+4
  request cannot start immediately and must queue until two nodes each expose
  four GPUs.
- No ID123 output, W&B run, rollout, optimizer step, checkpoint, or Slurm job
  has been created yet. Remote exact-commit regression and the final launch
  preflight remain required before submission.
