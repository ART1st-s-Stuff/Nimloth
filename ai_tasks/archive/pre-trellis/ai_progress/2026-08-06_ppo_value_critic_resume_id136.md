# 2026-08-06: PPO ValueHead resume ID136 on dgx-52 1x8

## Status

- Terminal failure before the first optimizer step. Formal Slurm step
  `508346.3` ran on the resource-only hold Job `508346` at
  `normal/dgx-52:8`, then hit a repeated NCCL collective mismatch during the
  first PPO critic epoch. ID136 is non-resumable and none of its rollout is
  reusable.
- Exact runtime commit:
  `d6197e843fcbfbfe59185b0280c1e6c1acccbfdc`.

## Purpose and objective

- Continue the corrected planner critic experiment through the configured
  60-update horizon. PPO clipping supervises executed-action `Q(s_t,a_t)` and
  differentiable full-prefix recomputation propagates the critic gradient into
  the Qwen language body. Direct-Qwen actor PPO remains disabled.
- Objective metadata:
  `receding_horizon_decision_state_ppo_value_v1`, gamma 1, zero truncated
  bootstrap, ValueHead clip 0.2 and four critic epochs. Only model-generated and
  persisted CoT is consumed.
- Trainable: Qwen language body, WM predictor and ValueHead. Frozen: Qwen
  vision, StateProjector, `lm_head`, DINO teacher and direct-Qwen actor/token
  policy.

## Resume, data and evaluation

- Immutable initialization and optimizer resume checkpoint:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8/train/policy_inputs/iter_0016`.
- The checkpoint records iteration/global step 15. ID136 starts iteration 16
  with fresh per-dataset seeds 121--128; no ID134 iteration-16 or ID135 partial
  rollout is reused.
- Training uses actual VAGEN `base_train` and `common_sense_train` assets only,
  eight episodes per dataset per update and at most 20 actions. Live preflight
  counted 1200/1200 training tasks and 60/60 held-out `base/common_sense` tasks;
  both corresponding train/eval scene intersections were empty.
- External held-out evaluation remains the fixed disjoint 120-episode contract
  every ten committed updates. Iterations are atomic: consumption is committed
  only after a complete `train/latest/rl_state.pt`; a later retry must use a new
  identity and the newest committed policy checkpoint.

## Identity and output

- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- Run ID/name:
  `136_ppo_value_syncfix_dgx52_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Exclusive output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/136_ppo_value_syncfix_dgx52_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Immediately before launch, the exact W&B name had zero matches and both the
  output and adjacent iteration-progress path were absent.

## Runtime and resources

- Server worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Config/entrypoint:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml` and
  `experiments/training/rl/train_8gpu_1x8.slurm`, launched as a Slurm step in
  hold `508346`.
- Allocation: `normal/dgx-52`, eight GPUs, 128 CPUs, 96 GiB, ending no later
  than `2026-08-07T01:27:30+08:00`. Maximum remaining segment cost is below
  64 GPU-hours and may stop at an intermediate committed checkpoint.
- Port bases: Ray 7590, environment 9690 and training rendezvous 32730.

## Completed launch gates

- Server tracked worktree content is clean at exact commit `d6197e84`; VAGEN
  and LeWM remain pinned at `192c35a9` and `8edfeb33`.
- Checkpoint/config/data/W&B/output gates passed. Config parses as iterations
  60, strict batch 16, attempts 3, nodes 1, world 4, two GPUs/rank, total eight
  GPUs, two TP4 workers, actor disabled, ValueHead clip 0.2/four epochs and
  external 120-episode validation every ten updates.
- Concurrent exact-visibility render probes on physical GPU 0 and 4 used
  `CUDA_VISIBLE_DEVICES=0/4` with relative `gpu_device=0`; both returned
  `AI2THOR_RENDER_OK` in 10.217/10.466 seconds with image dynamic range 246.
  Artifacts are under
  `outputs/experiments/training/rl/preflight/2026-08-06/136_dgx52_exact_visibility_508346`.
- Before formal launch, `dgx-52` had no GPU compute processes and the selected
  port ranges were free. Training must still be monitored through both formal
  navigation prewarms, both TP4 model warmups, strict 16-trajectory merge and
  the first finite synchronized PPO update/checkpoint before being called
  healthy.

## Terminal result

- Slurm step `508346.3` started at `2026-08-06T17:40:26+08:00`. Both formal
  navigation prewarms passed in 4.957/4.948 seconds, and two distinct TP4
  EngineCore processes loaded the model, allocated 57.81 GiB KV cache per GPU
  and completed warmup. The strict merge then produced all 16 fresh
  trajectories / 319 transitions for seeds 121--128.
- During PPO critic epoch 1, NCCL sequence 6046 diverged: ranks 0 and 3 entered
  the 1,057,800-element ValueHead `ALLREDUCE`, while ranks 1 and 2 entered a
  one-element `BROADCAST`. The watchdog fired after 600 seconds. This proves
  that `broadcast_buffers=False` plus the post-transition backward barrier did
  not remove the production failure; it does not yet identify which remaining
  code path issued that broadcast.
- The stuck Slurm step was cancelled after the watchdog at
  `2026-08-06T18:00:43+08:00`; `sacct` records `CANCELLED+`, exit `0:9`.
  Post-cleanup checks found no Unity, VAGEN, Ray, vLLM or GPU compute process on
  `dgx-52`. The outer hold Job `508346` remains running and reserved for a
  bounded diagnostic or corrected GPU gate.
- No optimizer step, metric row, `rl_state.pt`, policy checkpoint or held-out
  evaluation was produced. `train/train_step_log.csv` contains only its
  header. The fresh manifest consumption record remains `in_progress`, so
  ID136 and its iteration-16 rollout are forbidden from reuse.
- The only valid recovery boundary remains ID134
  `train/policy_inputs/iter_0016` at committed iteration/global step 15. Any
  later retry requires a new identity, empty output, unused W&B identity and a
  fresh iteration-16 rollout.
- W&B run `f5otsqrv` was finalized as `failed` with
  `terminal_status=failed_collective_timeout_before_optimizer`. The immutable
  output README and adjacent/RL-group progress records contain the same
  terminal boundary; their SHA256 values are respectively
  `b9ebf8da6409c09473938ca1ff8e339586b8782016c55bf605342a3f8f35abdc`,
  `226c99872aee55510fc2c5c37e3690331195f2df11f2003a4c37103bcab89227` and
  `2fb2794d81c5922e4d2b4bb877546ba95b05fdb0a8276ce7f1b0ee8db12f5674`.
