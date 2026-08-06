# 2026-08-06: PPO ValueHead resume ID136 on dgx-52 1x8

## Status

- Ready for formal launch inside resource-only hold Job `508346` on
  `normal/dgx-52:8`. ID135 is terminal and none of its partial rollout is
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
