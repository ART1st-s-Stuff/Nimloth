# 2026-08-07: ID142 PPO continuation from step 16 to step 20 plus held-out evaluation

## Identity

- Purpose: resume the completed ID141 PPO ValueHead checkpoint at global step
  16, train exactly four fresh-policy updates through global step 20, then run
  the standard held-out 120-episode VAGEN evaluation.
- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- ID/comment/run name:
  `142`, comment `continue16`,
  `142_continue16_rl20_eval20x120_greedyh1_k16_dino05_ppo4_ep16x20_1n4r2g_2xtp4`.
  Before launch, the exact training run name has zero W&B matches; evaluation
  will use the corresponding `-eval` suffix.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/142_continue16_rl20_eval20x120_greedyh1_k16_dino05_ppo4_ep16x20_1n4r2g_2xtp4`.
  The output and adjacent iteration-progress log were absent before launch.
- Runtime commit: `5ab8896493ae796e07271c03fe5c14b1c5f84396`;
  server worktree
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Entrypoints:
  `experiments/training/rl/run_vllm_online_ppo_full.sh` and
  `experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh`.
- Config:
  `configs/training/rl/planner_greedy_h1_continue16_to20_eval20_16rollout_8gpu_1x8.yaml`.

## Initialization, data, and objective

- Initial model, WM and optimizer resume checkpoint are all the complete ID141
  `train/final` checkpoint. Its `rl_state.pt` and step-16 consumption sidecar
  were rechecked: the consumption is committed from global step 15 to 16.
- Formal training collects 16 new trajectories per iteration: eight each from
  `base_train` and `common_sense_train`, max 20 environment steps and at most
  three attempts. No prior trajectory is reused.
- ID141 consumed per-dataset seeds 145--152. The continuation therefore sets
  `FIRST_ITERATION_SEED_OFFSET=153`: step 17 uses 153--160, step 18 uses
  161--168, step 19 uses 169--176, and step 20 uses 177--184 for each training
  dataset.
- Objective remains `receding_horizon_decision_state_ppo_value_v1` with
  executed-action `Q(s_t,a_t)`, gamma 1, zero truncated bootstrap, ValueHead PPO
  clip 0.2 and four critic epochs. Planning remains greedy H=1/K=16, history 1,
  DINO weight 0.5.
- Train Qwen language body, WM predictor and ValueHead. Freeze Qwen vision,
  StateProjector, lm_head, DINO teacher, actor PPO and token PPO.
- ID141 already passed the separate real 16,184-token single/DDP mechanics gate
  and a full formal step with this unchanged training implementation. ID142 is
  a direct production continuation; it does not consume the old diagnostic
  corpus or repeat that one-time non-consuming gate.

## Evaluation and terminal contract

- After the step-20 checkpoint is durably committed, the outer runner invokes
  the external evaluation phase on the same allocation and checkpoint.
- Evaluation uses held-out `base` and `common_sense`, 60 episodes each, seeds
  1--60, split `eval`, navigation profile `vagen_eval`, temperature 0 and
  top-p 1. It must merge exactly 120 trajectories and record both per-split and
  overall metrics in `evaluation/iter_0020/rollout_summary.json` and
  `evaluation/eval_step_log.csv`.
- Training success requires four contiguous finite metric rows ending at global
  step 20, committed consumption for iterations 17--20, and complete
  `train/latest` plus `train/final`. Evaluation success additionally requires
  `evaluation/iter_0020/eval_done.flag` and a finished `-eval` W&B run.
- Train-rollout `success_rate` remains separate from held-out evaluation
  success rate and is not policy-quality evidence by itself.
- Resume is fail-closed: the outer controller reconstructs the last contiguous
  committed iteration, archives uncommitted artifacts, and evaluates a due
  committed step before advancing. A failed rollout batch is not reused.

## Resource and launch plan

- Request one normal-partition node with eight H800 GPUs, 128 CPUs and 96 GiB
  RAM for 2 hours 30 minutes. This is at most 20 GPU-hours. At contract freeze
  no complete eight-GPU node was idle, so submit exactly one resource-only hold
  and inspect its expected start time instead of creating competing holds.
- After allocation, run one detached eight-GPU `srun` client with `nohup`, stdin
  `/dev/null`, and durable PID/client/controller logs. This preserves the
  previous detached-lifetime fix and allows reattachment without tying the
  Slurm step to SSH.
- Expected runtime is approximately 35--60 minutes for four rollout/update
  iterations plus roughly 20--40 minutes for the 120 held-out episodes; the
  2.5-hour hold includes recovery margin.

## Pre-launch validation

- Local shell syntax and diff checks passed.
- Server config load confirms iterations 20, built-in validation disabled,
  external validation enabled every 10 steps, 120 episodes, and held-out
  datasets `(base, common_sense)`.
- Server tests passed: `tests/training/rl/test_full_runner.py` 8/8 and
  `tests/training/rl/test_slurm_allocation.py` 25/25.
- Commit `5ab88964` adds an explicit first-iteration seed offset while retaining
  the original formula by default. The new regression proves a resumed output
  can start at seed 153, preventing overlap with ID141's noncanonical retry
  seeds.

## Current status

- Contract frozen and code pushed. The 1x8 hold, training and evaluation have
  not yet been submitted.
- No new durable memory is proposed: the held-out evaluation contract is
  already documented, and the seed offset is a run-specific continuation
  detail.
