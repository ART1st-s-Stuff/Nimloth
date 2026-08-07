# 2026-08-07: ID143 PPO continuation from step 16 to step 20 plus held-out evaluation

## Identity and purpose

- Resume the completed ID141 PPO ValueHead checkpoint at global step 16,
  perform exactly four fresh-policy updates through global step 20, then run
  the standard held-out 120-episode VAGEN evaluation requested by the human.
- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- ID/comment/run name: `143`, comment `continue16`,
  `143_continue16_rl20_eval20x120_greedyh1_k16_dino05_ppo4_ep16x20_2n4r2g_2xtp4`.
  Login-node W&B API preflight found zero exact matches for this name and its
  `-eval` suffix.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/143_continue16_rl20_eval20x120_greedyh1_k16_dino05_ppo4_ep16x20_2n4r2g_2xtp4`.
  The output, adjacent progress log and renderer-attempt root were absent at
  preflight.
- Runtime code commit: `e75e8942de063727bf114ddf917a8058eb14f288`;
  server worktree
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Entrypoints: batch-owned
  `experiments/training/rl/train_8gpu_44.slurm`, outer
  `run_vllm_online_ppo_full.sh`, and two-TP4/multi-rank
  `run_vllm_online_ppo_parallel_slurm.sh`.
- Config:
  `configs/training/rl/planner_greedy_h1_continue16_to20_eval20_16rollout_8gpu_44.yaml`.

## Initialization and lineage

- Initial model, WM and optimizer resume checkpoint are the complete ID141
  `train/final` export. Its `rl_state.pt` is 13,090,012,345 bytes and records
  global step/iteration 16, objective
  `receding_horizon_decision_state_ppo_value_v1`, value clip 0.2/four epochs,
  world size 4, replicated optimizer state, Qwen full tune, frozen vision and
  zero truncated bootstrap.
- ID141 consumption is committed from global step 15 to 16 and identifies
  `train/latest`. `train/latest` and `train/final` are two directory entries to
  the same checkpoint files: their complete relative file names, sizes and
  inodes match, including the shared `rl_state.pt` inode. Therefore `final` is
  the same committed step-16 payload rather than a different checkpoint.
- ID142 failed before any rollout/model load/W&B/optimizer/consumption because
  its submitted `ENV_REPO` pointed to Flower. ID142 output is terminal and no
  artifact from it is reused.

## Data, objective and evaluation

- Each update collects 16 fresh trajectories: eight from `base_train` and
  eight from `common_sense_train`, max 20 environment steps and at most three
  attempts. ID141 consumed per-dataset seeds 145--152; ID143 sets
  `FIRST_ITERATION_SEED_OFFSET=153`, giving per-dataset seeds 153--160 at step
  17, 161--168 at step 18, 169--176 at step 19 and 177--184 at step 20.
- Objective remains executed-action `Q(s_t,a_t)` under
  `receding_horizon_decision_state_ppo_value_v1`: gamma 1, zero truncated
  bootstrap, PPO ValueHead clip 0.2, four critic epochs, greedy H=1/K=16,
  history 1 and DINO weight 0.5.
- Train Qwen language body, WM predictor and ValueHead. Freeze Qwen vision,
  StateProjector, lm_head, DINO teacher, actor PPO and token PPO.
- After committed step 20, evaluate held-out `base` and `common_sense`, 60
  episodes each, seeds 1--60, split `eval`, profile `vagen_eval`, temperature 0
  and top-p 1. Success requires exactly 120 merged trajectories, per-split and
  overall metrics, `evaluation/iter_0020/eval_done.flag`, and a finished
  `-eval` W&B run. Train-rollout success remains distinct from held-out result.

## Resource and resume contract

- Per the human's instruction, request `preempt` with two nodes x four H800
  GPUs, 64 CPUs and 48 GiB RAM per node for three hours, total eight GPUs/128
  CPUs/96 GiB and at most 24 GPU-hours. Enable Slurm requeue and exclude known
  incompatible `dgx-32,dgx-37,dgx-51`.
- The batch owns the full controller lifecycle. On every fresh allocation or
  requeue it verifies the exact Nimloth commit, VAGEN commit
  `192c35a91f3941b72d5e1272af6603ef7a7d93e0`, LeWM commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, W&B identities and actual 4+4
  allocation, then runs an exact single-visible renderer probe on rollout slot
  0 of both allocated nodes before rollout.
- Recovery is fail-closed at the last contiguous committed iteration. An
  interrupted fresh batch is archived and recollected; no failed or partial
  rollout is reused. A due committed step-20 evaluation is completed before
  any further state transition.

## Pre-launch evidence

- Server worktree is at exact runtime commit and tracked-clean; its only status
  noise is existing ignored/untracked Python cache inside LeWM. Bash syntax,
  `test_full_runner.py` 8/8 and `test_slurm_allocation.py` 27/27 pass remotely.
- Exact config load returns iterations 20, nodes 2, world size 4, two GPUs per
  rank, built-in validation disabled, external validation enabled every ten
  steps, 120 environments and eval datasets `(base, common_sense)`.
- Login-node dependency gate accepts the correct Nimloth `ENV_REPO` and rejects
  the former Flower path. The explicit `.venv-vagen-main/bin/python3` is used;
  no copied activate script or console-script shebang is trusted.
- Current preempt snapshot has several mixed nodes. The exact `sbatch
  --test-only` request is accepted and estimates
  `2026-08-07T18:48:42+08:00` on `dgx-01,dgx-16`; this estimate is volatile and
  does not constrain the formal job to fixed nodes.
- No new durable memory is proposed. Relevant pending project memories were
  checked against live source/evidence for Python selection, train/eval split,
  W&B source, planner DDP, batch-owned Slurm lifetime, `dgx-51` exclusion and
  fixed VAGEN validation identity; memory state is not modified.

## Planned submission

```bash
sbatch --account=peilab --partition=preempt --nodes=2 --ntasks-per-node=1 \
  --cpus-per-task=64 --gres=gpu:4 --mem=48G --time=03:00:00 --requeue \
  --exclude=dgx-32,dgx-37,dgx-51 --job-name=id143-ppo-44 \
  --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out \
  --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err \
  --export=ALL,<the exact variables below> \
  /project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9/experiments/training/rl/train_8gpu_44.slurm
```

The exported contract sets `REPO=ENV_REPO` to the server Nimloth worktree,
`PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`, the runtime
and dependency commit gates above, the exact config/output/checkpoint paths,
`INITIAL_GLOBAL_STEP=16`, `TOTAL_ITERATIONS=20`,
`FIRST_ITERATION_SEED_OFFSET=153`, `WANDB_PROJECT=nimloth-rl`, the exact run
name, online W&B mode, and a fresh renderer attempt root ending
`143_preempt44`.

## Current status

- Formal Job `509368` was submitted at `2026-08-07T14:31:17+08:00` with the
  exact command and exports above. Slurm confirms `Requeue=1`, two requested
  nodes, 128 CPUs, 96 GiB total RAM, eight GPUs total,
  `TresPerNode=gres:gpu:4`, and exclusions `dgx-32,dgx-37,dgx-51`.
- The job is currently `PENDING(Priority)`, elapsed zero, with no node list and
  `AllocTRES=(null)`. No output, W&B run, renderer, rollout, optimizer,
  consumption or checkpoint has started. The monitor will use the actual
  allocation, not the earlier test-only estimate.
