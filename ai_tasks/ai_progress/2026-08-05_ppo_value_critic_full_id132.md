# 2026-08-05: PPO ValueHead full experiment ID132

## Status

- Approved for submission by the human. The requested resource is one physical
  eight-H800 node in the `normal` partition. Live resource inspection found no
  healthy fully idle normal node, so the expected initial Slurm state is pending.
- This document is the immutable pre-submission contract. The exact runtime
  commit is the commit containing this contract and is supplied to the batch as
  `EXPECTED_COMMIT`; the batch records and enforces it before creating the run.

## Scientific purpose and evidence boundary

- Test whether planner PPO supervision of the ValueHead, with differentiable
  full-prefix state recomputation, improves the receding-horizon planner while
  propagating critic gradients into the Qwen language body.
- The planner owns the executed action. Direct-Qwen actor PPO is disabled, so
  the executed action need not be the maximum-logit Qwen action token.
- Training-rollout success is an optimization diagnostic. Policy-quality claims
  require the scheduled held-out 120-episode evaluations and comparison against
  the compatible SFT2/iteration-0 baseline; no conclusion is implied at launch.

## Immutable initialization and objective

- Fresh model, WM predictor, StateProjector, and ValueHead source:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- RL resume checkpoint: none. Historical ID125 uses the predecessor objective
  and is not resumed or mixed into this run.
- Objective: outgoing executed-action `Q(s_t, a_t)` with
  `receding_horizon_decision_state_mc_v2`, `gamma=1`, and zero truncated
  bootstrap. Actual recorded/generated CoT is used; no fixed or invented CoT is
  introduced.

## Training and evaluation contract

- Config:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml`.
- Batch-owned controller:
  `experiments/training/rl/train_8gpu_1x8.slurm`.
- 60 iterations; 16 fresh training episodes per iteration, split evenly between
  `base_train` and `common_sense_train`; maximum 20 steps per episode.
- Greedy horizon-1 planner; K16 world-model candidates; DINO-grid auxiliary loss
  weight 0.5.
- PPO ValueHead clip range 0.2 and four critic epochs per collected batch.
- Held-out evaluation every ten iterations: all 60 `base` and all 60
  `common_sense` episodes, reported as the standard 120-episode
  `val_success_rate` contract.
- Trainable: Qwen language body, WM predictor, and ValueHead. Frozen: Qwen vision,
  StateProjector, `lm_head`, DINO teacher, and direct-Qwen actor/token policy.
- Distributed topology: four synchronized two-GPU training ranks; during
  rollout the same node hosts two independent TP4 vLLM workers.

## Identity, outputs, and observability

- Formal output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-05/132_ppo_value_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- W&B entity: `art2nd-hong-kong-university-of-science-and-technology`.
- W&B project: `nimloth-rl`.
- W&B run name:
  `132_ppo_value_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Live preflight found no existing output and no W&B run whose name starts with
  `132_`. W&B mode is online.
- Checkpoint and consumption state are committed after every iteration; periodic
  snapshots are retained every ten iterations. Fresh rollout manifests bind each
  update to its generating policy and fail closed on stale reuse.

## Runtime and resource contract

- Runtime worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- `ENV_REPO` is the same parent Nimloth runtime worktree, so the controller
  resolves its pinned `external/VAGEN` exactly once.
- Slurm: account `peilab`, partition `normal`, one node, eight GPUs, 128 CPUs,
  96 GiB RAM, eight-hour walltime. Maximum cost for one allocation is 64
  GPU-hours. The job is not pinned to a particular node; it excludes
  `dgx-32,dgx-37,dgx-51` according to the established navigation-node health and
  scheduling gate, and must not count `DOWN+NOT_RESPONDING` nodes as available
  capacity.
- A single allocation may not finish all 60 iterations. Any continuation must
  use the latest crash-consistent committed checkpoint, a new Slurm job, and the
  same formal run identity; it must not replay an already consumed rollout.

## Launch gates

- Exact clean runtime commit and pinned submodules.
- Complete non-empty SFT2 epoch-1 model/WM/StateProjector/ValueHead files.
- Exact dataset counts and train/held-out split boundaries.
- Config and shell syntax, Python/import environment, W&B credentials and unique
  identity, absent output path, node topology, and batch-owned lifecycle.
- After submission, record Slurm job ID/state/reason. If allocated, require both
  AI2-THOR prewarms, both TP4 vLLM groups, the first strict 16-rollout merge, and
  the first finite synchronized PPO update before describing runtime health.
