# 2026-08-05: PPO ValueHead full experiment ID132

## Status

- Failed before the first PPO update. Slurm Job `506953` ran on
  `normal/dgx-52:8` from `2026-08-05T20:59:22+08:00` to `21:12:03+08:00`, then
  ended `FAILED (NonZeroExitCode, 1:0)` after 12 minutes 41 seconds.
- Exact submitted runtime commit:
  `6acd0d7cd804e71682079c964ad4818f2d25cbd7`. The batch enforced this as
  `EXPECTED_COMMIT` before creating the formal run.
- `scontrol` confirms `ReqTRES=cpu=128,mem=96G,node=1,gres/gpu=8`,
  `TresPerNode=gres:gpu:8`, eight-hour limit, and exclusions
  `dgx-[32,37,51]`; the actual allocation was all eight GPUs on `dgx-52`.
- Navigation prewarm and one TP4 vLLM engine completed. Iteration 1 persisted 15
  of 16 attempted trajectories, then failed the strict complete-batch gate. No
  fresh manifest, consumption, optimizer step, checkpoint, held-out evaluation,
  or W&B run exists. ID132 is not resumable.

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

## Completed launch validation

- Live normal resources had no healthy fully idle eight-GPU node; `dgx-35` was
  `DOWN+NOT_RESPONDING` and was not counted. W&B credentials resolved the live
  entity and found zero existing ID132 names.
- Runtime worktree was clean at the submitted commit. VAGEN was initialized at
  exact gitlink `192c35a91f3941b72d5e1272af6603ef7a7d93e0`; LeWM matched
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Server Python targeted regression passed 99 tests. Shell syntax and config
  parsing passed. Iteration-1 dry preflight reported 16 episodes, 20 steps,
  one node, world size four, two GPUs per rank, eight total GPUs, and TP4.
- Actual assets contain 1,200 tasks each for `base_train` and
  `common_sense_train`, 60 each for held-out `base` and `common_sense`, and zero
  train/eval scene overlap. Required SFT2 files are non-empty and the root has no
  `rl_state.pt`, confirming fresh RL initialization.
- `sbatch --test-only` accepted the exact resource and environment contract.
  Formal submission then created Job `506953`; only the batch-owned controller
  owns the future experiment lifecycle.

## ID132 failure boundary

- AI2-THOR navigation prewarm passed on `base_train` seed 1 in 11.064 seconds.
  Ray started cleanly; the actual TP4 engine completed both checkpoint shards,
  NCCL/Gloo setup, KV-cache initialization, and real planner rollout generation.
- Episode `rl_000005` (`base_train`) failed after its fifth action with
  `RuntimeError: vLLM decoded '</think>' did not end at query injection`.
  The collector correctly discarded that incomplete trajectory and continued.
- The immutable JSONL contains 15 trajectories, IDs `rl_000001` through
  `rl_000016` excluding `rl_000005`, with 277 executed transitions and two
  successes. This incomplete-batch success count is not a training metric.
- Final failure was the strict guard
  `rollout produced an incomplete trajectory batch: 15 != 16`. There is no
  `fresh_policy_manifest.json`, rollout-consumption record, `train/latest`, PPO
  forward/backward, optimizer step, held-out evaluation, or W&B run.
- The partial rollout is unmanifested and must not be consumed or reused. No RL
  checkpoint exists, so retry cannot resume ID132; it requires a corrected code
  and runner contract, a new experiment identity, and an empty output initialized
  from the same SFT2 epoch-1 checkpoint.
- A separate launch-contract deviation was also confirmed. Submission set
  `ITERATION_RUNNER=run_vllm_online_ppo_slurm.sh`, so only one TP4 engine ran.
  `ROLLOUT_WORKERS=2` is consumed only by
  `run_vllm_online_ppo_parallel_slurm.sh`. This reduced rollout parallelism and
  violated the recorded two-worker topology, although it did not directly cause
  the decoded-`</think>` failure.
- The output README now records the exact job, command, commit, configuration,
  data/checkpoint contract, artifacts, cause, and non-resumable boundary.
