# 2026-08-06: PPO ValueHead full retry ID134

## Status

- Formally submitted as Slurm Job `507599`; it started ten seconds later on
  `normal/dgx-54:8` at `2026-08-06T02:00:57+08:00`.
- Human authorization remains the explicitly requested corrected retry of the
  full-scale single-node/eight-GPU experiment.
- Exact runtime commit:
  `f95b8c3394553dc30ad4d1f444941bd86df6f037`.

## Purpose and objective

- Train the planner ValueHead with PPO clipping while differentiable full-prefix
  state recomputation propagates critic gradients into the Qwen language body.
  The planner owns the executed action; direct-Qwen actor PPO remains disabled.
- Outgoing executed-action `Q(s_t,a_t)`, objective metadata
  `receding_horizon_decision_state_mc_v2`, gamma 1, zero truncated bootstrap,
  ValueHead clip 0.2, four critic epochs. Real generated/recorded CoT only.
- Trainable: Qwen language body, WM predictor, ValueHead. Frozen: Qwen vision,
  StateProjector, `lm_head`, DINO teacher, direct-Qwen actor/token policy.

## Initialization, data, and schedule

- Fresh SFT2 ID74 epoch-1 initialization:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- No RL resume checkpoint. ID132/133 rollout, optimizer, output, or identity is
  not reused.
- Config:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml`.
- 60 updates; 16 strict fresh trajectories per update, maximum 20 actions;
  `base_train` and `common_sense_train` each receive eight episodes. A failed
  episode is retried at most three times with the same ID/dataset/seed.
- Greedy H1 planner, K16 grid states, DINO auxiliary weight 0.5. Full held-out
  `base` 60 + `common_sense` 60 evaluation follows every ten updates.

## Reliability corrections retained

- Decoded close must terminate the exact tokenizer prefix before query
  injection; validation splits at the final terminal close and retains strict
  token round-trip.
- The one-node batch hard-requires
  `run_vllm_online_ppo_parallel_slurm.sh`, producing two TP4 shards of eight
  episodes and four synchronized two-GPU training ranks.
- Runtime commit `f95b8c33` creates only `RUN_OUT`'s parent before the first
  adjacent progress-log write. It preserves the empty `RUN_OUT` gate and fixes
  ID133's new-date startup failure. A real shell-order regression and the full
  focused suite passed: 20 and 157 tests respectively.

## Identity, output, and recovery

- W&B entity:
  `art2nd-hong-kong-university-of-science-and-technology`.
- Project: `nimloth-rl`.
- Run name:
  `134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Formal output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Start only if the exact W&B name is unused and both output and adjacent
  progress paths are absent.
- Each completed update atomically commits consumption and a resumable
  checkpoint. Before the first update there is no resume boundary; a startup
  failure requires another new identity.

## Runtime and resources

- Worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Batch-owned entrypoint:
  `experiments/training/rl/train_8gpu_1x8.slurm`.
- Slurm `normal`, one node, eight H800 GPUs, 128 CPUs, 96 GiB, eight hours;
  maximum 64 GPU-hours. Exclude `dgx-32,dgx-37,dgx-51`.
- New port bases: Ray 7540, environment 9640, train rendezvous 32640.

## Launch gates

- Exact clean runtime commit/submodules; complete SFT2 model and planner files;
  root has no `rl_state.pt`.
- Re-read actual asset counts/splits; parse exact config including attempts=3;
  repeat shell, Python, dry-run, W&B uniqueness, empty output, live resources,
  and `sbatch --test-only` checks.
- After submission record the exact job/allocation. If allocated, require two
  navigation prewarms and two distinct TP4 engines, then the strict 16-rollout
  merge and first finite synchronized PPO update before declaring health.

## Completed launch validation

- Server runtime was tracked-clean at exact commit `f95b8c33`; pinned VAGEN and
  LeWM gitlinks matched. Required SFT2 files remained complete and no root
  `rl_state.pt` exists.
- Exact config/data validation again reported attempts 3, one node/world4 x
  two GPUs, total eight/TP4, counts 1200/1200/60/60, and zero train/eval scene
  overlap. Shell syntax, login dry preflight, W&B uniqueness, absent output and
  adjacent progress path all passed.
- Exact corrected runtime focused regression passed `157 tests`. Live resources
  contained multiple healthy idle eight-GPU normal nodes; test-only accepted the
  8 GPU/128 CPU/96 GiB/eight-hour request. Formal Job `507599` uses the same
  environment, parallel runner, exclusions, and port bases recorded above.
- Runtime passed ID133's parent-path boundary and durably wrote iteration-1
  `starting`. Two distinct shard directories/environment services came up on
  ports 9641/9642, and both real `base_train` navigation prewarms passed in
  12.439 seconds (shard 0 seed 1, shard 1 seed 5).
- Two distinct EngineCore processes (`1237571`, `1237572`) each initialized a
  four-rank TP group, loaded both safetensor shards, reported 57.81 GiB/GPU
  available KV memory and completed warmup. Both collectors then started real
  eight-episode shards with `max_attempts=3`: shard 0 from
  `rl_base_train_000001`, shard 1 from `rl_base_train_000005`.
- At the last live log read, shard 0 had durably completed its first three
  trajectories on attempt 1 (20, 20, and 20 steps); shard 1 was still generating
  its first episode. No retry, decoded-close error, traceback, CUDA/NCCL/OOM, or
  Slurm stderr was present. This proves corrected startup and real double-TP4
  rollout, not the strict 16-trajectory merge, PPO update, checkpoint, or quality.
- Subsequent SSH ProxyJump attempts were closed with `UNKNOWN port 65535` twice.
  Monitoring stopped rather than repeatedly reconnecting; the batch-owned
  controller is independent of the SSH session and remains responsible for the
  live job.
