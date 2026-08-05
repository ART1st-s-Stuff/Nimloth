# 2026-08-06: PPO ValueHead full retry ID133

## Status

- Failed before experiment setup and is not resumable. Slurm Job `507576` ran
  on `normal/dgx-54:8` from `2026-08-06T01:50:02+08:00` to `01:50:33+08:00`,
  then ended `FAILED (exit 1:0)` after 31 seconds.
- Human authorization is the existing full-scale one-node/eight-GPU contract,
  followed after the ID132 diagnosis by the explicit instruction to modify and
  retry.
- Exact runtime code commit:
  `543991596adda189132666b299148f4b119ef131`.

## Scientific purpose and evidence boundary

- Test whether planner PPO supervision of the ValueHead, with differentiable
  full-prefix state recomputation, improves the receding-horizon planner while
  propagating critic gradients into the Qwen language body.
- The planner owns the executed action. Direct-Qwen actor PPO is disabled, so
  the executed action need not be the maximum-logit Qwen action token.
- Training-rollout success is an optimization diagnostic. Policy-quality claims
  require the scheduled held-out 120-episode evaluations and a compatible
  baseline comparison.

## ID132 corrections

- The vLLM turn processor now injects latent queries only when decoded reasoning
  ends exactly in `</think>`. Policy validation splits at the final terminal
  close and retains strict full-continuation round-trip validation.
- A failed trajectory is retried at most three times with the exact same
  `episode_id`, dataset, and seed. Retry exhaustion still fails the full
  iteration; the 16/16 complete-batch, fresh-manifest, and one-time-consumption
  gates are unchanged.
- The one-node eight-GPU batch rejects any iteration runner except
  `run_vllm_online_ppo_parallel_slurm.sh`, ensuring two independent TP4 rollout
  workers rather than ID132's accidental serial TP4 execution.
- Server regression on the exact source diff passed 156 tests; a focused subset
  passed 9 tests. Shell syntax and `git diff --check` also passed.

## Immutable initialization and objective

- Fresh model/WM/StateProjector/ValueHead source:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- RL resume checkpoint: none. ID132 produced no update or checkpoint, and none of
  its 15 partial trajectories may be reused.
- Objective: outgoing executed-action `Q(s_t, a_t)` with
  `receding_horizon_decision_state_mc_v2`, `gamma=1`, zero truncated bootstrap,
  ValueHead PPO clip range 0.2, and four critic epochs. Actual generated/recorded
  CoT is used; no fixed or invented CoT is introduced.

## Training and evaluation contract

- Config:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml`.
- Batch-owned entrypoint:
  `experiments/training/rl/train_8gpu_1x8.slurm`.
- Iteration runner:
  `experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh`.
- 60 iterations; 16 fresh training episodes per iteration, split evenly between
  `base_train` and `common_sense_train`; at most 20 actions per episode; up to
  three attempts for each fixed episode identity.
- Greedy horizon-1 planner, K16 world-model candidates, DINO-grid auxiliary
  weight 0.5.
- Held-out evaluation after every ten updates: all 60 `base` and all 60
  `common_sense` episodes, reported as the 120-episode `val_success_rate`.
- Trainable: Qwen language body, WM predictor, ValueHead. Frozen: Qwen vision,
  StateProjector, `lm_head`, DINO teacher, direct-Qwen actor/token policy.

## Identity, output, and recovery

- ID: `133`. IDs 126--132 are already used by project experiment records even
  though the live `nimloth-rl` W&B numeric maximum remains 125; no old experiment
  identity is reused.
- W&B entity:
  `art2nd-hong-kong-university-of-science-and-technology`.
- W&B project: `nimloth-rl`.
- W&B run name:
  `133_ppo_value_closefix_retry3_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Formal output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/133_ppo_value_closefix_retry3_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Start only if the output is absent/empty and the exact W&B name is unused.
- Checkpoint/consumption state commits after every completed update; periodic
  snapshots remain every ten updates. A walltime continuation may use only the
  latest crash-consistent committed checkpoint and must not replay consumed
  rollouts. Before the first update there is no resume boundary.

## Runtime and resources

- Runtime worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- `ENV_REPO` is the same parent runtime worktree; its pinned VAGEN gitlink is
  resolved exactly once.
- Slurm account `peilab`, partition `normal`, one node, eight H800 GPUs, 128
  CPUs, 96 GiB RAM, eight-hour walltime; maximum allocation cost 64 GPU-hours.
- Four synchronized two-GPU training ranks use all eight GPUs. Rollout uses two
  independent TP4 workers, each collecting eight episodes.
- Exclude `dgx-32,dgx-37,dgx-51`; do not count `DOWN+NOT_RESPONDING` nodes as
  available. The batch owns the controller lifecycle; external monitoring is
  read-only.

## Required launch gates

- Exact clean runtime commit and expected submodule gitlinks.
- Complete non-empty SFT2 model, WM predictor, StateProjector, and ValueHead;
  root must contain no RL resume state.
- Actual train/eval dataset counts and zero scene overlap.
- Config summary must report one node, world four, two GPUs/rank, total eight,
  TP4, 16 episodes, 20 actions, and three attempts per episode identity.
- Python/import path, shell syntax, required environment, unique W&B name,
  absent output, live resource snapshot, and `sbatch --test-only` acceptance.
- After submission record the Slurm job and current reason. If allocated, require
  both navigation prewarms, both TP4 engines, the strict 16-rollout merge, and
  the first finite synchronized PPO update before calling the run healthy.

## Completed preflight and submission

- Exact runtime commit, tracked-clean server worktree, VAGEN
  `192c35a91f3941b72d5e1272af6603ef7a7d93e0`, LeWM
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, shell syntax, Python, checkpoint
  files, output absence, and W&B uniqueness all passed.
- The SFT2 checkpoint was re-read as complete epoch 1, step 776, K16 inject,
  H=1/T=4, DINO-grid 0.5, and ValueHead objective
  `decision_state_executed_action_mc_v3`; its root contains no `rl_state.pt`.
- Actual assets contain 1,200 `base_train`, 1,200 `common_sense_train`, 60
  `base`, and 60 `common_sense` tasks. Train/eval scene overlap is zero for
  both paired datasets.
- Parsed config reported attempts 3, 16 episodes, 20 steps, batch 16, world
  4 x 2 GPUs, total 8 GPUs, TP4, four ValueHead PPO epochs, and external
  120-episode evaluation every ten iterations. Login dry preflight passed.
- Immediately before submission, the exact W&B name remained unused and the
  output/progress paths were absent. Live normal resources showed several
  eight-GPU idle nodes. `sbatch --test-only` accepted the contract; formal Job
  `507576` requested and received 8 GPUs, 128 CPUs, and 96 GiB on `dgx-54`.

## ID133 failure boundary

- The full controller defines the adjacent progress file as
  `RUN_OUT.iteration_progress.log`, but runtime commit `54399159` created only
  `FORMAL_OUTPUT_ROOT`. The new date parent `.../2026-08-06/` did not exist.
  The first iteration-start progress write therefore failed with
  `No such file or directory`; the EXIT trap repeated the same failed write.
- Failure occurred before Ray, environment startup, either navigation prewarm,
  vLLM/model loading, trajectory collection, fresh manifest, W&B initialization,
  PPO forward/backward, optimizer, consumption, checkpoint, or evaluation.
  Slurm stdout is empty; stderr is 718 bytes and contains only the two parent-path
  errors. Live W&B query still returns zero exact-name matches.
- The server output README archives this boundary with SHA256
  `2f3db44fee40b0e43b38272ee13a1c805cf0e337c9366dd9955be97e8e1714cd`.
  ID133 has no checkpoint or rollout and cannot resume or reuse its identity.
- The correction creates only `RUN_OUT`'s parent before the first progress write,
  preserving the empty `RUN_OUT` guard. A real shell execution regression uses a
  previously absent date parent and confirms the adjacent progress log is
  durable before an injected iteration-runner failure; the Slurm suite passes
  `20 passed`. Retry requires a new ID134/W&B/output and repeated launch gates.
