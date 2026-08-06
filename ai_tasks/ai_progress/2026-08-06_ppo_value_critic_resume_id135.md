# 2026-08-06: PPO ValueHead resume ID135 on 4+2+2 GPUs

## Status

- Prepared after ID134 ended at global step 15; not submitted. The final
  `sbatch --test-only` connection was closed by the SSH ProxyJump with
  `UNKNOWN port 65535` before Slurm executed.
- Human explicitly authorized starting training with `dgx-50` and two additional
  two-GPU nodes.
- Exact candidate runtime commit:
  `8f77fdc577b0061c68c93119555da2c9104f36d4`.

## Purpose and objective

- Continue the corrected planner critic experiment through the configured
  60-update horizon. PPO clipping supervises executed-action `Q(s_t,a_t)` while
  differentiable full-prefix recomputation propagates ValueHead gradients into
  the Qwen language body. Direct-Qwen actor PPO remains disabled.
- Objective metadata:
  `receding_horizon_decision_state_ppo_value_v1`, gamma 1, zero truncated
  bootstrap, ValueHead clip 0.2, four critic epochs. Only actual generated and
  persisted CoT is used.
- Trainable: Qwen language body, WM predictor, ValueHead. Frozen: Qwen vision,
  StateProjector, `lm_head`, DINO teacher, direct-Qwen actor/token policy.

## Corrected resume boundary

- Initialize model, WM modules and optimizer from ID134's immutable complete
  checkpoint:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8/train/policy_inputs/iter_0016`.
- The checkpoint represents iteration/global step 15; set
  `INITIAL_GLOBAL_STEP=15`, and continue at iteration 16 with per-dataset seeds
  121--128.
- ID134 iteration-16 rollout and any partial backward state are not reused. ID135
  requires a fresh strict 16/16 rollout, a new consumption identity, a new W&B
  name and an empty output directory.
- Resume remains iteration-atomic. Every completed update commits consumption
  only after a complete `train/latest/rl_state.pt`; a preemption, walltime or
  later failure resumes under another fresh identity from the newest committed
  policy snapshot.

## Collective-order correction

- ID134's exact 319-prefix diagnostic ruled out the 16,384 token gate; per-rank
  maxima were 14,441/16,005/16,178/14,268.
- Runtime commit `8f77fdc5` disables per-forward DDP buffer broadcasts for the
  explicitly synchronized planner modules and adds a distributed boundary after
  every transition backward. CPU regression asserts both wrapper parameters and
  the exact transition-by-epoch synchronization count.

## Data, config and evaluation

- Config:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_422.yaml`.
- Training split: actual VAGEN `base_train` and `common_sense_train` assets only;
  eight episodes per dataset per update, maximum 20 actions, same-identity
  collection retry at most three times.
- Greedy H1 planner, K16 grid states, DINO auxiliary weight 0.5, strict batch 16.
- External held-out evaluation remains the full disjoint `base` 60 plus
  `common_sense` 60 contract every ten completed updates.

## Identity and output

- W&B entity:
  `art2nd-hong-kong-university-of-science-and-technology`.
- Project: `nimloth-rl`.
- Candidate run ID/name:
  `135_ppo_value_syncfix_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_3n4r2g_1xtp4_preempt4_normal2x2`.
- Candidate output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/135_ppo_value_syncfix_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_3n4r2g_1xtp4_preempt4_normal2x2`.
- Submission is allowed only after confirming this exact W&B name is unused and
  both output and adjacent iteration-progress paths are absent.

## Runtime and requested resources

- Server worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Batch-owned entrypoint:
  `experiments/training/rl/train_8gpu_422.slurm`; iteration/evaluation runner:
  `experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh`.
- Heterogeneous component 0: `preempt/dgx-50`, one node, four GPUs, 64 CPUs,
  48 GiB. Component 1: `normal`, two nodes, two GPUs/32 CPUs/24 GiB each; exclude
  `dgx-32,dgx-37,dgx-51` and use two currently compatible nodes.
- Total: three nodes, eight GPUs, four two-GPU synchronized training ranks. Only
  the four-GPU node can host the single TP4 rollout/evaluation worker; all eight
  GPUs participate in training updates.
- Walltime: eight hours, maximum 64 GPU-hours for this segment. Based on ID134's
  measured iteration/evaluation times and the reduction from two TP4 workers to
  one, the remaining 45 updates are expected to require roughly 11--14 hours.
  This segment is therefore expected to stop at a committed intermediate
  checkpoint unless preempted earlier; it is not represented as a full-horizon
  completion request.
- Candidate port bases: Ray 7580, environment 9680, train rendezvous 32720.
- Lifecycle is owned by the Slurm batch. External monitoring is read-only.

## Required launch gates

1. Server worktree clean at exact commit `8f77fdc5`; pinned VAGEN/LeWM gitlinks
   and explicit Python path match the corrected runtime.
2. Focused server tests and the broader Qwen/rollout/config/Slurm/loop/fresh
   regression pass at the exact commit.
3. Resume checkpoint has complete model shards, planner artifacts and a readable
   `rl_state.pt` whose global step is exactly 15; config objective/checkpoint
   metadata match.
4. Actual dataset counts/split/scene disjointness are rechecked; config parses as
   nodes 3, world 4, two GPUs/rank, total 8, TP4, 16 episodes and attempts 3.
5. Exact W&B identity/output/progress are unused; ports are unique; shell syntax,
   login dry preflight and every heterogeneous component's `sbatch --test-only`
   request pass.
6. Immediately before submission re-query all jobs/resources. Require component
   0 to resolve to `dgx-50:4` and component 1 to two distinct compatible 2-GPU
   nodes. After allocation, verify the expanded node/GPU/rank mapping.
7. Monitor through real AI2-THOR prewarm, TP4 model warmup, strict 16-rollout
   merge and the first finite synchronized PPO update/checkpoint before declaring
   the resumed training healthy.

## Completed preflight and current blocker

- The server runtime worktree was clean at exact commit `8f77fdc5` under the
  tracked-files/submodule-untracked gate. VAGEN remained `192c35a9` and LeWM
  remained `8edfeb33`.
- Fixed-runtime focused regression passed `83 tests`; the full RL plus vLLM
  logits/policy boundary suite passed `208 tests` with only two known third-party
  or explicit-std warnings.
- Exact config parsed as iterations 60, strict episodes/batch 16, attempts 3,
  nodes 3, world 4, two GPUs/rank, total 8, TP4, actor disabled, ValueHead clip
  0.2/four epochs and external 120-episode validation every ten updates.
- Mmap checkpoint inspection confirmed iteration/global step 15, training world
  4, objective `receding_horizon_decision_state_ppo_value_v1`, matching planner
  and ValueHead metadata, replicated optimizer state, and all required model,
  processor, StateProjector, WM predictor, ValueHead and `rl_state.pt` files.
- Actual VAGEN assets contained 1200/1200 training tasks and 60/60 held-out
  tasks. The corresponding train/eval scene intersections were both empty.
- The exact W&B display name had zero matches. Candidate output and adjacent
  progress paths were absent, and the current user had no Slurm jobs.
- Last resource snapshot before the connection failure showed
  `preempt/dgx-50:4` free and compatible two-GPU capacity on normal nodes
  `dgx-10`, `dgx-14`, `dgx-21` and `dgx-30`; `dgx-37` remained excluded.
- The subsequent exact heterogeneous `sbatch --test-only` SSH session closed
  before reaching Slurm. Per server rules, no repeated reconnect was attempted.
  No job ID, allocation, W&B run, output, rollout or optimizer work exists for
  ID135. After VPN/ProxyJump recovery, re-run live jobs/resources, exact W&B and
  empty-output checks because those facts can drift, then execute test-only and
  the same formal submission.
