# 2026-08-06: PPO ValueHead resume ID135 on 4+4 GPUs

## Status

- Prepared after ID134 ended at global step 15; not submitted under the stale
  4+2+2 contract. The earlier `sbatch --test-only` connection was closed by the
  SSH ProxyJump before Slurm executed.
- Human authorized up to eight GPUs. An initial 2026-08-06 resource query found
  `dgx-50` unavailable and five free GPUs on each of normal `dgx-14` and
  `dgx-31`, motivating a 4+4 topology. The final pre-submit refresh found those
  GPUs already consumed and no node with four immediately free GPUs, so the
  request remains 4+4 but does not pin stale node names.
- Exact candidate runtime commit:
  `d6197e843fcbfbfe59185b0280c1e6c1acccbfdc`.

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
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_44.yaml`.
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
  `135_ppo_value_syncfix_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_2n4r2g_2xtp4_normal2x4`.
- Candidate output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/135_ppo_value_syncfix_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_2n4r2g_2xtp4_normal2x4`.
- Submission is allowed only after confirming this exact W&B name is unused and
  both output and adjacent iteration-progress paths are absent.

## Runtime and requested resources

- Server worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Batch-owned entrypoint:
  `experiments/training/rl/train_8gpu_44.slurm`; iteration/evaluation runner:
  `experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh`.
- Homogeneous allocation: `normal`, any two compatible nodes, four GPUs,
  64 CPUs and 48 GiB per node; exclude `dgx-32,dgx-37,dgx-51`. The request is
  expected to remain `PENDING(Resources)` until two such nodes are available.
- Total: two nodes, eight GPUs, four two-GPU synchronized training ranks. Each
  node hosts one TP4 rollout/evaluation worker; all eight GPUs participate in
  training updates.
- Walltime: eight hours, maximum 64 GPU-hours for this segment. Based on ID134's
  measured iteration/evaluation times with two TP4 workers, the remaining 45
  updates are expected to require roughly 8--10 hours. This segment may stop at
  a committed intermediate checkpoint; it is not represented as guaranteed
  full-horizon completion.
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
   nodes 2, world 4, two GPUs/rank, total 8, TP4, 16 episodes and attempts 3.
5. Exact W&B identity/output/progress are unused; ports are unique; shell syntax,
   login dry preflight and the homogeneous `sbatch --test-only` request pass.
6. Immediately before submission re-query all jobs/resources. Do not pin node
   names from a stale snapshot. After allocation, require two distinct compatible
   nodes with exactly four GPUs each and verify the expanded node/GPU/rank map.
7. Monitor through real AI2-THOR prewarm, TP4 model warmup, strict 16-rollout
   merge and the first finite synchronized PPO update/checkpoint before declaring
   the resumed training healthy.

## Completed preflight and live resource change

- The server runtime worktree was clean at exact commit `8f77fdc5` under the
  tracked-files/submodule-untracked gate. VAGEN remained `192c35a9` and LeWM
  remained `8edfeb33`.
- Fixed-runtime focused regression passed `83 tests`; the full RL plus vLLM
  logits/policy boundary suite passed `208 tests` with only two known third-party
  or explicit-std warnings.
- The replacement 4+4 config parsed at exact commit `d6197e84` as iterations
  60, strict batch 16, attempts 3, nodes 2, world 4, two GPUs/rank, total 8,
  TP4, actor disabled, ValueHead clip 0.2/four epochs and external 120-episode
  validation every ten updates. Config and Slurm regression passed `64 tests`.
- Mmap checkpoint inspection confirmed iteration/global step 15, training world
  4, objective `receding_horizon_decision_state_ppo_value_v1`, matching planner
  and ValueHead metadata, replicated optimizer state, and all required model,
  processor, StateProjector, WM predictor, ValueHead and `rl_state.pt` files.
- Actual VAGEN assets contained 1200/1200 training tasks and 60/60 held-out
  tasks. The corresponding train/eval scene intersections were both empty.
- The new 4+4 W&B display name had zero matches; its output and adjacent
  progress paths were absent. All required files in the immutable ID134
  checkpoint remained nonempty.
- The first refreshed resource query reported `normal/dgx-14:5` and `dgx-31:5`,
  while `preempt/dgx-50` was unavailable. The final pre-submit refresh reported
  only 1/1/2 free GPUs on normal `dgx-29/30/37` and 2/2 on preempt
  `dgx-16/22`; there is no immediately schedulable TP4 node. The request must be
  scheduler-flexible and is expected to pend rather than use an incompatible
  topology or the excluded `dgx-37`.
- No job ID, allocation, W&B run, output, rollout or optimizer work exists for
  ID135 yet. The corrected commit is synchronized and all gates except the
  final live resource refresh and `sbatch --test-only` have passed.
