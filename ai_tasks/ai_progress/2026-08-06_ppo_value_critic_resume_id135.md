# 2026-08-06: PPO ValueHead resume ID135 on 4+4 GPUs

## Status

- The 1x8 ID135 attempt failed before global merge and was cancelled at the
  affected Slurm steps. It is not resumable and its output/W&B identity cannot
  be reused.
- Original 4+4 Job `508170` was cancelled before the 1x8 attempt. Hold Job
  `508268` remained running only long enough to verify cleanup and archive the
  failed output, then must be released.
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
  `135_ppo_value_syncfix_dgx32qual_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
- Candidate output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/135_ppo_value_syncfix_dgx32qual_resume15_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8`.
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

1. Server worktree clean at exact commit `d6197e84`; pinned VAGEN/LeWM gitlinks
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

- The server runtime worktree is clean at exact commit `d6197e84`, containing
  the collective fix from `8f77fdc5` plus the 4+4 retry config and regression.
  VAGEN remains `192c35a9` and LeWM remains `8edfeb33`.
- Collective-fix focused regression passed `83 tests`; the full RL plus vLLM
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
  ID135 before formal submission. The exact scheduler-flexible
  `sbatch --test-only` passed with an estimated `2026-08-08T12:56:41` start on
  `dgx-[09,21]`; this estimate did not allocate resources.
- Formal Job `508170` was submitted at `2026-08-06T15:37:51+08:00` and is
  `PENDING(Priority)`, elapsed zero. `scontrol -dd` confirms normal QOS, two
  nodes, two tasks, 64 CPUs/task, four GPUs/node, 48 GiB/node, eight hours and
  exclusions `dgx-[32,37,51]`. It has no allocation, output, W&B run, rollout or
  optimizer evidence yet. After allocation, verify actual nodes and monitor
  through prewarm, two TP4 engines, strict merge and the first finite update.

## dgx-32 requalification and possible 1x8 replacement

- A later live query found `normal/dgx-32` fully `IDLE` with 8/8 GPUs, 224/224
  CPUs and about 1892 GiB free memory. Job `508170` cannot use it because its 4+4
  contract excludes `dgx-32` and requires two physical nodes.
- The exclusion is evidence-based: ID116's only failed shard exceeded the
  300-second AI2-THOR navigation prewarm on `dgx-32`; ID117 moved the renderer
  within both TP4 groups and both groups still stopped before the first
  navigation observation while six non-`dgx-32` renderers passed in 3--4 seconds.
- Before changing the formal topology, submit exactly one batch-owned hold for
  `normal/dgx-32:8`, 128 CPUs, 96 GiB and eight hours. Keep Job `508170` pending
  until the replacement allocation and render qualification are proven.
- In the hold, run the real `CloudRendering` direct render probe on every
  allocated GPU with the established 60-second AI2-THOR internal timeouts and
  150-second shell limit. A 1x8 formal launch requires at minimum both renderer
  slots used by the two TP4 groups (allocated ordinals 0 and 4) to emit
  `AI2THOR_RENDER_OK`; probing all eight avoids hiding another broken mapping.
- The probe does not train or freeze any module, consume a dataset/checkpoint,
  create a W&B run or write into the formal ID135 output. Its artifacts use a
  separate requalification directory. If it fails, release the hold and retain
  Job `508170`; if it passes, recheck a new 1x8 W&B/output identity, cancel the
  superseded pending 4+4 job, and launch the existing
  `planner_greedy_h1_full_16rollout_8gpu_1x8.yaml` contract in the same hold via
  `srun`.

## dgx-32 qualification result and 1x8 launch gate

- Hold Job `508268` started at `2026-08-06T16:37:38+08:00` on `dgx-32` with
  8 GPUs, 128 CPUs, 96 GiB and an eight-hour limit. Job `508170` was deliberately
  kept pending during qualification.
- All eight allocated ordinals were probed in parallel with isolated AI2-THOR
  homes. Ordinals 0 and 4 emitted `AI2THOR_RENDER_OK` in 37.132 and 37.334
  seconds with dynamic range 246; ordinals 1/2/3/5/6/7 reached the 150-second
  shell timeout. Artifacts are under
  `outputs/experiments/training/rl/preflight/2026-08-06/135_dgx32_render_requalification_508268`.
- This is a layout-specific qualification, not a claim that the whole node is
  render-healthy. The 1x8 parallel runner forms TP4 groups 0--3 and 4--7 and
  starts each environment on the group's first GPU, exactly ordinals 0 and 4;
  the other six GPUs serve policy inference/training and need not render.
- Replacement config/entrypoint:
  `configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml` and
  `experiments/training/rl/train_8gpu_1x8.slurm`. Exact config parsed as nodes 1,
  world 4, two GPUs/rank, total 8, TP4, strict batch 16 and attempts 3.
- The new 1x8 W&B display name above had zero matches, its output and adjacent
  progress path were absent, the immutable step-15 checkpoint files remained
  nonempty, the server worktree was clean at `d6197e84`, and no probe/Ray/vLLM
  GPU processes remained. Immediately before launch, cancel only the superseded
  pending Job `508170`, then run the batch-owned 1x8 controller as a Slurm step
  in hold `508268` and monitor through prewarm, two TP4 engines, strict merge and
  the first finite update/checkpoint.

## 1x8 terminal result

- The launch began iteration 16 at `2026-08-06T16:47:00+08:00`. Shard 0's
  formal environment on physical GPU 0 passed real navigation prewarm in 4.924
  seconds, initialized one TP4 EngineCore, loaded both model shards, completed
  vLLM warmup and wrote seven local trajectories.
- Shard 1's formal environment used the actual runner contract
  `CUDA_VISIBLE_DEVICES=4` with relative `navigation.devices=[0]`. It never
  returned the first observation, exceeded the 300-second prewarm gate and then
  hung during environment cleanup. It initialized no TP4 engine and wrote zero
  trajectories.
- This invalidates the earlier qualification inference. The passing direct
  probe used all eight GPUs visible with `gpu_device=4`, which is not equivalent
  to the formal single-visible mapping. Known error E0085 records the required
  exact-visibility probe contract.
- Strict 16-shard merge was impossible. Slurm steps `508268.2` and `.3` were
  cancelled after the hard gate; all Unity, VAGEN, Ray, vLLM and GPU processes
  were then absent. The hold itself was not a training success.
- Output contains seven local shard-0 trajectories and no shard-1 trajectory,
  global fresh manifest, consumption, optimizer update, train log or checkpoint.
  The W&B display name still has zero matches. ID135 is failed/non-resumable;
  recovery must use a new ID, W&B name and empty output from the unchanged
  immutable ID134 step-15 checkpoint.
- Server output README and the RL experiment-group `progress.md` were updated;
  their SHA256 values are `c10129a1...cd1286` and `9f20178c...6e53eb`.
  Hold Job `508268` was released at `2026-08-06T17:00:00+08:00` after 22m22s;
  no ID135 or fallback Slurm job remains queued or running.

## Post-ID135 compatible 1x8 resource hold

- The human requested queueing one new compatible 1x8 node. Resource-only Job
  `508346` was submitted at `2026-08-06T17:27:26+08:00` to `normal` for one
  node, eight GPUs, 128 CPUs, 96 GiB and eight hours, excluding
  `dgx-32,dgx-37,dgx-51`. It runs only the batch-owned hold loop and does not
  load a model, read a checkpoint, create a W&B run or start formal training.
- Immediately before submission, the server worktree tracked content was clean
  at exact runtime commit `d6197e84`; VAGEN and LeWM remained pinned at
  `192c35a9` and `8edfeb33`. LeWM contained only an untracked runtime
  `__pycache__/`, which was preserved and not treated as a source modification.
- `sbatch --test-only` gave a non-binding `2026-08-07T22:22:26+08:00` estimate
  on `dgx-52`, but the real job obtained `normal/dgx-52:8` almost immediately
  and entered `RUNNING` at `2026-08-06T17:27:30+08:00`. The scheduler contract
  reports `AllocTRES=cpu=128,node=1,gres/gpu=8` and
  `MinMemoryNode=96G`; the hold log confirms `nodes=dgx-52`.
- ID135 remains terminal and non-resumable. Before any formal `srun`, recovery
  still requires a new experiment ID, unused W&B name and empty output rooted
  at the immutable ID134 step-15 checkpoint. The allocated node must first pass
  the exact single-visible-GPU AI2-THOR preflight required by known error E0085.
