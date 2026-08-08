# 2026-08-07: ID146 PlannerPolicyHead PPO from corrected SFT2 epoch 1

## Purpose and evidence boundary

- Start a fresh RL run at global step 0 from the corrected SFT2 epoch-1
  checkpoint, train the H=1 PlannerPolicyHead with action-level PPO through
  global step 20, and then run the standard 120-episode held-out evaluation.
- This is the first formal training run for the PlannerPolicyHead objective.
  The completed ID145 4+4 gate is mechanics evidence only; ID143 and all other
  RL checkpoints, optimizer states, rollout batches and consumption records are
  excluded from initialization.
- Training-rollout success is an optimization diagnostic. Policy quality is
  measured only by the final held-out `base/common_sense` evaluation and must
  not be described as an improvement without a compatible baseline comparison.

## Initialization and supervision contract

- SFT2 checkpoint:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- The server preflight re-read non-empty Qwen, StateProjector, WM predictor,
  ValueHead and SFT2 training-state files. The root has no `rl_state.pt`, so the
  run begins without an RL optimizer or consumption state.
- PlannerPolicyHead is initialized independently with seed 42 using the exact
  SFT2 ValueHead architecture. No ValueHead parameter is copied into the policy
  head.
- For each training environment step, the H=1 policy planner samples the
  executed action from the PlannerPolicyHead categorical distribution and
  persists its real behavior log-probabilities. Four clipped-ratio PPO epochs
  supervise PlannerPolicyHead; the state-only ValueHead supplies the advantage
  baseline and is trained against MC returns. PPO gradients use differentiable
  full-prefix Qwen recomputation through the frozen StateProjector.
- Trainable: Qwen language body, WM predictor, ValueHead and PlannerPolicyHead.
  Frozen: Qwen vision, `lm_head`, StateProjector, DINO teacher and direct-Qwen
  token actor. No fixed or invented CoT is introduced.

## Data, schedule and evaluation

- Config:
  `configs/training/rl/planner_policy_h1_full_16rollout_8gpu_44_step20.yaml`.
- Entrypoints: batch-owned `experiments/training/rl/train_8gpu_44.slurm`, outer
  `run_vllm_online_ppo_full.sh`, and two-worker
  `run_vllm_online_ppo_parallel_slurm.sh`.
- Each of 20 iterations collects 16 fresh episodes, split evenly across
  `base_train` and `common_sense_train`, with at most 20 environment steps and
  bounded per-episode retries. The server assets contain 1,200 tasks each.
- After committed global step 20, external evaluation uses held-out `base` and
  `common_sense`, 60 tasks/seeds each and 120 total. These assets contain 60
  tasks each and never enter the optimizer.
- PlannerPolicyHead PPO: clip ratio 0.2, entropy coefficient 0.01, temperature
  1.0 and four epochs. Qwen backbone LR is `1e-6`; WM, ValueHead and
  PlannerPolicyHead LRs are `1e-4`. DINO-grid auxiliary weight is 0.5.

## Identity, output and recovery

- Code/config commit: `42e29ee25efae03ce89909f76fc2a954b7231782`.
- W&B project/run:
  `nimloth-rl/146_plannerpolicy_sft2ep1_rl16_eval20x120_policyh1_k16_dino05_ppo4_iter20_2n4r2g_2xtp4_preempt44`.
- Live W&B exact-name queries for the training and `-eval` names returned zero
  matches before submission.
- Formal output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/146_plannerpolicy_sft2ep1_rl16_eval20x120_policyh1_k16_dino05_ppo4_iter20_2n4r2g_2xtp4_preempt44`.
- Every committed iteration writes a complete checkpoint, atomically commits
  the matching fresh-rollout consumption record and preserves the next policy
  input. A preempted job may resume only from the latest complete committed
  checkpoint. It must never consume an uncommitted or stale rollout.

## Resources and completed launch validation

- Requested topology: `preempt`, two physical nodes with four H800 GPUs each,
  four synchronized two-GPU training ranks, and two TP4 rollout workers. The
  top-level controller is owned by the Slurm batch rather than the login SSH
  session.
- Expected elapsed time is roughly 5--7 hours (40--56 GPU-hours). The batch may
  be preempted and requeued; the allocation must be revalidated on every start.
- Resource snapshot before submission showed 32 free preempt GPUs, including
  one idle eight-GPU node and several nodes able to satisfy a flexible 4+4
  request. Submission must refresh this snapshot and must not pin stale nodes.
- Remote worktree was tracked-clean at the code/config commit (apart from
  ignored submodule untracked content). Shell syntax and the focused config,
  full-runner, Slurm, PlannerPolicyHead PPO, checkpoint and rollout suite passed:
  `133 passed, 1 expected warning in 114.10s`.
- Pending: commit this launch record, synchronize the server worktree, create
  and hash the fresh PlannerPolicyHead artifact, run exact login-node preflight,
  submit one flexible preempt 4+4 batch, and monitor allocation, renderer,
  rollout and the first finite synchronized PPO update.

## Submission status

- Runtime worktree is pinned and tracked-clean at
  `082e24f1a966781c2f57ff30a415e143a4c76c59`; VAGEN and LeWM are pinned to
  `192c35a91f3941b72d5e1272af6603ef7a7d93e0` and
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Fresh PlannerPolicyHead artifact:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/146_plannerpolicy_head_seed42_from_sft2ep1`.
  Its weight SHA256 is
  `f8577c9a25d8587208b08024a1b0fd7c13e60f012a4a545335dbd9aefc9e9a42`;
  metadata SHA256 is
  `ca919e00eafc330327a7a9fc207a157f634ee42aa3a1d0134c16807108a1f60c`.
- Exact-name W&B queries for the corrected `eval20x120` training and evaluation
  identities returned zero matches. The full Slurm `--test-only` contract was
  accepted before submission.
- Formal batch Job `510253` was submitted to `preempt` as one batch-owned 4+4
  allocation: two nodes, four GPUs/node, 64 CPUs/node, 48 GiB/node, eight-hour
  limit and `Requeue=1`; exclusions remain `dgx-32,dgx-37,dgx-51`.
- Immediately after submission, Slurm reported `PENDING(Priority)`, elapsed 0,
  `AllocTRES=(null)`, empty NodeList and unknown start time. No formal output,
  W&B run, renderer, rollout, PPO step, consumption record or RL checkpoint
  existed at that observation. Monitoring had to use the eventual real
  allocation rather than the earlier test-only estimate.

## Terminal status: cold isolated renderer preflight timed out

- A fresh Slurm query after handoff established that Job `510253` had already
  ended. It received `dgx-16,dgx-17`, four GPUs and 64 CPUs per node, at
  `2026-08-08T13:50:41+08:00` and ended at `13:53:24+08:00` after 2m43s.
  Slurm records the job as `FAILED 1:0`, derived exit `124:0`; renderer steps
  were `dgx-16 COMPLETED 0:0` and `dgx-17 FAILED 124:0`.
- Runtime commit, VAGEN/LeWM pins and W&B uniqueness gates passed. Both nodes
  used exact single-visible-GPU renderer contracts with fresh isolated homes,
  so each cold-downloaded and unpacked the 797 MiB AI2-THOR CloudRendering
  release. `dgx-16` emitted `AI2THOR_RENDER_OK` in 72.467 seconds with a
  255x255 frame and dynamic range 246. `dgx-17` reached 100% download, unpacked
  the 1.2 GiB release, initialized Vulkan on its H800 and created a render
  texture, but emitted no result before the 150-second outer timeout.
- The batch correctly failed closed before the outer controller. The formal
  output and adjacent iteration-progress log did not exist at failure time;
  there is no rollout, trajectory, fresh manifest, behavior fingerprint, W&B
  run, optimizer step, consumption state, RL checkpoint or held-out evaluation.
  The failure is therefore a cold renderer-preflight wall-clock failure, not a
  model, PPO or policy-quality result; `dgx-17` is not renderer-qualified by
  this attempt.
- The required terminal README was created at the formal output path and the
  RL experiment-group `progress.md` was updated. Their SHA256 values are
  `af17f2e0fd2a8025d21e51f7935ae4d1c0f6be8d6792c4a7eb4b9c85d3bd090f`
  and `3299918d47efb3af99ee95434ef2e31358c623f33475cca5ca16a4c55b51d279`.
  Post-terminal W&B API queries still returned zero exact-name matches for both
  the training and `-eval` identities.
- ID146 has no resumable state and its identity/output must not be reused. A
  retry requires a new experiment ID, W&B names and empty output, beginning
  again at RL global step 0 from the corrected SFT2 epoch-1 checkpoint plus a
  revalidated immutable PlannerPolicyHead seed artifact. Before resubmission,
  the cold-start renderer contract must be fixed and validated without
  exceeding the project-wide 300-second AI2-THOR limit; subsequent phases must
  still be checked independently rather than assumed healthy.
- This terminal boundary is recorded in existing progress/output documentation;
  no new durable memory was created because the next action requires a concrete
  launcher fix and human approval for another expensive formal attempt.
