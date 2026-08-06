# 2026-08-06: bounded PPO ValueHead failure-unmask smoke ID137

## Status

- Terminal failure before the first optimizer step. Formal step `508346.14`
  ran inside resource-only hold Job `508346` on `normal/dgx-52:8` and exposed
  rank-local Qwen activation OOMs that ID136's exception cleanup had masked.
- ID137 and its fresh rollout are non-resumable and forbidden from reuse. The
  only valid recovery boundary remains ID134 global step 15.
- Exact runtime commit:
  `c321559271df0c3f01230e99401bdf6593aabc0e`.
- This is a bounded production-shaped diagnostic, not a policy-quality or
  held-out-evaluation claim.

## Purpose and objective

- Re-run exactly one fresh iteration after removing the rank-local exception
  masking path. A failing rank now prints its original exception and traceback,
  skips the distributed abort broadcast and leaves consumption fail-closed as
  `in_progress`; failed distributed cleanup destroys the process group without
  an additional barrier.
- PPO clipping supervises executed-action `Q(s_t,a_t)` and differentiable
  full-prefix recomputation propagates the critic gradient into the Qwen
  language body. Direct-Qwen actor PPO remains disabled.
- Objective metadata:
  `receding_horizon_decision_state_ppo_value_v1`, gamma 1, zero truncated
  bootstrap, ValueHead clip 0.2 and four critic epochs. Only model-generated and
  persisted CoT is consumed.
- Trainable: Qwen language body, WM predictor and ValueHead. Frozen: Qwen
  vision, StateProjector, `lm_head`, DINO teacher and direct-Qwen actor/token
  policy.

## Resume, data and evaluation

- Immutable initialization and optimizer resume checkpoint:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8/train/policy_inputs/iter_0016`.
- The checkpoint records iteration/global step 15. The smoke config has
  `rl.iterations: 16`, so ID137 runs only iteration 16 and then stops.
- ID137 generates a fresh strict batch: actual VAGEN `base_train` and
  `common_sense_train`, eight episodes per dataset, per-dataset seeds 121--128,
  at most 20 actions and at most three attempts. No ID136 rollout is reused.
- No held-out evaluation is due at iteration 16. The continuing formal contract
  remains the fixed disjoint 120-episode `base/common_sense` evaluation every
  ten committed updates.
- Iteration consumption is committed only after a complete optimizer state and
  policy checkpoint. Failure before the step leaves the new identity
  non-resumable and requires another fresh identity and rollout.

## Identity and output

- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- Run ID/name:
  `137_smoke_exception_unmask_dgx52_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Exclusive output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/137_smoke_exception_unmask_dgx52_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Pre-launch checks must reconfirm that the exact W&B name is unused and both
  the output and adjacent progress paths are absent.

## Runtime and resource bound

- Server worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Config:
  `configs/training/rl/planner_greedy_h1_smoke_iter16_16rollout_8gpu_1x8.yaml`.
  Entrypoint: `experiments/training/rl/train_8gpu_1x8.slurm`, launched as an
  attached step inside hold `508346`.
- Allocation: `normal/dgx-52`, eight GPUs, 128 CPUs and 96 GiB. Expected wall
  time is 20--30 minutes; the diagnostic is capped operationally at 45 minutes,
  or at most 6 GPU-hours. The outer hold is preserved after the step.
- Port bases: Ray 7610, environment 9710 and training rendezvous 32810.
- NCCL diagnostics enable a 200000-event flight-recorder buffer, timeout dump
  and detailed distributed diagnostics.

## Required launch and terminal gates

- Before launch: exact clean runtime commit and pinned submodules, checkpoint
  metadata/files, parsed config, train/eval split integrity, empty output and
  unused W&B identity, free selected ports, no residual compute processes and
  concurrent exact-visibility AI2-THOR render probes for physical GPUs 0 and 4.
- During launch: both formal navigation prewarms, two distinct TP4 EngineCore
  model/KV-cache warmups, strict 16-trajectory merge and then either the
  original rank-local failure or a finite synchronized four-epoch PPO update.
- Success means exactly one committed iteration-16 optimizer/checkpoint result.
  Failure before the optimizer must produce no checkpoint and must retain the
  original traceback. Either terminal outcome must run the experiment-end hook
  and record cleanup evidence; neither outcome alone establishes policy quality.

## Terminal result

- Concurrent exact-visibility physical GPU 0/4 probes passed in 9.685/9.259
  seconds with dynamic range 246 after isolated homes reused the verified
  shared AI2-THOR release cache. The first probe attempt had timed out only
  while downloading an uncached 797 MiB build and produced no GPU-0 render
  conclusion. Both formal navigation prewarms then passed in 3.457/3.406
  seconds, and two distinct TP4 EngineCore workers completed model load,
  57.81 GiB/GPU KV-cache allocation and warmup.
- Strict merge completed all 16 fresh train-split trajectories / 319
  transitions for per-dataset seeds 121--128. Training-batch success was 0/16,
  average reward -0.425 and average length 19.9375; these are rollout
  observations, not held-out model-quality evidence.
- Before optimizer, rank 1 OOMed in Qwen language MLP forward on physical GPU 3
  while requesting 338 MiB with only 316.06 MiB free. Its process used 78.86
  GiB total and PyTorch had allocated 77.15 GiB. Rank 2 independently OOMed on
  physical GPU 5 while requesting 64 MiB with 32.06 MiB free; its process used
  79.14 GiB total and PyTorch had allocated 77.38 GiB. Rank 0 then detected the
  failed peers at the monitored barrier and torchrun terminated the remaining
  ranks.
- The new failure records prove that ID136's one-element `BROADCAST` came from
  the exception-cleanup path and was secondary to rank-local activation OOM.
  They do not show a ValueHead/DDP collective-order bug in the successful
  execution path.
- Formal step `508346.14` ran from `19:10:36` to `19:19:41+08:00` and failed
  with exit `1:0`; rollout step `508346.15` completed normally. There is no
  optimizer step, metric row, `rl_state.pt`, policy checkpoint or held-out
  evaluation. The train CSV contains only its header and merged-manifest
  consumption remains fail-closed as `in_progress` from global step 15.
- W&B run `tc2o89q8` is finalized as `failed` with terminal status
  `failed_rank_local_cuda_oom_before_optimizer`. Server output README,
  run-level progress and RL-group progress SHA256 values are respectively
  `1772aa0e...8740`, `5a9ec31f...b3e` and `30698d1c...ade8`.
- Post-failure checks found no GPU compute process, Unity/VAGEN, vLLM, Ray,
  training process or selected-port listener on dgx-52. Hold Job `508346`
  remains running for a later memory-safe diagnostic; this failure does not
  authorize reusing ID137 or restarting full-scale training unchanged.

## VAGEN/verl reuse boundary

- The rollout/environment side already reuses the pinned VAGEN navigation
  assets and protocol. Its strict fresh-manifest and real-CoT contract should
  remain unchanged.
- verl provides useful memory mechanisms: FSDP-sharded Qwen construction,
  gradient checkpointing, parameter/optimizer offload, token-budgeted dynamic
  micro-batches and optional Ulysses sequence parallelism. Those mechanisms are
  relevant to the activation OOM exposed by ID137.
- The stock VAGEN/verl critic is not a drop-in replacement for this objective.
  It builds `Qwen2_5_VLForTokenClassification`, predicts response-token values
  and applies the standard token-value PPO loss. Nimloth instead recomputes the
  real decision-state prefix, selects executed-action `Q(s_t,a_t)`, jointly
  trains WM/DINO/ValueHead, keeps actor PPO disabled and must retain the current
  atomic manifest/checkpoint lineage.
- Runtime provenance also needs correction before reuse: pinned VAGEN commit
  `192c35a9` records verl gitlink `65316156`, but the current server venv imports
  editable verl 0.6.1 from the main server checkout at commit `138a1d17` rather
  than the runtime worktree gitlink. A new backend must pin and verify one exact
  verl source before any GPU claim.
- Recommendation: reuse verl's FSDP/sharding and micro-batch components behind a
  Nimloth-specific critic worker/adapter; do not switch to the unmodified
  VAGEN `RayPPOTrainer` or report its standard token critic as the requested
  executed-action planner PPO. First gate the custom backend on the same 319
  transition shape with synthetic/non-consumable inputs, then run a new-ID
  fresh-rollout production smoke.

## Follow-up root fix (local, GPU gate pending)

- The formal environment's Transformers 4.55.4 source was checked directly.
  `PreTrainedModel.from_pretrained()` ends by calling `model.eval()`, while the
  Qwen2.5-VL text forward enters its checkpoint function only under
  `self.gradient_checkpointing and self.training`.
- The RL loader called `gradient_checkpointing_enable()` but the planner trainer
  never restored the loaded Qwen to train mode. Thus ID137's configured
  checkpointing was not effective; the observed 77 GiB activation allocation is
  explained without changing the executed-action `Q(s_t,a_t)` objective.
- The local fix puts the differentiable planner Qwen in train mode before DDP
  wrapping, counts runtime checkpoint-enabled modules, and fails closed if the
  requested mode is ineffective. The independent vLLM behavior-policy process
  is unchanged.
- This is source and interface evidence only. A fresh, non-consumable,
  production-shaped GPU memory gate must still complete a real long-prefix
  forward/backward/optimizer step before training is resumed. ID137 and its
  in-progress consumption record remain forbidden from reuse.
