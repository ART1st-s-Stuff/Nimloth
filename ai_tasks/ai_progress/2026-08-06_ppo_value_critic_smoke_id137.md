# 2026-08-06: bounded PPO ValueHead failure-unmask smoke ID137

## Status

- Authorized and prepared for launch in the existing resource-only Slurm hold
  Job `508346` on `normal/dgx-52:8`.
- Exact runtime commit:
  `c3215592b7b3084d4fc2b9b33be90f0259974eed`.
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
