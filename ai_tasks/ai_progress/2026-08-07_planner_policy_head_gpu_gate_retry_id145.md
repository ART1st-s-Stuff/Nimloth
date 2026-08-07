# 2026-08-07: ID145 PlannerPolicyHead PPO 4+4 gate retry

## Purpose and identity

- Retry only the mechanics gate that ID144 did not reach. This is not a policy
  quality experiment and does not produce a resumable checkpoint.
- W&B project/run identity: `nimloth-rl` /
  `145_smoke_plannerpolicy_gpu_gate_retry_h1_base8_2n4g`; W&B is disabled.
- New output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/145_smoke_plannerpolicy_gpu_gate_retry_h1_base8_2n4g`.

## Input and supervision contract

- Reuse only ID144's immutable, unconsumed fresh rollout: eight `base_train`
  trajectories, seeds 185--192, 144 transitions. The gate revalidates behavior
  model, StateProjector, ValueHead, WM and PlannerPolicyHead fingerprints.
- ID143 committed step-20 initializes Qwen, WM, StateProjector and ValueHead;
  the same independently seeded PlannerPolicyHead artifact used by ID144 is
  retained.
- Select the four longest distinct real final prefixes from the batch and only
  require a nonempty prefix. The observed maximum is 4,136 tokens. This retry
  is a policy-gradient mechanics test, not a synthetic long-context memory
  stress test.
- Trainable during the gate: Qwen language body, ValueHead and
  PlannerPolicyHead. Frozen: Qwen vision, lm_head, StateProjector and WM.
- Required evidence: single-rank nonzero gradients, then four synchronized DDP
  ranks with two GPUs/rank for four PPO epochs, nonzero ValueHead and
  PlannerPolicyHead parameter deltas, and absent frozen-module gradients.

## Resource and recovery contract

- Reuse the still-running preempt hold Job `509867`, actual allocation
  `dgx-01 + dgx-16`, four H800 GPUs per node. Launch only through `srun`.
- ID144 already passed renderer preflight on both nodes; ID145 records this as
  same-allocation reused evidence and performs no environment rollout.
- The output must be absent before launch. A failure is terminal for this
  identity; there is no checkpoint resume.
