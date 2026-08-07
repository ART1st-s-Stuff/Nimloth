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

## Terminal result

- Commit `203a16aad99f62644130997a2485d04b6a8696c3` ran on hold Job `509936`,
  actual `dgx-01 + dgx-16`, world size four with two GPUs per rank.
- Single-rank evidence passed on a 4,136-token real prefix: Qwen language-body
  gradient max `0.0071105957`, PlannerPolicyHead `0.1091669872`, ValueHead
  `0.1236376762`; lm_head, vision and StateProjector gradients were absent.
- All four distributed ranks completed four PPO epochs on four distinct real
  prefixes of 4,136/4,076/4,076/4,075 tokens. Their common maxima were Qwen
  gradient `0.0023193359`, ValueHead gradient `0.0550164022`, ValueHead
  parameter delta `0.0004001334` and PlannerPolicyHead parameter delta
  `0.0003984426`. Gradient and parameter replica differences were all zero.
- Qwen witness parameter delta was zero at BF16 and learning rate `1e-6`.
  Therefore the result proves PPO gradient reachability into the language body,
  but does not claim a measurable Qwen parameter change at that witness.
- Both node-level `srun` steps returned `1:0` after the four passed JSON files
  were written. Strict post-validation of every required field passed; ID145 is
  completed with this controller warning. There was no traceback, OOM, NCCL
  error or assertion failure. The output produces no checkpoint and is not
  policy-quality evidence.
- Canonical summary:
  `145_smoke_plannerpolicy_gpu_gate_retry_h1_base8_2n4g/gate_summary.log`,
  SHA256 `ad9e5e3b4491bb4c99bfd07b31934f74cffadffffa5750355eb8fd8dcc818775`.
  Hold Job `509936` was cancelled after evidence finalization to release all
  eight GPUs.
