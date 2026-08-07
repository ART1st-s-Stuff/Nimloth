# 2026-08-07: ID144 PlannerPolicyHead PPO real-GPU mechanics gate

## Identity and purpose

- Purpose: collect one behavior-checkpoint-matched H=1 PlannerPolicyHead batch,
  then prove on real H800 hardware that PPO reaches the Qwen language body and
  PlannerPolicyHead while preserving frozen-module and replica-sync boundaries.
- This is a mechanics smoke, not a policy-quality experiment and not a resumable
  training run.
- W&B project/run identity: `nimloth-rl` /
  `144_smoke_plannerpolicy_gpu_gate_h1_base8_2n4g`; W&B mode is disabled for
  this gate, so no online run is expected.
- Planned output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/144_smoke_plannerpolicy_gpu_gate_h1_base8_2n4g`.
- PlannerPolicyHead initialization artifact:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/144_smoke_plannerpolicy_head_seed42_from_id143_step20`.

## Launch contract

- Code/config commit: `4fbc0ce3` on `dev`; the server worktree was synchronized
  through Git and verified tracked-clean at that revision.
- Both allocated nodes must pass a fresh AI2-THOR rendered-frame preflight
  before rollout starts.
- Entrypoints: `initialize_planner_policy_head.py`, then
  `run_planner_policy_gpu_gate_4x4_on_hold.sh`; the latter runs the parallel
  rollout and both gate phases through `srun` inside one held allocation.
- Config: `configs/training/rl/planner_policy_h1_gpu_gate_4x4.yaml`.
- Initialization: ID143 committed global-step-20 `train/final` supplies Qwen,
  WM predictor, frozen StateProjector and ValueHead. PlannerPolicyHead uses
  deterministic seed 42 fresh initialization with the ValueHead architecture
  only; critic parameters are not copied.
- Data: eight new `base_train` trajectories, split `train`, seeds 185--192,
  maximum 20 environment steps and configured bounded retries. No eval split is
  consumed.
- Gate trainable modules: Qwen language body, ValueHead and PlannerPolicyHead.
  Frozen: Qwen vision, lm_head, StateProjector and WM predictor. Direct-Qwen
  token PPO/reference KL remain disabled.
- Gate objective: one `single_grad` backward followed by four DDP ranks × two
  GPUs/rank across two nodes for four PlannerPolicyHead PPO epochs. Required
  evidence includes nonzero Qwen/ValueHead/PolicyHead gradients, no
  lm_head/vision/StateProjector gradients, nonzero ValueHead/PolicyHead
  parameter deltas and synchronized rank replicas.
- Output/recovery: the output and initialization artifact must be absent before
  launch. The gate writes JSON/log evidence but no training checkpoint and has
  no resume path; a failed retry must use a new output identity.
- Resources: preempt, two nodes × four H800 GPUs/node, 64 CPUs/node, 128 GiB
  RAM/node, 45-minute hold. Expected runtime is 15--30 minutes, at most 6
  GPU-hours.

## Preflight status

- CPU focused suite on `82cda7f3`: `128 passed, 1 warning`.
- Resource snapshot after VPN reconnect: `dgx-01/16/22/42` had 6/7/7/5 free
  preempt GPUs, respectively, so a flexible two-node 4+4 request fit at that
  snapshot without a fixed node constraint. Resources must be refreshed at
  submission time.
- Refreshed preempt snapshot immediately before submission showed no free GPU
  nodes; ID144 must therefore enter the queue as one two-node 4+4 hold.
- Project memory search found no reusable checkpoint/gate memory; this contract
  is derived from current code, current progress evidence and live server state.
- Pending: commit the 4+4 config/launcher/metadata change, synchronize the
  server worktree, verify ID143 files and output absence, initialize the head,
  submit/attach the hold, and monitor rollout plus both gate phases.
