# 2026-08-05 PPO ValueHead critic GPU mechanics gate

## Status

- Human confirmed the exact resource contract. ID126 was submitted and failed
  before backward; a corrected retry must use a new output identity.
- PPO critic implementation commit: `46be9368`.
- GPU gate harness commit: `64726911`.
- ID126 submitted commit: `5d02fb1e`.

## Purpose and evidence boundary

- Prove on real H800 hardware that the planner PPO ValueHead loss reaches the Qwen
  language body through the differentiable full-prefix state recomputation even
  though the environment action belongs to the planner, not Qwen action-token
  argmax selection.
- Prove the formal `world_size=2`, `gpus_per_rank=2` model-parallel DDP path can run
  all four critic PPO epochs, take synchronized AdamW steps, and keep replicas equal.
- This is a mechanics-only gate. It does not collect a new rollout, save a model
  checkpoint, run held-out evaluation, or provide policy-quality evidence. W&B is
  disabled.

## Immutable input contract

- Model/WM/ValueHead initialization:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
- Real behavior trajectories:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-04/125_sft2ep1_restart_rl16_eval10x120_greedyh1_k16_dino05_qwenwmvalue_iter60_ep16x20_1n4r2g_2xtp4_normal1x8/rollouts/iter_0001/trajectories.jsonl`.
- Freshness manifest beside that JSONL has policy fingerprint
  `794295edc9e015510f71971336af116a5383aabd022401dbd456f8b54f97c15e`,
  trajectory fingerprint
  `73fb1d272cb350e5197efe046b5604948e70277186895a5d64141d4923df9532`,
  and explicitly points to the same SFT2 epoch-1 model plus the same
  StateProjector, WM predictor, and ValueHead.
- The first two persisted trajectories are `rl_base_train_000001` and
  `rl_common_sense_train_000001`; each rank uses the first real transition. Their
  actual observation images and generated CoT remain present. No fixed or invented
  CoT is used.
- The manifest has already been consumed by historical ID125. This gate validates
  its immutable hashes and reads it only as behavior-checkpoint-matched mechanics
  input; it does not claim a new fresh rollout or alter its consumption record.

## Runtime and resource contract

- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3` (Python 3.12,
  PyTorch `2.8.0+cu128`).
- Entry:
  `experiments/training/rl/gpu_gate_ppo_value_critic.slurm`.
- Config: `configs/training/rl/planner_greedy_h1_full.yaml` with
  `ppo_clip_range=0.2`, `ppo_epochs=4`, Qwen LR `1e-6`, and ValueHead LR `1e-4`.
- Slurm: `preempt`, one node, four H800 GPUs, 32 CPUs, 128 GiB RAM, 20-minute
  walltime. Maximum allocation cost is 1.33 GPU-hours.
- Initial ID/output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-05/126_gpu_gate_ppo_value_critic_sft2ep1_realtraj1_1g_then_2r2g`.
  Live preflight confirmed this path is absent and the highest existing numeric RL
  output ID was 125. After the recorded ID126 failure, the corrected retry output is
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-05/127_gpu_gate_ppo_value_critic_sft2ep1_realtraj1_1g_then_2r2g`.
- Resume: none. A preemption or failure uses a new empty output identity; the gate
  never writes a training checkpoint.

## Two phases

1. `single_grad`: one GPU, real Qwen/StateProjector/ValueHead forward and critic-only
   backward on one persisted transition. It asserts nonzero Qwen final-norm and
   ValueHead gradients, `lm_head.grad is None`, and absent StateProjector/vision
   parameter gradients. It performs no optimizer step.
2. `ddp_step`: two ranks with two GPUs per rank, matching the formal planner topology.
   Each rank consumes a different real transition, freezes old action values before
   update, runs four PPO critic epochs, and takes production AdamW steps. It asserts
   finite metrics, nonzero Qwen/ValueHead gradients, a ValueHead parameter change,
   and exact Qwen/ValueHead witness equality across ranks after every step.

WM predictor, DINO teacher, Qwen vision, StateProjector, actor/token policy, and
`lm_head` are excluded from optimization in this focused gate. The test retains the
actual full-prefix Qwen -> StateProjector -> ValueHead graph requested by the user.

## ID126 end status

- Slurm Job `506808` ran on `preempt` node `dgx-16` with the requested four H800s,
  32 CPUs, and 20-minute limit. It started at `2026-08-05T18:46:30+08:00` and
  ended `FAILED (exit 1:0)` after 2 minutes 44 seconds.
- Exact runtime commit was `5d02fb1e0514a7556196397cd18c2573df8ae826`.
  The contract log confirms the intended SFT2 checkpoint, ID125 iteration-1
  trajectory and manifest, Python 3.12, and four visible H800s. Slurm stderr was
  empty.
- The run completed immutable policy/trajectory/planner fingerprint validation
  and loaded both Qwen checkpoint shards on GPU0. It then failed at the processor
  check because the gate called `freshness_validator.validate_processor(...)` on
  `FreshJSONLRolloutCollector`; the method belongs to `FreshRolloutManifest`.
- No state-prompt forward, backward, optimizer construction/step, DDP phase,
  result JSON, or checkpoint occurred. ID126 provides no gradient evidence and is
  not resumable. Its output must remain as the failed record; retry uses ID127.
