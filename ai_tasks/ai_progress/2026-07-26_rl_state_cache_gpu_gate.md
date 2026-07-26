# ID107 RL rollout-state cache GPU gate

## Purpose

Validate commit `82d4d0e` beyond CPU tests: a real 20-step multimodal greedy-planner
rollout must persist all `T + 1` Qwen latent states, training must consume the sampled
cache without the former state-sequence Qwen OOM, and the official multi-device DDP
boundary must complete one finite optimizer step and checkpoint.

This gate validates the current Nimloth objective. It remains environment Monte Carlo
return plus within-turn token GAE and is explicitly not VAGEN Bi-Level GAE.

## Launch contract

- Entry: `experiments/training/rl/run_vllm_online_ppo_slurm.sh` inside one Slurm hold.
- Config: `configs/training/rl/planner_greedy_h2_state_cache_t20_gate.yaml`.
- Data: verified VAGEN `base_train`, seeds 1 through 4, four episodes, at most 20
  environment steps each. No held-out quality claim is made.
- Initialization and frozen reference:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-24/sft2/46_dinogrid_k16_h4_untiedhead_fp32aux_v1cache_all3217_ep2_b1_ga8_ws8_px100352/epoch_001`.
- Planner: horizon 2, greedy, real CoT, maximum full response length 512 tokens.
- WM: DINO-grid `history_size=4`; one sampled batch of two continuous windows.
- Trainable: Qwen language body, TemporalSpatialGridPredictor, ValueHead and
  TokenValueHead.
- Frozen: vision tower, GridStateProjector, EMA target encoder and DINO decoder.
- Losses: WM, trajectory value, token PPO/value, greedy planner distillation, and
  frozen-reference low-variance KL weight 0.001; no DINO, SIGReg or ranking loss.
- Resources: normal partition, two nodes, two H800 GPUs per node; rollout vLLM TP4;
  training two ranks times two-GPU model parallel; official
  `DistributedDataParallel(device_ids=None)` around the complete RL step.
- W&B: project `nimloth-rl`, run name
  `107_smoke_greedyh2_k16_ep4x20_cache_mp2ddp2_vllmtp4_ws2`.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-07-26/107_smoke_greedyh2_k16_ep4x20_cache_mp2ddp2_vllmtp4_ws2`.
- Expected cost: approximately 1.5 to 3 hours, or 6 to 12 GPU-hours.
- Resume: the fresh rollout and reference-enriched artifacts can be reused by phase
  after fingerprint validation. Without a completed `latest/rl_state.pt`, the RL
  update itself is not checkpoint-resumable and any retry must keep the same immutable
  behavior artifact or use a new output identity as appropriate.

## Required evidence

- Four valid trajectories with `T + 1` finite cached states and greedy planner traces.
- No state-path HF Qwen forward during training.
- Finite total/component losses and gradients on both ranks.
- No DDP/NCCL hang or collective mismatch.
- Exactly one optimizer/global step, changed trainable parameters, unchanged frozen
  parameters, committed fresh-consumption marker, and complete `latest`/`final`
  checkpoints.

## ID107 result: environment cold-start failure

- Commit `66918a5` ran under preempt hold `488085` on `dgx-11,dgx-22`, two
  H800s per node. Ray exposed all four GPUs and imported the exact worktree on
  both nodes; the real checkpoint loaded successfully under vLLM TP4.
- The HTTP health endpoint became ready after 13 seconds, but navigation itself
  starts lazily. Episode 0's first `POST /environments` spent about 607 seconds
  constructing AI2-THOR/Unity and exceeded the then-current 600-second client
  timeout. The environment log recorded its first successful `Initialize` about
  seven seconds after the client had already timed out.
- Once one of four attempts was discarded, the strict four-trajectory gate could
  no longer pass. The controller was stopped and the allocation cancelled instead
  of waiting for three more possible timeouts. Ray actor and connection errors
  after shutdown are cleanup effects, not the original failure.
- The run produced zero valid trajectories and an empty JSONL. It has no manifest,
  reference replay, W&B training run, optimizer step, consumption marker, or
  checkpoint and is not resumable. Its remote README and adjacent launch contract
  retain the complete evidence; the output must not be reused.
- The HTTP health route is therefore not a sufficient navigation readiness gate.
  The retry will first complete a real create/reset/close prewarm before loading
  vLLM. Per the human's explicit limit, both this prewarm's total wall time and
  normal navigation requests are capped at 300 seconds; a timeout rejects the
  node and the experiment moves to another node rather than waiting longer.
