# ID107 RL rollout-state cache GPU gate

## Purpose

Validate the corrected segment semantics beyond CPU tests: a real 20-step multimodal
greedy-planner rollout must persist sparse Qwen anchors plus the dense mixed sequence
of anchor/predicted WM states. Segment TD backward, detached episode ValueHead MC
backward, and the official multi-device DDP boundary must complete one finite optimizer
step and checkpoint without retaining all Qwen replay graphs.

This gate validates the current Nimloth objective. It is WM endpoint TD, Qwen action
distillation, and environment Monte Carlo ValueHead regression. It is explicitly not
VAGEN Bi-Level GAE.

## Launch contract

- Entry: `experiments/training/rl/run_vllm_online_ppo_slurm.sh` inside one Slurm hold.
- Config: `configs/training/rl/planner_greedy_h2_state_cache_t20_gate.yaml`.
- Data: verified VAGEN `base_train`, seeds 1 through 4, four episodes, at most 20
  environment steps each. No held-out quality claim is made.
- Initialization and frozen reference:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-24/sft2/46_dinogrid_k16_h4_untiedhead_fp32aux_v1cache_all3217_ep2_b1_ga8_ws8_px100352/epoch_001`.
- Planner: horizon 2, greedy, real CoT, maximum full response length 512 tokens.
- WM: DINO-grid `history_size=4`; no sampled training windows. Each planner segment
  retains at most four mixed real/predicted states as causal context.
- Trainable: Qwen language body, TemporalSpatialGridPredictor and ValueHead.
- Frozen: vision tower, GridStateProjector, EMA target encoder and DINO decoder.
- Losses: WM segment endpoint MSE, greedy planner action distillation and detached
  full-episode ValueHead MC loss; no token PPO/value, reference KL, DINO, SIGReg or
  ranking loss.
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

- Four valid trajectories with sparse finite Qwen anchors, `T + 1` finite mixed WM
  states and one greedy planner trace per segment.
- No state-path HF Qwen forward during training.
- Finite total/component losses and gradients on both ranks.
- No DDP/NCCL hang or collective mismatch.
- Exactly one optimizer/global step, changed trainable parameters, unchanged frozen
  parameters, committed fresh-consumption marker, and complete `latest`/`final`
  checkpoints.

## Corrected implementation status

- The former `TrajectoryWindow` replay workaround has been removed from the planner
  path. `TemporalDifferenceStep` follows consecutive Qwen anchors and replays every
  actually executed action in that segment; the terminal-short segment is retained.
- Every TD forward is followed immediately by backward. A final MC forward uses only
  detached mixed WM states and updates ValueHead. One optimizer step follows all TD/MC
  backward calls, so parameters remain fixed across the collected episode batch.
- The greedy WM actor owns behavior and Qwen receives action distillation only. A
  separate action-training provenance interface reserves future PPO semantics but
  rejects PPO until Qwen samples and owns the executed action.
- Pair-parallel training uses official DDP dynamic unused-parameter handling for the
  alternating TD/MC parameter subsets. No manual gradient averaging remains.
- Local targeted tests pass as recorded in `AI_branch_progress.md`. The replacement
  GPU gate has not run yet; ID107 and ID108 below remain historical failed attempts,
  not evidence for the corrected implementation.

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

## ID108 result: state-cache path passed, token replay OOM

- Commit `1c238a9` ran under preempt allocation `488111` on `dgx-11,dgx-22`,
  two H800s per node. `dgx-22` was explicitly the Ray and environment head. A
  real create/prompt/reset/close prewarm completed in 2.536 seconds under the
  300-second wall-time cap.
- All four `base_train` episodes completed 20 steps, for 80 transitions. An
  independent artifact check found, per trajectory, 20 actions, 21 observations
  and images, 21 finite cached states with shape `(16, 2048)`, 20 greedy
  horizon-2 planner traces, and a nonempty separately generated terminal CoT.
  Rewards were -1.9, -1.6, -1.6 and -1.9; success was 0/4. These are mechanics
  facts and not a policy-quality result.
- Frozen-reference replay completed all four trajectories and recorded policy
  fingerprint `f067b6f57461972dccd9c7cb8cbc94db1c0f842980480019b59bcc05478bac9a`.
  This proves that the real state cache and immutable token replay artifacts
  reached the training boundary.
- Both official DDP ranks validated two-GPU Qwen placement and entered the first
  policy replay forward. Both then OOMed on the second model-parallel GPU: rank 0
  had 42.75 MiB free and rank 1 had 30.12 MiB free while requesting 74 MiB;
  PyTorch had allocated 77.95 GiB on each affected GPU. The current replay loop
  retains every selected turn's Qwen activation graph until one monolithic
  backward, so long real histories accumulate rather than release per replay
  chunk. No NCCL hang or cross-device error occurred.
- `train_step_log.csv` has only its header. No backward, optimizer/global step,
  checkpoint or consumption commit completed. The pre-optimizer consumption
  transaction was aborted, leaving no consumption marker. The rollout and
  reference JSONL may be reused after fingerprint validation, but there is no
  training checkpoint to resume.
- W&B run `ui4uj84d` closed without training metrics. The controller exited and
  allocation `488111` was cancelled and released after 00:16:23.
- At the end of ID108, the proposed next step was microbatch/chunk replay while
  preserving token-GAE normalization. The later human clarification superseded that
  proposal: the corrected implementation above uses executed WM segments, action
  distillation and detached episode MC instead. Reducing semantic history, truncating
  recorded responses, manual cross-rank gradient averaging or increasing the
  AI2-THOR timeout remain invalid fixes.
