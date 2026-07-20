# 2026-07-20 DINO-supervised SFT1 grid reconstruction

## Human request and protocol

- Reconstruct from the latest completed DINO-supervised SFT1 ID18 epoch5/final result.
- SFT2 is incomplete: do not load, render, or compare WM-predicted state.
- Human selected only the directly DINO-supervised projected grid, not preprojection query hidden.
- Train the recon CFM on all strict train transitions and evaluate on disjoint strict val.
- Preserve matched noise, Euler50, CFG2 and diverse40/200-frame comparison against the proven Qwen ViT-token CFM.
- Human requested one 8GPU allocation: parallel cache first, then CFM/eval in the same allocation.

## Representation and implementation

- SFT1 source: `.../vagen_legacy_wm_k16_grid/2026-07-19/sft1/18_retry1_dino2l_grid4_k16_prefix_success7309_l1_ep5_b1_ga8_ws8_px602112/final/hf_merged`.
- State is the shared token-wise projector output `[16,1024]`, row-major 4x4, directly aligned to DINOv2-large pooled patch-grid targets during SFT1.
- Commit `80aaac5` added an explicit `dino_grid_state` cache projector, CFM semantics and no-WM evaluator.
- Commit `f0d4204` added strict source-model/cache lineage, row-major metadata validation, BF16 projector execution matching SFT1, CFM/cache eval fingerprint validation, and known error E0049.
- Server targeted tests: `14 passed, 1 Pillow deprecation warning`; compile and diff checks passed.

## Data and training

- Strict train: 3,217 training-split trajectories / 59,389 transitions, successes and failures.
- Strict val: 355 held-out validation trajectories / 6,054 transitions; validation only.
- CFM-only training: 128px TokenConditionedFlowUNet, 30 epochs/max55,680 steps, b32, lr3e-5→1e-5 at37,120, condition dropout0.15, shape-compatible initialization from proven Qwen ViT CFM.
- Frozen: SFT1 Qwen/query adapter/shared slot projector. DINO teacher and WM are absent.

## Running experiment

- Human first requested preempt `dgx-40`; agent failed to hold it before preparation, and another user's job took all GPUs. Human corrected this and specified then-IDLE `dgx-03`; error recorded as `E0049`.
- Holder Slurm `482045` is RUNNING on preempt `dgx-03`: one node / 8 GPUs / 128 CPUs / 768G / 8h.
- Same-allocation step `482045.1` launched the cache→projection→CFM→eval pipeline from clean server commit `f0d4204`.
- W&B identities reserved before the long cache phase:
  - ID45 CFM: `f9wj5gza`, `45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015`.
  - ID46 eval: `ek552cqe`, `46_sft1e5_dinogridcfm_qwenvit_diverse40`.
- Output root: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction`.
- Startup gate: eight torchrun ranks exist and all eight GPUs have initialized CUDA memory; no traceback/OOM/NCCL/NaN observed. Cache completion, CFM metrics and final visual result remain pending.
