# 2026-07-23 SFT2 DINO grid refactor and retraining

## Goal

- Train a second SFT2 variant with 16 Qwen query slots aligned to the frozen
  DINOv2 next-image 4x4 grid target.
- Preserve current SFT2 semantics: H=4 causal history, one current-step CE,
  detached older history states, and global-batch SIGReg with gradients only
  through the online next state.
- Keep the refactor-before DINO objective authoritative: latent next-state MSE,
  decoded DINO grid MSE at weight 0.5, SIGReg at weight 0.1, and value loss.

## Authoritative artifacts

- SFT1 checkpoint:
  `.../18_retry1_dino2l_grid4_k16_prefix_success7309_l1_ep5_b1_ga8_ws8_px602112/final/hf_merged`
- DINO target cache:
  `.../k16_all3217_px100352_bf16_dino4x4_f32_b8659fe`
- Historical SFT2 checkpoint:
  `.../33_cached_lewmgrid_dino05_sig01_ema099_all3217_ep10_b2_ga4_ws8_px100352/latest`

## Plan

- [x] Restore the exact 16-token DINO cache identity and SFT1 slot-projector contract.
- [x] Add grid-specific WM modules without putting DINO logic in the generic SFT2 algorithm.
- [x] Generalize only the shared state/history interfaces needed for vector or grid state.
- [x] Add fail-closed warm-start and checkpoint invariants.
- [ ] Run distributed and remote GPU smoke gates.
- [ ] Commit, push, and submit the two-epoch world8 preempt run.

## Validation

- Branch commits through `c65838c` are pushed. Grid modules live in
  `nimloth.wm.grid`, DINO cache identity/readout in
  `nimloth.backbone.dino_grid`, and the explicit objective in
  `nimloth.training.sft2.dino_grid`; the generic algorithm contains no hidden
  DINO branch.
- Fixed legacy expansion dropping each trajectory's final transition. The
  final real observation now gets a target-only assistant query prefix; it
  produces no extra CE. New v2 caches use expansion fingerprint
  `wm_expand_v2_terminal_next`, so incomplete existing v2 caches fail closed.
- Historical `dedup_sharded_v1` is supported read-only. Its current rows and
  pixels remain immutable; the compatibility adapter tokenizes only a missing
  terminal next prompt using cached `grid_thw` and BF16 pixel indices. It does
  not reopen or reprocess source images during training.
- Real-cache gate `validate_dino_grid_cache.py` passed. Train is 3217 records,
  59,389 transitions, 62,606 cached images, and 59,389 sampler-owned current
  steps; val is 355/6,054/6,409/6,054. Sampled current and next encodings,
  including terminal rows, match fresh processor output tensor-for-tensor after
  applying the training boundary that removes target labels. DINO fingerprint
  is `b50d261e2b533f3e`.
- Remote PyTorch 2.8 regression: `84 passed, 1 skipped`; focused first pass:
  `23 passed`. ID33 strict warm start loads epoch10/step9280 with every old
  auxiliary parameter mapped, only zero-initialized `temporal_position` new,
  and all parameters finite.
- No GPU/Slurm experiment has started for this branch. GPU smoke remains a
  required gate.
