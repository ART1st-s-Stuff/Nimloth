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

- [ ] Restore the exact 16-token DINO cache identity and SFT1 slot-projector contract.
- [ ] Add grid-specific WM modules without putting DINO logic in the generic SFT2 algorithm.
- [ ] Generalize only the shared state/history interfaces needed for vector or grid state.
- [ ] Add fail-closed warm-start and checkpoint invariants.
- [ ] Run unit, distributed, and remote GPU smoke gates.
- [ ] Commit, push, and submit the two-epoch world8 preempt run.

## Validation

Pending.
