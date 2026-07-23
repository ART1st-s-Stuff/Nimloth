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

## Planned GPU smoke

- Project/run: `nimloth-sft2`, ID44,
  `44_smoke_dinogrid_k16_h4_terminalcache_b1_ga1_ws8_px100352`; `smoke` is the
  comment and params identify k16/H4/read-only terminal cache/B1/GA1/world8.
- Commit/entry/config: `8dba9d7d13e9dab15211f0428d6aba118584f870`,
  `experiments/training/sft2/train_dino_grid_world8.sh`,
  `configs/training/sft2/dino_grid_k16_h4.yaml`.
- Data: the authoritative 3217/355 task-disjoint records and historical cache;
  smoke limits train/val to the first eight records. The complete data/cache
  lineage was already covered by the CPU gate above.
- Initialization: k16 SFT1 plus ID33 epoch10/step9280 auxiliary warm start;
  fresh optimizer, no resume. Output is an isolated ID44 smoke directory and
  is not a formal initialization artifact.
- Trainable: Qwen vision, online grid encoder, H4 temporal-spatial WM, DINO
  decoder, and ValueHead. Frozen: Qwen language, SFT1 slot projector, DINO
  teacher/cache, EMA target encoder; older online history is detached.
- Resources: reuse preempt hold job `485251` on dgx-42, one node/8 H800,
  per-rank B1/GA1. No additional allocation. Expected wall time 10--25 minutes;
  stop after long-prefix/terminal finite evidence. Monitor CE/WM/DINO/value,
  global SIGReg B, OOM/finite, DDP/NCCL, memory, and step timing.

## ID44 attempt 1 ended before model load

- Step `485251.3` used the existing preempt hold on dgx-42 and commit
  `9f9a59204c8abd5133cbada3a4fd1af911d99159`. All eight ranks exited argparse
  code 2 because the launcher omitted required `--model`, even though the YAML
  contained `init.sft1_checkpoint`.
- No model or cache was loaded, no W&B run was created, no optimizer step or
  checkpoint exists, and the attempt is not resumable. The step ended; only the
  hold batch remains and all eight GPUs are available inside it. Output README
  records the exact command/config/data/init and failure.
- Launcher fix adds an explicit checked `MODEL_PATH`, passes `--model`, and
  makes B/GA runtime values explicit in both logs and CLI. Retry may keep ID44
  because attempt 1 created no W&B identity or training artifact.

## ID44 attempt 2 ended at cache preflight

- Step `485251.4` used commit `775c5777546126e73de19c88bfce46b8ec4f0bb2`.
  The k16 SFT1 model, ID33 warm start, W&B run `k76tiux2`, and NCCL world8 all
  initialized; no OOM occurred.
- All ranks then failed before the first training forward because cache
  manifest validation compared the full v1 cache count (59,389 transitions)
  with the eight-record smoke prefix count as if both represented full data.
  There are no metrics, optimizer steps, or checkpoints; ID44 is not resumable.
- Commit `b380387` fixes this without weakening cache identity checks: an
  explicit unfiltered `max_records` dataset may read the corresponding prefix
  from a larger full cache, while full runs and non-prefix filtered datasets
  retain exact count validation. Remote focused regression is `11 passed`.
- Attempt 2 unexpectedly wrote ID44 to the unrelated `flower` project because
  the shared credential `.env` overwrote the launcher's earlier project
  export. The target `nimloth-sft2` project still ends at ID43, so the corrected
  retry uses ID44 there but a fresh, distinct output directory. The launcher
  now restores its explicit project identity after loading credentials.

## Corrected nimloth-sft2 ID44 reached the first WM forward

- Step `485251.6`, commit `baa159c`, and W&B `nimloth-sft2/f2d3i7e9`
  confirmed the corrected project identity and passed model/ID33/NCCL/cache
  manifest/DataLoader initialization. The full v1 cache successfully served
  the requested 8-record prefix, so commit `b380387` is runtime-validated.
- The first real WM forward then failed on every rank before loss completion:
  FP32 one-hot actions were passed to ID33's BF16 LeWM `action_encoder`, whose
  Conv1d requires input and bias dtypes to match. This was not an OOM; no
  backward, optimizer step, metric, or checkpoint exists, so the run is not
  resumable.
- The fix casts action one-hot inputs to the grid predictor's own parameter
  dtype at the module boundary and adds a BF16-module/FP32-state regression.
  A fresh W&B ID and output directory are required for the next retry.

## SFT1 untied lm_head repair is required before retry

- Human pointed to `fix/sft1-merge-untied-head`; its commit `306295f` proves
  that calling `resize_token_embeddings()` after `merge_and_unload()` silently
  re-tied and overwrote the trained policy head. The fix validates independent
  input/output storage and updates metadata without resizing merged weights.
- The k16 SFT1 used by the DINO smoke is affected: its index has
  `model.embed_tokens.weight` but no `lm_head.weight`, while nested
  `text_config.tie_word_embeddings=true`. Therefore all earlier DINO smoke
  attempts used an invalid frozen CE head and cannot validate loss quality.
- This branch now contains the merge fix and regression, plus a fail-closed
  prebuilt-cache processor-source field. The k16 epoch5 adapter will be merged
  into a new output; old Qwen/DINO cache reuse is allowed only after proving
  the corrected export's processor files are byte-identical to the original
  cache source. The original SFT1 and cache remain immutable.
