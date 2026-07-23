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
- The first narrow cast fix exposed the real refactor drift: authoritative ID33
  kept online encoder/WM/DINO decoder/ValueHead in FP32, while the new builder
  cast all auxiliaries to Qwen BF16. The builder now restores FP32 grid
  auxiliaries and keeps only the frozen SFT1 slot projector at Qwen dtype;
  a builder-level regression locks this boundary. A fresh W&B ID and output
  directory are required for the next retry.

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

## K16 merge repair ID2 failed closed on PEFT layout difference

- Checkpoint-only step `485251.7` loaded the four-shard vagen79 base and k16
  epoch5 adapter, then the independent-storage gate rejected the merged model
  before export. There was no OOM, training, W&B, or usable `hf_merged`.
- Refined cause: k16 saves complete embedding/head tensors under PEFT's
  `save_embedding_layers` keys while `adapter_config.json` has no
  `modules_to_save`; unlike the validated k1 layout, `merge_and_unload()` leaves
  the two public modules tied.
- The merge path now detects the saved input/head pair, creates a distinct
  output `Linear`, copies each authoritative adapter tensor to its own module,
  then applies the same vocab/config/storage gates. ID2 is not reused; config
  points to a fresh ID3 export pending validation.

## K16 untied-head restore ID3 completed

- Step `485251.8` on commit `327f34c` produced the isolated corrected export
  `sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged`.
- All 698 adapter tensors verified. Exported input embedding and `lm_head` each
  exactly equal their distinct adapter tensor; safetensors contains both,
  Transformers reload keeps independent storage, and both config levels are
  untied. Slot projector SHA256 is
  `340d90a84a17f7aba3525f2f49e20921fd4f73a6534149587de2b3c875542ce0`.
- Corrected and old processors have identical tokenizer vocab, special IDs,
  and image-processor dictionaries. The Qwen/DINO v1 cache therefore retains
  the same preprocessing semantics; the config records its original processor
  source solely to preserve legacy path-based fingerprint lineage.
- No training, W&B, OOM, or resumable state was involved. The next SFT2 smoke
  must use this corrected model and a fresh `nimloth-sft2` ID45/output.

## Corrected DINO-grid SFT2 ID45 smoke passed

- Step `485251.10`, W&B `nimloth-sft2/4v68cj6z`, and commit `cf8f9df`
  ran the corrected k16 untied-head export plus ID33 auxiliary warm start on
  eight H800 GPUs with per-rank B1 and GA1. Twenty optimizer steps completed
  with finite CE, WM, DINO-grid, value, and global SIGReg losses; there was no
  OOM, NaN, NCCL, or DDP failure.
- Context length reached H4 and the detached online history cache reached 20
  entries. Global SIGReg B fell from 8 to 5 as short trajectories terminated,
  giving runtime evidence that terminal transitions participate in the same
  one-current-step objective. Step 20 reports CE `7.638404`, WM `0.123687`,
  DINO `0.492780`, value `0.082413`, and SIGReg `2.104651`.
- Observed peak GPU memory was `60,469 MiB / 81,559 MiB`, so the intended full
  world8 B1 topology has headroom. The step was intentionally cancelled after
  the smoke gate to avoid a large bounded-run checkpoint: Slurm state is
  `CANCELLED by 3738`, elapsed `00:03:16`, exit `0:9`.
- Cancellation interrupted epoch-end checkpoint output. `epoch_001` lacks the
  required `wm_predictor/` and `value_head/` trees and `best/` contains only its
  first model shard; neither directory is resumable or usable. The full run
  must use a fresh ID46/output and fresh optimizer, with periodic checkpoints.

## Corrected DINO-grid SFT2 ID46 two-epoch run started

- Hold `485251`, step `485251.14`, W&B `nimloth-sft2/yapevfpy`, and commit
  `f060a25` started the full 3,217/355 task-disjoint train/validation records on
  preempt/dgx-42 with eight H800 GPUs, per-rank B1, and GA8. Initialization is
  the corrected k16 untied-head SFT1 plus ID33 auxiliary warm start with a fresh
  optimizer; immutable v1 Qwen/DINO caches are read-only.
- Runtime sampler identity reports all 59,389 training current steps. The first
  eight optimizer steps have finite CE, WM, DINO, value, and global SIGReg
  losses with global B8 and H4 history growth. Step 8 is total `5.219125`, CE
  `4.527827`, WM `0.210067`, DINO `0.482060`, value `0.096401`, and SIGReg
  `3.327872`; no OOM, NaN, NCCL, traceback, or fatal DDP error occurred.
- GPU usage is `62,031--62,091 / 81,559 MiB` at 88--100% utilization. The first
  eight steps average about 7.4 seconds, giving a current 4.5--5.5 hour ETA
  including full validation/checkpoint overhead. Twenty-minute interval and
  epoch/best/final checkpoints are enabled; resume must use a complete ID46
  checkpoint, never the partial ID45 save.
