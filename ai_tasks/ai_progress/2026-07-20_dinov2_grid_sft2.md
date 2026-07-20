# 2026-07-20 LeWM-style grid SFT2 with decoded DINO supervision

## Human-approved design

- SFT2 uses all 3,217 train trajectories / 59,389 transitions, including failures.
- Input representation is the completed SFT1 4×4 query grid. Qwen and the SFT1 shared slot projector stay frozen.
- Add a trainable WM encoder and decoder around one joint spatial WM.
- WM latent shape remains `[B,16,1024]`.
- Latent target uses an EMA encoder with decay `0.99` and stop-gradient.
- Preserve next-query latent MSE and LeWM SIGReg (`λ=0.1`).
- Decoder predicts next-RGB DINOv2 final-patch features pooled to row-major 4×4; decoded DINO MSE weight is exactly `0.5`.
- Value loss remains enabled.

## LeWM source audit

Pinned `external/le-wm` has:

- pixel ViT encoder;
- LeWM `MLP` projector;
- action `Embedder`;
- causal AdaLN-zero `ARPredictor`;
- LeWM `MLP` pred-proj;
- next-embedding MSE + SIGReg.

It has no decoder and no EMA target encoder. The approved adaptation therefore:

- reuses LeWM MLP for shared per-slot online encoder and DINO decoder;
- reuses LeWM Embedder and AdaLN-zero ConditionalBlock;
- changes predictor attention from causal temporal attention to bidirectional spatial attention over the 16 grid slots;
- adds the explicitly requested EMA target encoder and decoded-DINO objective. This is documented as a spatial adaptation, not an unmodified LeWM decoder.

## SFT1 dependency completed

Canonical checkpoint:

`/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/sft1/18_retry1_dino2l_grid4_k16_prefix_success7309_l1_ep5_b1_ga8_ws8_px602112/final/hf_merged`

SFT1 Slurm `481494` completed 5 epochs/575 steps in 3h15m36s; W&B `ie19vs47` finished. Val DINO grid MSE improved monotonically from `0.501212` to `0.337753`; final checkpoint gates passed.

## Local implementation in progress

- Added LeWM per-slot encoder/decoder, noncausal spatial AdaLN-zero predictor, and EMA target encoder.
- Updated SFT2 loss to combine latent MSE + `0.5×` decoded DINO MSE + `0.1×` SIGReg + value MSE.
- Terminal transitions retain DINO and value supervision even when no next-query prefix exists.
- Compact preprocess cache now propagates `next_image_path` for DINO targets.
- Reworked `experiments/training/sft2/train_grid.py` for frozen SFT1 modules, trainable encoder/WM/decoder/value, EMA updates, checkpoint/resume, CSV and W&B.
- Added exact fail-closed float32 DINO 4×4 sidecar cache with teacher/processor/parent/image fingerprints, resumable shards, terminal-next-RGB coverage, and online/cache bitwise gate. Existing CLS cache is incompatible and cannot substitute.
- Added dedicated k16 compact Qwen cache builder and CPU/GPU cache Slurm scripts.
- Added checkpoint/resume round-trip, terminal transition, EMA, noncausal connectivity, grid cache, and compact-cache next-path tests.
- Broader SFT1+SFT2+DINO suite: 86 passed (excluding the inherited known-broken trajectory-prefix test); compileall, shell syntax, and diff-check pass.

## Approved gate chain submitted

- Human approved CPU compact cache → dependent 1-GPU DINO grid cache → dependent world2 smoke. Formal SFT2 remains unapproved.
- Fixed code commit: `b8659fe4d04e8fea47450e2b71006daf0131cf13`.
- Qwen cache job: Slurm `482287` (CPU, 8 CPU/128G/12h).
- DINO grid sidecar job: Slurm `482288`, `afterok:482287` (1 GPU/1h).
- World2 smoke: Slurm `482289`, `afterok:482288` (2 GPU/30m).
- W&B reserved without creating a duplicate: project `nimloth-sft2`, ID31, run `dz48wt5c`, name `31_smoke_lewmgrid_dino05_ema099_all1_ep1_b1_ga1_ws2_px100352`.
- Cache root: `outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe`.
## Gate results

- Qwen cache Slurm `482287` completed in 1h50m02s: train/val 59,389/6,054 transitions, 62,606/6,409 unique images, fingerprints `b94ea4f1f05d08b5`/`42b4de9e90f8b42c`, k16/inject/masked/BF16 metadata valid.
- DINO grid cache Slurm `482288` completed in 15m01s: train/val manifests `b4bc4650a987677d`/`da67814e8537dbcd`, combined fingerprint `b50d261e2b533f3e`; four online/cache samples bitwise equal. Total cache ~94GB, including ~5GB DINO grids. Both caches are valid and reusable.
- World2 smoke Slurm `482289`, W&B `dz48wt5c`: **INVALID despite exit0**. Ten train steps were finite, but all validation metrics became NaN. Online encoder checkpoint had 37 negative BatchNorm running-variance entries.
- Root cause: `SafeBatchNorm1d` used uninitialized scratch running buffers; train mode hid corruption by using batch statistics and eval exposed it. ID31 epoch/latest/final checkpoints are forbidden.
- Fixed scratch buffers to clone current running stats, added standard-BN/repeated-forward regression tests, and added fail-fast non-finite train/val checks. Registered `E0034`.

## BatchNorm-fix world2 result

- Fix commit `1e5208bf512f437a879b0d17679c736f7225a6e7`; 88 relevant tests passed.
- Slurm `482446`, W&B ID32 `3qhd3t97`, completed 0:0 in 3m23s on dgx-10.
- Two ranks completed 20 train transitions / 10 optimizer steps and distributed validation over 12 transitions.
- Final train total/latent/DINO/SIGReg/value: `1.185793 / 0.782094 / 0.692199 / 0.556406 / 0.001958`.
- Val total/latent/DINO/SIGReg/value: `2.228710 / 0.729252 / 1.110717 / 0.478685 / 0.896231`; all finite.
- Epoch/best/latest/final checkpoints exist. All saved module floating tensors are finite; no BatchNorm running variance is negative. State is epoch1/step10/best2.2287099063.
- CPU MaxRSS was ~11.3GB. W&B did not capture system GPU telemetry; one initialization sample was 7.4GiB/GPU and is not a train-peak measurement.

## Formal launch approval

- Human explicitly approved starting formal SFT2 after the valid cache and world2 gates.
- Canonical configuration: all 3,217 train / 355 val trajectories, required compact and DINO caches, world8 on one node, 10 epochs, batch2/GA4 (effective batch64), SIGReg projections1024, 48h allocation.
- Added `train_grid_world8.slurm`; it fails closed on both caches and resumes only from this run's latest completed epoch. Invalid ID31 checkpoints are never used.
- Formal W&B/output IDs and Slurm job are recorded below after submission.
