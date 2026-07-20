# 2026-07-19 DINOv2 4×4 grid SFT1

## Human-approved design

- Work only on `feat/sft1-dino16-grid-wm` in `/workspace/remote2/nimloth-feat-sft1-dino16-grid-wm`.
- SFT1 uses 16 injected query tokens per assistant step.
- Frozen public teacher is exactly `facebook/dinov2-large`; no model/cache fallback.
- Teacher final spatial patch map is adaptively average-pooled to 4×4 in row-major order.
- One shared `2048→2048→1024` slot projector is applied independently to every query slot.
- SFT1 uses success-only trajectories, prefix-exact transition forwards, CE + `1.0 × DINO grid MSE`.
- SFT2 must not run DINO. It consumes the SFT1 Qwen and shared projector; one joint grid WM predicts the complete next `[16,1024]` state from current grid + action.

## Implemented locally

- Extended `FrozenDINOEncoder` with parameter-free final-patch adaptive grid pooling and fail-closed patch-shape checks.
- Added `SharedSlotProjector` and one joint transformer `GridLatentWMPredictor`.
- Added SFT1 grid alignment and grid-WM loss helpers.
- Added prefix-exact experimental SFT1 entry point `experiments/training/sft1/train_dino_grid.py`:
  - transition-expanded success-only data;
  - LLM LoRA, frozen Qwen vision, trainable additive 16-query adapter and shared slot projector;
  - frozen online DINOv2 grid targets;
  - epoch checkpoint/resume with separate LoRA, query-adapter and projector states;
  - final merged HF checkpoint plus explicit grid-state interface metadata.
- Added config manifest `configs/training/sft1/qwen25vl_lora_k16_dinov2_grid4.yaml`.

## Validation

- Targeted grid/DINO tests: `15 passed`.
- Broader SFT1+SFT2 suite: `78 passed` when excluding the inherited known-broken `test_trajectory_prefix_encoding.py` (its local `token_id_map` is referenced before assignment and is unrelated to this branch).
- `compileall` and `git diff --check` pass.
- No GPU, Slurm, remote training, or model-weight smoke has been started for this design.

## GPU smoke attempts

- Two pre-allocation submission attempts created no job/GPU/W&B run: the first non-login shell lacked the Slurm module; retry1 was rejected because the script lacked `account=peilab`. Both output directories are marked `SUBMISSION_FAILED` and were not reused.
- Approved smoke retry2: W&B name reserved as `14_smoke_retry2_dino2l_grid4_k16_prefix_success1_l1_ep1_b1_ga1_ws1_px602112`; Slurm `481441`, commit `b913fcc`, one GPU on `dgx-13`.
- Job `481441` failed in 20 seconds before model/data/W&B initialization because the new entry point unpacked three values from canonical four-value `setup_dist()`. No metrics/checkpoint and not resumable; output README and experiment-group progress were updated.
- Fixed both new SFT1/SFT2 entry points to use `(rank, world_size, local_rank, device)` and registered `E0030`.
- Smoke retry3 Slurm `481444` (commit `8c64e1c`) loaded Qwen, then failed in 25 seconds before W&B/data/steps because offline DINO lookup did not use the complete shared HF cache. No metrics/checkpoint, GPU released. Launcher now explicitly exports `HF_HOME=/project/peilab/atst/.cache/huggingface`; registered `E0031`. The cache contains the requested `facebook/dinov2-large` commit `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` config, processor, and weight blobs; no substitute model is used.
- Smoke retry4 Slurm `481457`, commit `037c6e1`, W&B `35rpx073` (`failed`) loaded exact Qwen+DINO and expanded 1 train/val trajectories to 17/12 transitions. First forward failed before optimizer step because embedding resize did not synchronize Qwen config vocab (`151936` stale). No metrics/checkpoint; GPU released; W&B ID14 consumed. Added canonical vocab-sync helper/test and registered `E0032`.
- **Valid smoke retry5 passed**: commit `d098732`, Slurm `481468` completed in `00:01:34` on one dgx-13 GPU; W&B `27eybfpd` finished. 17 train/12 val transitions, 17 optimizer steps all finite. Aggregate train CE/DINO/total=`7.5728999/0.86995595/8.44285597`; val CE/DINO=`5.61814144/0.85102797`. Sampled memory ~47.8GiB. `epoch_001`, `latest`, and `final/hf_merged` passed metadata, finite query/698 adapter tensors, projector reload `[1,16,1024]`, and merged two-shard index gates. This validates execution/checkpoint semantics only, not model effectiveness.
- Formal world8 ID16: commit `801de41`, Slurm `481482`, W&B `4k53mvne`; failed after 1m14s before step1. PEFT suffix matching installed 96 unintended visual LoRA tensors even though vision mode was freeze; DDP correctly rejected those unused trainables. Single-GPU checkpoint audit showed 252 intended language LoRA-B and query adapter updated, while all 96 visual LoRA-B stayed zero. Fix re-applies path freeze after PEFT and fail-closes on any visual trainable; `find_unused_parameters` is not used. Registered `E0033`.
- World2 freeze gate ID17 passed: commit `05a3e8a`, Slurm `481492` completed in 1m30s, W&B `lyn0h127` finished. Nine optimizer steps plus distributed val/checkpoint passed; query and 252/252 language LoRA-B updated, all 96 frozen visual LoRA-B remained strict zero.
- Formal retry1 ID18 is running: commit `05a3e8a`, Slurm `481494` on dgx-52 world8, W&B `ie19vs47`. Reached at least step5 after passing the former DDP failure point; CE/DINO/total finite, all GPUs active, sampled memory 23.5–55.8GiB, and no traceback/OOM/NCCL/NaN. First resumable checkpoint will exist after epoch1.

## Remaining before formal training

1. Build a fast preprocess path/cache for prefix-expanded SFT1; current online processor path is correct but may be CPU-bound.
2. Request human approval for exact initialization checkpoint, output path, epochs, GPUs, walltime and smoke/formal launch.
3. Run GPU smoke under experiment-start protocol before formal training.

## SFT2 handoff now implemented

- `experiments/training/sft2/train_grid.py` loads the merged SFT1 Qwen and fail-closed slot-projector metadata, freezes both, and trains one joint spatial-transformer WM plus value head.
- The WM maps current full `[B,16,1024]` grid + action directly to next full grid; DINO is absent.
- Legacy SFT2 DINO CLI combinations are now rejected explicitly, and historical configs are marked forbidden.
