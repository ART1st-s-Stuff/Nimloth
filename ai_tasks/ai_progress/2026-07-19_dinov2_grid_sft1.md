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

## Remaining before formal training

1. Build a fast preprocess path/cache for prefix-expanded SFT1; current online processor path is correct but may be CPU-bound.
2. Request human approval for exact initialization checkpoint, output path, epochs, GPUs, walltime and smoke/formal launch.
3. Run GPU smoke under experiment-start protocol before formal training.

## SFT2 handoff now implemented

- `experiments/training/sft2/train_grid.py` loads the merged SFT1 Qwen and fail-closed slot-projector metadata, freezes both, and trains one joint spatial-transformer WM plus value head.
- The WM maps current full `[B,16,1024]` grid + action directly to next full grid; DINO is absent.
- Legacy SFT2 DINO CLI combinations are now rejected explicitly, and historical configs are marked forbidden.
