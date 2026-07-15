# k=1 inject SFT1/SFT2 control

## Purpose

Train a controlled k=1 comparison for the formal k=8 inject pipeline. The intended independent variable is latent query count only: k=1 instead of k=8.

## Frozen design

- Code change commit: `09fa71a`; clean detached server launch worktree commit: `3d46066`.
- Runtime: `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Query protocol: `inject`; query CE label masked in SFT1 and SFT2.
- Source checkpoint: `/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-24/vagen_legacy_wm_entropy01_kl001_60step_2env4train/checkpoints/global_step_60/actor/huggingface`.
- Strict records root: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c`.
- Data semantics: SFT1 trains the 613 strictly valid successful train rollouts and validates on 355 strictly valid val rollouts. SFT2 trains all 3217 strictly valid train rollouts and validates on the same 355 val rollouts. Validation-issue records remain excluded. Existing verified train/val split is task-disjoint; test scenes are scene-disjoint from train scenes.
- The JSONL source contains k=8 formatting, but the canonical renderer normalizes each latent block to exactly k=1 before tokenization; no stored target with eight queries is used directly.

## SFT1

- Config: `configs/training/sft1/qwen25vl_lora_k1_inject.yaml`.
- Trainable: Qwen LoRA r64/alpha128 and newly added query-token embedding row; base weights otherwise frozen.
- Budget: 5 epochs, 8 GPUs, per-rank batch1, grad accumulation8, LR2e-4, embedding LR5e-4, max length12000, max pixels602112.
- CPU preprocess cache is built first; GPU training requires the fingerprint-matching prebuilt cache.
- W&B: project `nimloth-sft1`, ID1, run `1_k1inject_success613_lora64a128_ep5_b1_ga8_ws8_px602112`.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/control_k1/sft1/1_k1inject_success613_lora64a128_ep5_b1_ga8_ws8_px602112`.
- Checkpoints are saved every epoch plus best/final. Resume auto-selects latest complete epoch checkpoint; W&B run ID is persisted and reused.

## SFT1 merge

- After SFT1 succeeds, epoch5 is merged into a BF16 standalone HF checkpoint on one GPU. This avoids a CPU merge producing a large FP32 init and gives SFT2 cache/train one canonical init path.

## SFT2

- Config: `configs/training/sft2/latent_wm_value_k1_control.yaml`.
- Trainable: full vision tower, state projector, WM predictor, value head, and one-row additive query adapter. Qwen backbone is frozen; vision EMA is enabled.
- Budget: 10 epochs, 8 GPUs, per-rank batch2, grad accumulation4, full trajectory batching, max pixels100352, max images/batch12. Each accumulation micro-batch synchronizes DDP gradients because of the PyTorch 2.8 static-graph/no_sync regression.
- Loss and learning rates match the formal k=8 setup. Best checkpoint is selected by validation WM MSE.
- Compact BF16 CPU cache is built after the canonical merged SFT1 checkpoint exists; GPU training requires the matching completed cache.
- W&B: project `nimloth-sft2`, ID3, run `3_k1inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_ws8_px100352_img12_bestwm`.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/control_k1/sft2/3_k1inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_ws8_px100352_img12_bestwm`.
- SFT2 checkpoints include periodic `latest`, every epoch, best, and final. Resume checks k/mode/query-tune/cache/training invariants and reuses the persisted W&B run ID.

## Pipeline and resources

Dependency chain: SFT1 CPU cache (8 CPU, up to ~3h, ~47GiB) -> SFT1 train (8 H800, historically ~30m) -> SFT1 merge (1 GPU, estimate <30m) -> SFT2 CPU compact cache (8 CPU, historically ~2h, ~85GiB) -> SFT2 train (8 H800, estimate ~16h including validation/checkpoint overhead). Preempt queue delay and repeated preemption are additional and unpredictable.

No k=8 job, checkpoint, cache, CSV, or output will be modified or deleted.

## Status

- Prepared dedicated k=1 inject configs and SFT1 W&B project/run-ID persistence.
- Local syntax checks passed. Clean server worktree `/project/peilab/atst/nimloth/.worktree/k1-sft-control` is pinned to `3d46066`; 19 relevant server tests passed.
- Human confirmed the exact controlled setup. Dependency pipeline submitted: SFT1 cache `474974` -> SFT1 train `474975` -> BF16 merge `474976` -> SFT2 compact cache `474977` -> SFT2 train `474978`.
- SFT1 cache job `474974` completed `0:0` on `intel-01` in 02:38:46. Log confirms commit `3d46066`, source checkpoint, complete success613/val355 inputs, k=1 inject, masked query labels, BF16 pixels, and both completed cache fingerprints. Cache uses 47GiB and `preprocess_cache_done.flag` exists.
- SFT1 train job `474975` is now `PENDING (Priority)` for one 8-GPU preempt node. No preempt node currently has 8 free GPUs; queue ETA is unknown. W&B run has correctly not been created before training starts. Downstream jobs remain dependency-gated.
