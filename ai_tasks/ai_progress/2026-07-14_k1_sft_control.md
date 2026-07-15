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
- Original budget was 5 epochs on 8 GPUs with per-rank batch1/grad accumulation8. At human direction after cache completion, SFT1 was changed to 4 GPUs with grad accumulation16, preserving effective batch64; all other settings remain unchanged.
- CPU preprocess cache is built first; GPU training requires the fingerprint-matching prebuilt cache.
- W&B: project `nimloth-sft1`, ID1, run `1_k1inject_success613_lora64a128_ep5_b1_ga16_ws4_px602112`.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/control_k1/sft1/1_k1inject_success613_lora64a128_ep5_b1_ga16_ws4_px602112`.
- Checkpoints are saved every epoch plus best/final. Resume auto-selects latest complete epoch checkpoint; W&B run ID is persisted and reused.

## SFT1 merge

- After SFT1 succeeds, epoch5 is merged into a BF16 standalone HF checkpoint on one GPU. This avoids a CPU merge producing a large FP32 init and gives SFT2 cache/train one canonical init path.

## SFT2

- Config: `configs/training/sft2/latent_wm_value_k1_control.yaml`.
- Trainable: full vision tower, state projector, WM predictor, value head, and one-row additive query adapter. Qwen backbone is frozen; vision EMA is enabled.
- Original budget was 10 epochs on 8 GPUs, per-rank batch2, grad accumulation4. At human direction after cache completion, final training was changed before launch to 2 GPUs on dgx-27 with grad accumulation16, exactly preserving nominal effective batch64. Full trajectory batching, max pixels100352, max images/batch12, and all other settings remain unchanged. Each accumulation micro-batch synchronizes DDP gradients because of the PyTorch 2.8 static-graph/no_sync regression.
- Loss and learning rates match the formal k=8 setup. Best checkpoint is selected by validation WM MSE.
- Compact BF16 CPU cache is built after the canonical merged SFT1 checkpoint exists; GPU training requires the matching completed cache.
- W&B: project `nimloth-sft2`, ID4, run `4_k1inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga16_ws2_px100352_img12_bestwm`. ID3 was initially reserved for this control but was consumed by a concurrent smoke run before this control received any allocation/W&B init, so the control moved to the next ID.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/control_k1/sft2/4_k1inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga16_ws2_px100352_img12_bestwm`.
- SFT2 checkpoints include periodic `latest`, every epoch, best, and final. Resume checks k/mode/query-tune/cache/training invariants and reuses the persisted W&B run ID.

## Pipeline and resources

Dependency chain: SFT1 CPU cache (8 CPU, ~47GiB) -> SFT1 train (4 H800) -> SFT1 merge (1 GPU) -> SFT2 CPU compact cache (8 CPU, ~84GiB) -> SFT2 train (now 2 H800, rough estimate ~64h including validation/checkpoint overhead). Queue delay is additional and unpredictable. Normal QOS permits at most 48h per job, so a checkpoint resume is expected unless measured ws2 throughput is faster than this estimate.

No k=8 job, checkpoint, cache, CSV, or output will be modified or deleted.

## Status

- Prepared dedicated k=1 inject configs and SFT1 W&B project/run-ID persistence.
- Local syntax checks passed. Clean server worktree `/project/peilab/atst/nimloth/.worktree/k1-sft-control` is pinned to `3d46066`; 19 relevant server tests passed.
- Human confirmed the exact controlled setup. Dependency pipeline submitted: SFT1 cache `474974` -> SFT1 train `474975` -> BF16 merge `474976` -> SFT2 compact cache `474977` -> SFT2 train `474978`.
- SFT1 cache job `474974` completed `0:0` on `intel-01` in 02:38:46. Log confirms commit `3d46066`, source checkpoint, complete success613/val355 inputs, k=1 inject, masked query labels, BF16 pixels, and both completed cache fingerprints. Cache uses 47GiB and `preprocess_cache_done.flag` exists.
- Human directed using the available GPUs on dgx-51. The unstarted 8-GPU job `474975` and dependency-only downstream jobs `474976`–`474978` were cancelled at elapsed0 and replaced without deleting data. Completed cache/run directory was atomically renamed for accurate ws4/ga16 identity.
- SFT1 job `475713` completed `0:0` on dgx-51 in 00:39:26: 5 epochs/50 steps. Val loss by epoch=`0.22636521, 0.07191491, 0.06348853, 0.06022014, 0.05827980`; staged inject format rate was 1.0 every epoch. Best/final=epoch5; all epoch checkpoints and done flag complete; no OOM/traceback/NaN/Inf. W&B `wlxx2qsp` is finished at project `nimloth-sft1`.
- Merge job `475714` completed `0:0` in 54s, verifying/merging 702 adapter tensors into canonical BF16 `epoch_005/hf_merged`. SFT1 output including cache/checkpoints/merged HF is ~114GiB. Preliminary k1 val0.05828 is slightly below prior k8 val0.05948, but this is not an end-to-end conclusion and SFT1 world size differs despite equal effective batch64.
- SFT2 compact cache `475715` completed `0:0` on intel-01 in 02:09:48, 84GiB: train 59,389 transitions/images in 464 image + 232 transition shards; val 6,054 in 48 + 24. Both manifests/done flag and k1/inject/masked/BF16 metadata are complete.
- At human direction, elapsed0 8-GPU train `475716` was cancelled. A 3-GPU dgx-44 replacement was prepared, but two of the three free GPUs were allocated to another job before ours started; jobs `476022`/`476023` were cancelled at elapsed0 with no output/W&B. The cache/output root was atomically renamed again to accurate ws2/ga16 identity.
- Job `476029` targeted the two free dgx-27 GPUs but remained elapsed0 and the node dropped to one free GPU. During this wait a concurrent `nimloth-sft2` smoke created ID3. To avoid ID reuse, `476029` was cancelled elapsed0 with no output/W&B; cache/output was atomically renamed to ID4.
- Final job `476338` requests the two currently available normal-partition GPUs on dgx-52, batch2/GA16/effective batch64, explicit `EXTRA_TRAIN_ARGS=--grad-accum=16`, 16 CPU, 128G RAM, and 48h. It is `PENDING (Priority)`; no control SFT2 W&B run yet. Periodic latest checkpoints support the expected resume if 10 epochs exceed 48h.
