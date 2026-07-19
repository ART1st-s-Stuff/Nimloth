# Rollout image resolution correction and paired probe

## Goal

1. Make navigation rollout retain RGB 255×255 PIL images instead of forcing them to 512×512.
2. Build a non-destructive 255×255 derivative of the converted rollout dataset.
3. Compare the source step-60 checkpoint on the same 120 train tasks under the old and corrected resolution paths.

## Confirmed mismatch

- Effective production rollout used Nimloth VAGEN `e7cc2d0` with nested verl `65316156`.
- AI2-THOR emitted 255×255 frames, but both Qwen rollout managers called verl `process_image()` whose default `min_pixels=512*512`; this changed each image to 512×512.
- Qwen2.5-VL then smart-resized 512 to 504, giving `image_grid_thw=[1,36,36]` and 18×18=324 merged vision tokens.
- Historical source VAGEN `f7aefd3` passed 255×255 images directly; standard Qwen processing maps this to 252, `image_grid_thw=[1,18,18]`, and 9×9=81 merged vision tokens.
- Existing 512×512 rollout dumps and the recorded native 504/grid36 SFT diagnostics corroborate the production path.

## Human decisions

- Synchronize all active experiment branches while preserving each branch's VAGEN lineage.
- Preserve existing images and create a derived dataset with rewritten JSONL paths.
- Probe 120 train tasks as base_train seeds 1–60 plus common_sense_train seeds 1–60.
- Run a paired A/B comparison: old 512→504 path versus corrected 255→252 path, with identical checkpoint/tasks/eval settings.

## Completed

### VAGEN fix

- Canonical VAGEN fix: `a01f7af fix(rollout): preserve 255px navigation images` on `nimloth/fix-rollout-image255`.
- Added `vagen.rollout.image_utils.prepare_rollout_image()` and used it in both local and service Qwen rollout managers.
- RED test failed before implementation; GREEN validation: `2 passed`, compileall and diff-check passed.
- Ported and pushed equivalent fixes for every distinct active VAGEN baseline:
  - `7908040 -> 55c7b7f`
  - `bb26c0d -> d3e82f1`
  - `154c537 -> 178a6dc`
  - `93c1124 -> d28a41e`
  - `3003c2e -> d30d6d9`

### Nimloth branch synchronization

- dev: `8ce2512`
- exp/k8-preprojection-recon: `56fa44a`
- exp/latent-repr-ablation: `150face`
- exp/vagen-1action: `463d000`
- feat/dinov3-query-alignment: `378fd3b`
- feat/fsdp-dynamic-rollout: `54cac57`
- feat/reconstruct: `2ecfe59`
- feat/rl: `8d9cf2c`
- feat/sft1-dino16-grid-wm: `630fa4f`
- feat/sft1-hligb-step10-rollout: `037b519`
- nimloth-lewm-repro: `6edab8d`
- recon-compare-qwen: `c6451fe`

All root commits modify only the VAGEN gitlink. Existing unrelated `.memory/memories.jsonl` changes in feat/reconstruct and recon-compare-qwen were left untouched.

### Dataset/probe tooling prepared on dev

- `derive_rollout_images_255.py`: hashes unique source paths, writes RGB 255×255 BICUBIC PNGs into a separate tree, rewrites all top-level JSONL files, preserves sources, supports resume, and writes a manifest.
- `derive_rollout_images_255.slurm`: CPU conversion job wrapper.
- `rollouts_greedy_parallel.slurm`: prepared `ROLLOUT_TRAIN120=1`, alternate `VAGEN_DIR`, exact source eval kwargs, W&B logging, and expected PNG-size gate.
- `compare_rollout_resolution_probe.py`: paired success, per-source rates, discordant outcomes, exact McNemar p-value, and PNG-size evidence.
- Current local validation: two dataset/comparison tests pass; Python compileall, shell `bash -n`, and diff-check pass.

## Active jobs (2026-07-20 server time)

- CPU image derivation job `481070`: `COMPLETED 0:0` in `00:08:55` on `intel-01` with 8 CPUs/64 GiB. The initial 32-CPU request was rejected before job creation by `QOSMaxCpuPerNode`, so the approved conversion used 8 workers. Exhaustive validation passed: all four JSONL counts preserved; 81,570 references map to 73,648 existing unique RGB 255×255 images; all 73,648 source images remain RGB 512×512 and aggregate source bytes remain `11,161,340,020`. Derived logical bytes=`4,510,928,566` (4.201 GiB), while NFS `du` reports 16 GiB because of allocation units. Output: `converted_strict_k8_b6c811c_images255`.
- Old-resolution A job `481071`: `COMPLETED 0:0` in `00:20:22` on dgx-13 with normal 6 GPUs (2 env + 4 policy). It loaded old VAGEN `e7cc2d0`, passed config/vLLM health checks, and W&B run `9l4vjc1j` finished under the exact ID10 name. All 120 paired keys and 2,370 RGB 512×512 PNG references passed. Success was base 11/60, common 8/60, overall 19/120=15.83%; action validity was 1.0. W&B logged mean scores base/common=0.36917/0.32250, while direct dumped-row recomputation is 0.37083/0.32083 (the success indicators agree exactly).
- Corrected-resolution B job `481072`: running on dgx-32 with normal 6 GPUs. GPU probes selected physical GPUs0/4 for the two AI2-THOR env services and GPUs1/2/3/5 for policy; both HTTP services became healthy. It loaded exact fixed VAGEN `a01f7af`, exact step60 path, and exact greedy/train120 arguments, then started rollout task0 at 03:27:44 server time. W&B ID11 is reserved; output is `outputs/experiments/rollout_resolution/2026-07-20/11_resprobe_step60_train120_img252_greedy`.
- Exact server worktrees verified clean: Nimloth `f7ea3da`, old VAGEN `e7cc2d0`, fixed VAGEN `a01f7af`, both verl `65316156`.
- Dataset scope measured exactly: 4 JSONLs, 4,485 records, 81,570 references, 73,648 unique source images, 10.39 GiB source bytes; estimated derived PNG bytes 4.28 GiB.

## Pending

- Monitor `481072` to healthy rollout and completion; verify W&B/run ID11, 120 records, RGB255 persisted PNG gate, and no hidden parameter differences.
- Run paired comparison over identical `(data_source, env_seed)` keys and record overall/per-source rates, discordant outcomes, exact McNemar p-value, and interpretation.
