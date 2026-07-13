# SFT2 training (Phase 2)

Canonical location for SFT2 per `ai_tasks/sft2_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Thin experiment entry → `nimloth.training.sft2.trainer` |
| `train_vagen79_default.slurm` | 8-GPU Slurm job (reads yaml config) |
| `build_compact_cache.slurm` | CPU-only compact preprocess-cache build |
| `submit_compact_cache.sh` | Submit only the CPU cache build |
| `submit_cache_then_train.sh` | Submit cache, then dependency-gated 8-GPU training |
| `submit_default_8gpu.sh` | Default: LLM freeze + vision full |
| `submit_packed_forward_8gpu.sh` | Opt-in packed-forward (`PACKED_FORWARD=1` after GPU equiv) |
| `submit_llmvis_lora_8gpu.sh` | LLM+Vision LoRA variant |
| `submit_llmvis_lora_preempt6g_dgx11.sh` | Preempt 6-GPU variant |
| `eval_val_rollout_success.py` | Offline val trajectory success rate |
| `upload_sft2_wandb_from_csv.py` | Retroactive wandb upload from CSV |
| `upload_sft2_eval_wandb.py` | Upload VAGEN greedy rollout eval per-ckpt metrics to wandb |
| `smoke_speedup.py` / `smoke_speedup.slurm` | 1-GPU speedup smoke (encode/cache/trajectory-once equiv) |
| `validate_trajectory_once_2step.py` / `validate_2step.slurm` | 2-step GPU 验收：text/image synthetic + real record 前 2 step |
| `probe_trajectory_once_equiv.py` | GPU: legacy vs trajectory-once (latent + CE + WM + value) |
| `estimate_preprocess_cache.py` | Sample transition vs trajectory cache byte estimates |
| `resolve_sft1_init_for_sft2.sh` | Merge SFT1 LoRA → hf_merged for SFT2 |

Config: `configs/training/sft2/latent_wm_value.yaml`

Profiling / speedup (see `ai_tasks/sft2_speedup_plan.md`):

| Config | Purpose |
|--------|---------|
| `latent_wm_value_profiling.yaml` | `batch_size=2`, `grad_accum=4`, `--step-timing` |
| `latent_wm_value_vision_freeze_profiling.yaml` | P6 vision-freeze diagnostic |

CLI knobs: `--preprocess-cache-dir`, `--preprocess-cache-format`, `--preprocess-cache-image-dtype`, `--require-prebuilt-cache`, `--dataloader-workers`, `--dataloader-prefetch-factor`, `--packed-forward`, `--latent-token-count`, `--latent-query-mode inject|generate`, and `--query-tune freeze|adapter`. `adapter` trains a small additive latent-query embedding table, folds it into the saved embedding rows, and is intentionally rejected with LoRA. `--[no-]mask-latent-query-labels` remains a deprecated compatibility alias. `--full-trajectory-batching` is **enabled by default**; use `--no-full-trajectory-batching` to disable.

### Compact preprocess cache (default)

`preprocess_cache_format: compact` preserves independent per-prefix Qwen rows while removing on-disk image-prefix duplication:

- processor image tensors are stored once per unique resolved image path, in BF16 by default;
- input IDs, labels, image grids, and compact image indices are stored in transition shards;
- worker-local `torch.load(..., mmap=True)` shard LRUs avoid thousands of tiny-file decodes;
- DataLoader workers reconstruct the exact per-prefix image batch, including next-state rows, before pinned-memory/non-blocking GPU transfer.

Build the cache on CPU first and make GPU training fail rather than rebuild it:

```bash
export PREPROCESS_CACHE_DIR=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/cache/<run>
export SFT1_RUN=/path/to/sft1_run
export BASE_HF=/path/to/source_hf
export RECORDS_ROOT=/path/to/records
bash experiments/training/sft2/submit_cache_then_train.sh
```

The training job is submitted with `afterok:<cache_job>` and `REQUIRE_PREBUILT_CACHE=1`; no GPU is held during PIL/processor preprocessing. Compact cache is for the standard per-prefix forward path, including full-trajectory batch sampling. Packed-forward keeps its separate trajectory cache until packed semantics are accepted.

For k=8 latent-query runs, either use `configs/training/sft2/latent_wm_value_k8.yaml` or set `LATENT_TOKEN_COUNT=8` in Slurm wrappers. The k=8 config uses `query_mode: inject` and `query_tune: adapter`.

### `--full-trajectory-batching` (方案 B, 默认启用)

Each micro-batch = one complete trajectory (all transitions for one record).
Qwen still sees each prefix as independent rows (per-prefix forward), not
packed-forward.  This lets SIGReg access the full trajectory's projected
embeddings without violating Qwen-VL prefix non-invariance.

- Distinct from `--packed-forward` (which fuses all steps into one sequence).
- **Cannot** combine with `--packed-forward` or `--trajectory-aware-batching`.
- Micro-batch size = trajectory length (variable).  Tune `--grad-accum` so
  `world * grad_accum * avg_trajectories_per_rank` matches your target effective
  batch size.
- Validation loader remains regular per-prefix transition batching (no
  full-trajectory grouping).  This is intentional: eval correctness only
  requires per-prefix WM MSE, which is independent of batch composition.

### `--packed-forward` prerequisites

1. Run GPU equivalence first:
   ```bash
   PYTHONPATH=src python experiments/training/sft2/probe_trajectory_once_equiv.py \
     --model /path/to/hf_merged --train-jsonl /path/to/train_all.jsonl --max-records 3
   ```
2. Optional smoke gate: `smoke_speedup.py --require-packed-once-equiv`
3. Packed mode uses **one trajectory per micro-batch**; tune `grad_accum` so effective steps match legacy (`world * grad_accum * avg_T ≈ 64` on 8 GPU).
4. For preprocess cache with packed mode, build trajectory cache (`train_trajectory/` / `val_trajectory/`). Estimate size:
   ```bash
   PYTHONPATH=src python experiments/training/sft2/estimate_preprocess_cache.py \
     --model /path/to/hf_merged --train-jsonl /path/to/train_all.jsonl
   ```
5. Do **not** default `PACKED_FORWARD=1` in production until equiv passes on your data.

`validate_2step` job **456802** (2026-06-20): text synthetic **0/0** latent diff (encoding OK); image synthetic step0 **~0.41**; real nav record step0 **~10** — failures are **forward semantics** (full trajectory + all images in one forward ≠ per-prefix legacy), not token/index bugs.

Library: `src/nimloth/training/sft2/`；WM 在 `wm/`；Qwen 调参在 `backbone/`；离线 eval 在 `eval/`。

SFT1 checkpoints and rollout records stay under `experiments/navigation_baseline/runs/` (legacy); SFT1 scripts are in `experiments/training/sft1/`.

**Outputs:** train checkpoints/logs → `outputs/experiments/training/sft2/<YYYY-MM-DD>/<experiment_name>/`; Slurm logs → `outputs/experiments/training/sft2/slurm/`. Legacy SFT2 runs under `experiments/navigation_baseline/runs/` are not overwritten; resume via `TRAIN_OUT_OVERRIDE`.
