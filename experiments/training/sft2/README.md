# SFT2 experiments

This directory contains launchers and thin entry points. Production library
code lives in `src/nimloth/training/sft2/`.

| File | Purpose |
|------|---------|
| `train.py` | Thin entry point for `nimloth.training.sft2.trainer` |
| `train_vagen79_default.slurm` | Config-driven 8-GPU training job |
| `build_preprocess_cache.py` | CPU preprocess-cache entry point |
| `build_compact_cache.slurm` | CPU-only compact-cache job |
| `submit_cache_then_train.sh` | Cache job followed by dependency-gated training |
| `submit_default_8gpu.sh` | Default LLM-freeze, vision-full run |
| `submit_llmvis_lora_8gpu.sh` | LLM and vision LoRA variant |
| `resolve_sft1_init_for_sft2.sh` | Resolve/merge the SFT1 initialization checkpoint |
| `upload_sft2_wandb_from_csv.py` | Upload training CSV history |
| `upload_sft2_eval_wandb.py` | Upload actual greedy-rollout evaluation results |

Configuration is owned by `configs/training/sft2/`. Unknown YAML fields fail
at startup. Production accepts only `trajectory_online_cache`: complete
trajectory lanes stay on one rank and advance in time order. A state is encoded
once when it is the current transition, then reused as detached history. Removed
row-by-row and activation-offload OOM fallbacks are not accepted by the CLI.

## Compact preprocess cache

`preprocess_cache_format: compact` deduplicates image tensors while preserving
the independent per-prefix forward contract. Build it before reserving GPUs:

```bash
export PREPROCESS_CACHE_DIR=/path/to/cache
export SFT1_RUN=/path/to/sft1_run
export BASE_HF=/path/to/source_hf
export RECORDS_ROOT=/path/to/records
bash experiments/training/sft2/submit_cache_then_train.sh
```

The training dependency uses `REQUIRE_PREBUILT_CACHE=1`, so missing or stale
cache fingerprints fail instead of rebuilding inside the GPU allocation.

Static JSONL success-label prevalence can be inspected with
`diagnosis/report_dataset_success.py`. It is a dataset statistic, not a model
metric and never selects SFT2 checkpoints. Model selection uses `val_wm_mse`;
actual agent quality must be measured by a rollout evaluator.

Outputs go to `outputs/experiments/training/sft2/<date>/<experiment>/`, with
Slurm logs under `outputs/experiments/training/sft2/slurm/`.
