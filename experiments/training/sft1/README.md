# Phase 1 — format SFT (SFT1)

Canonical location for SFT1 per `ai_tasks/sft1_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Qwen2.5-VL SFT on Nimloth rollout records |
| `train_8gpu.slurm` | 8-GPU DDP train (`SFT1_TUNE_MODE=lora\|embedlr`) |
| `build_preprocess_cache.slurm` | CPU-only BF16 preprocess-cache build |
| `submit_cache_then_train_8gpu.sh` | Submit cache, then dependency-gated training |
| `convert_rollouts.py` | VAGEN rollout JSONL → Nimloth SFT records |
| `derive_rollout_images_255.py` | Preserve sources and derive RGB 255×255 images with rewritten JSONLs |
| `derive_rollout_images_255.slurm` | CPU wrapper for the non-destructive image derivation |
| `merge_lora_ckpt.py` | LoRA adapter → `hf_merged` for VAGEN eval / SFT2 init |
| `rollouts_greedy_parallel.slurm` | Greedy rollout collection (Slurm array) |
| `eval_greedy_valtest.slurm` | Val/test rollout eval for a checkpoint |
| `env_external_4gpu.slurm` | Shared 4-GPU AI2-THOR env for rollouts/eval |
| `ckpt_eval_watcher.slurm` | Per-epoch eval during training |
| `summarize_eval_rollouts.py` | Aggregate eval JSONL success rates |
| `summarize_before_after_rollouts.py` | Before/after training comparison |
| `compare_eval_summaries.py` | Compare eval summary CSVs |
| `compare_rollout_resolution_probe.py` | Paired comparison for dumps with verified stable metadata; fails on visible runtime/metadata mismatch |
| `recover_rollout_resolution_pairs.py` | Diagnostic recovery for E0030-corrupted dumps via batch/runtime/instruction/initial-frame identity |
| `validate_rollout_train120_dump.py` | Exact 120-key, stable metadata/UID, runtime-config and RGB PNG completion gate |
| `submit_*.sh` | Thin sbatch wrappers (no hardcoded nodes by default) |

Config: `configs/training/sft1/qwen25vl_lora.yaml`; k=8 run manifest: `configs/training/sft1/qwen25vl_lora_k8.yaml`.

Latent query token count can be set with `LATENT_TOKEN_COUNT=<k>` in Slurm wrappers or `--latent-token-count <k>` in `train.py`. Select the protocol with YAML `latent.query_mode` or `--latent-query-mode inject|generate`: `inject` masks query-token CE labels and uses staged format evaluation, while `generate` supervises and freely generates the query-token block. `--[no-]mask-latent-query-labels` remains a deprecated compatibility alias; conflicting settings fail fast.

Library (planned): `src/nimloth/training/phase1_sft/`

## Paths

- **Scripts**: `experiments/training/sft1/`
- **Slurm logs**: `outputs/experiments/training/sft1/slurm/`
- **New train outputs**: `outputs/experiments/training/sft1/<date>/<name>/`
- **Legacy runs** (records, rollouts, eval): `experiments/navigation_baseline/runs/` — override via `SFT1_RUNS_ROOT`

Default init checkpoint: VAGEN `retry2` `global_step_79` actor HF export.

## Quick start

```bash
cd /project/peilab/atst/nimloth

# Recommended: build cache on CPU, then start LoRA training after cache succeeds.
# TRAIN_OUT, TRAIN_JSONL, VAL_JSONL, and INIT_HF must be exported.
SFT1_TUNE_MODE=lora bash experiments/training/sft1/submit_cache_then_train_8gpu.sh

# Rollout collection
ENV_NODE=dgx-13 bash experiments/training/sft1/submit_env_external_4gpu.sh
bash experiments/training/sft1/submit_rollouts_greedy.sh

# Per-epoch eval watcher
TRAIN_OUT=.../sft1_train_lora BASE_MODEL=.../global_step_79/actor/huggingface \
  bash experiments/training/sft1/submit_ckpt_eval_watcher.sh
```

For the fixed 120-task resolution probe, set `ROLLOUT_TRAIN120=1`; the dataset is exactly `base_train` seeds 1–60 plus `common_sense_train` seeds 1–60. `VAGEN_DIR` selects the old or corrected VAGEN worktree, and `EXPECTED_ROLLOUT_PNG_SIZE=512|255` makes the job fail if its persisted image path is wrong. The probe always uses greedy `temperature=0`, `top_p=1`, `top_k=-1`, `n=1`, 20 turns, one action per turn, and 512 response tokens per turn.

Validation dumps produced before the E0030 stable-identity fix may have trajectory metrics paired with the wrong `data_source/env_seed`. Direct paired comparison now fails on visible `config_id/eval_set` mismatch. `recover_rollout_resolution_pairs.py` is diagnostic-only: it can recover task pairs from control-batch membership, runtime config, instruction, and initial-frame similarity, but cannot restore exact seed labels.

SFT1 stores cached `pixel_values` as BF16 by default (`CACHE_PIXEL_DTYPE=bfloat16`), which matches the GPU visual encoder input dtype and halves their disk/read bandwidth versus FP32. The dependency-gated wrapper sets `REQUIRE_PREBUILT_CACHE=1`, so the GPU allocation never performs image preprocessing.

## Legacy

SFT1 scripts in `experiments/navigation_baseline/` are frozen. Do not add new files there.
