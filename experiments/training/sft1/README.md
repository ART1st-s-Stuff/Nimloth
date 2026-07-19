# Phase 1 — format SFT (SFT1)

Canonical location for SFT1 per `ai_tasks/sft1_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Qwen2.5-VL SFT on Nimloth rollout records |
| `train_8gpu.slurm` | 8-GPU DDP train (`SFT1_TUNE_MODE=lora\|embedlr`) |
| `build_preprocess_cache.slurm` | CPU-only BF16 preprocess-cache build |
| `submit_cache_then_train_8gpu.sh` | Submit cache, then dependency-gated training |
| `convert_rollouts.py` | VAGEN rollout JSONL → Nimloth SFT records |
| `merge_lora_ckpt.py` | LoRA adapter → `hf_merged` for VAGEN eval / SFT2 init |
| `rollouts_greedy_parallel.slurm` | Greedy rollout collection (Slurm array) |
| `eval_greedy_valtest.slurm` | Val/test rollout eval for a checkpoint |
| `env_external_4gpu.slurm` | Shared 4-GPU AI2-THOR env for rollouts/eval |
| `ckpt_eval_watcher.slurm` | Per-epoch eval during training |
| `summarize_eval_rollouts.py` | Aggregate eval JSONL success rates |
| `summarize_before_after_rollouts.py` | Before/after training comparison |
| `compare_eval_summaries.py` | Compare eval summary CSVs |
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

# hligb step10 source-compatible rollout profile. This keeps the checkpoint's
# original prompt during collection; convert_rollouts.py performs the later
# Nimloth-format conversion.
ROLLOUT_PROTOCOL=hligb_step10_eval ROLLOUT_INCLUDE_TEST=0 \
  INIT_HF=/project/peilab/atst/vagen_ckpt_JUL19 \
  bash experiments/training/sft1/submit_rollouts_greedy.sh
# Convert this source profile only with the explicit answer-tag selector:
#   python experiments/training/sft1/convert_rollouts.py ... --source-action-tag answer

# Per-epoch eval watcher
TRAIN_OUT=.../sft1_train_lora BASE_MODEL=.../global_step_79/actor/huggingface \
  bash experiments/training/sft1/submit_ckpt_eval_watcher.sh
```

SFT1 stores cached `pixel_values` as BF16 by default (`CACHE_PIXEL_DTYPE=bfloat16`), which matches the GPU visual encoder input dtype and halves their disk/read bandwidth versus FP32. The dependency-gated wrapper sets `REQUIRE_PREBUILT_CACHE=1`, so the GPU allocation never performs image preprocessing.

## Legacy

SFT1 scripts in `experiments/navigation_baseline/` are frozen. Do not add new files there.
