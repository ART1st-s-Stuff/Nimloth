# Phase 1 — format SFT (SFT1)

Canonical location for SFT1 per `ai_tasks/sft1_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Qwen2.5-VL SFT on Nimloth rollout records |
| `train_8gpu.slurm` | 8-GPU DDP train (`SFT1_TUNE_MODE=lora\|embedlr`) |
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

Config: `configs/training/sft1/qwen25vl_lora.yaml`

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

# Train (LoRA, 8 GPU)
SFT1_TUNE_MODE=lora NODELIST=dgx-52 bash experiments/training/sft1/submit_train_8gpu.sh

# Rollout collection
ENV_NODE=dgx-13 bash experiments/training/sft1/submit_env_external_4gpu.sh
bash experiments/training/sft1/submit_rollouts_greedy.sh

# Per-epoch eval watcher
TRAIN_OUT=.../sft1_train_lora BASE_MODEL=.../global_step_79/actor/huggingface \
  bash experiments/training/sft1/submit_ckpt_eval_watcher.sh
```

## Legacy

SFT1 scripts in `experiments/navigation_baseline/` are frozen. Do not add new files there.
