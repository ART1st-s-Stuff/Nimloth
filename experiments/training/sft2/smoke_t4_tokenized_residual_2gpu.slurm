#!/bin/bash
#SBATCH --job-name=sft2-t4tok8-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail
REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
CONFIG=$REPO/configs/training/sft2/latent_wm_value_k8_state8192_t4_tok8_residual_ep5_img8.yaml
MODEL=$ROOT/sft1/epoch_005/hf_merged
RECORDS=$ROOT/converted_strict_k8_b6c811c
RUN_NAME=${RUN_NAME:?set RUN_NAME}
cd "$REPO"
set -a; source /project/peilab/atst/flower/.env; set +a
export PYTHONPATH=$REPO/src:$REPO/external/le-wm:$REPO/external/VAGEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=nimloth-sft2 WANDB_MODE=online
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((23001 + SLURM_JOB_ID % 10000))
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 NIMLOTH_DDP_GPU_STRIDE=2
mkdir -p "$OUT_ROOT/train" "$OUT_ROOT/logs"
exec "$PY" experiments/training/sft2/train.py \
  --config "$CONFIG" --model "$MODEL" \
  --train-jsonl "$RECORDS/train_all.jsonl" --val-jsonl "$RECORDS/val_all.jsonl" \
  --output-dir "$OUT_ROOT/train" --epochs 1 --batch-size 2 --grad-accum 2 \
  --max-train-records 4 --max-val-records 2 \
  --latent-token-count 8 --latent-query-mode inject --query-tune freeze \
  --llm-tune lora --vision-tune lora --lora-r 64 --lora-alpha 128 \
  --wandb-run-name "$RUN_NAME" --gradient-checkpointing --seed 42 \
  --checkpoint-interval-minutes 20
