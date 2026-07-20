#!/usr/bin/env bash
#SBATCH --job-name=sft1-dino-grid-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00

set -euo pipefail

: "${REPO:?set REPO to the server feature worktree}"
: "${INIT_HF:?set INIT_HF to the exact VAGEN actor HF checkpoint}"
: "${TRAIN_JSONL:?set TRAIN_JSONL}"
: "${VAL_JSONL:?set VAL_JSONL}"
: "${OUTPUT:?set OUTPUT to a new exclusive directory}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME}"

ROOT=/project/peilab/atst/nimloth
PY="${ROOT}/.venv-vagen-main/bin/python3"
DINO_MODEL="${DINO_MODEL:-facebook/dinov2-large}"
mkdir -p "${OUTPUT}"

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_PROJECT=nimloth-sft1
export WANDB_RUN_NAME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

cat >"${OUTPUT}/README.md" <<EOF
# DINOv2 4x4 grid SFT1 smoke

- Status: RUNNING
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- W&B project: nimloth-sft1
- W&B run: ${WANDB_RUN_NAME}
- Init: ${INIT_HF}
- Train data: ${TRAIN_JSONL}; success-only; max 1 trajectory; prefix-expanded
- Val data: ${VAL_JSONL}; max 1 trajectory; prefix-expanded
- Trainable: language LoRA r64/alpha128, 16-query additive adapter, shared slot projector
- Frozen: Qwen vision encoder, DINOv2-L/14 teacher
- Loss: CE + 1.0 * row-major adaptive-pooled DINO final-patch 4x4 MSE
- Budget: 1 GPU, 16 CPUs, 96G RAM, 30 minutes, 1 epoch, batch1/GA1
- Resume: epoch checkpoint contains LoRA, query adapter, projector, optimizer
EOF

cd "${REPO}"
srun "${PY}" experiments/training/sft1/train_dino_grid.py \
  --model "${INIT_HF}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output "${OUTPUT}" \
  --dino-model "${DINO_MODEL}" \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-train-records 1 \
  --max-val-records 1 \
  --max-length 12000 \
  --max-pixels 602112 \
  --lr 2e-4 \
  --query-lr 5e-4 \
  --projector-lr 2e-4 \
  --dino-loss-weight 1.0 \
  --wandb-run-name "${WANDB_RUN_NAME}"
