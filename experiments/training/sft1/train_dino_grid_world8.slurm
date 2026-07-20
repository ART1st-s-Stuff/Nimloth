#!/usr/bin/env bash
#SBATCH --job-name=sft1-dino-grid16
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=08:00:00

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
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

cat >"${OUTPUT}/README.md" <<EOF
# DINOv2 4x4 grid SFT1 formal training

- Status: RUNNING
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- W&B project: nimloth-sft1
- W&B run: ${WANDB_RUN_NAME}
- Init: ${INIT_HF}
- Train data: ${TRAIN_JSONL}; success-only; 613 trajectories / 7309 prefix transitions
- Val data: ${VAL_JSONL}; 355 trajectories / 6054 prefix transitions
- Trainable: language LoRA r64/alpha128, 16-query additive adapter, one shared slot projector
- Frozen: Qwen vision encoder, exact facebook/dinov2-large teacher
- Loss: CE + 1.0 * row-major adaptive-pooled final-patch 4x4 DINO MSE
- Budget: 1 node / 8 GPUs / 128 CPUs / 768G RAM / 8h; 5 epochs; batch1/GA8/effective batch64
- Checkpoints: every completed epoch, latest copy, final merged HF; epoch-level resume
EOF

cd "${REPO}"
srun "${PY}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  experiments/training/sft1/train_dino_grid.py \
  --model "${INIT_HF}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output "${OUTPUT}" \
  --dino-model "${DINO_MODEL}" \
  --epochs 5 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 12000 \
  --max-pixels 602112 \
  --lr 2e-4 \
  --query-lr 5e-4 \
  --projector-lr 2e-4 \
  --dino-loss-weight 1.0 \
  --wandb-run-name "${WANDB_RUN_NAME}"
