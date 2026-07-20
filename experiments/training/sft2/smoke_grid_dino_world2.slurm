#!/usr/bin/env bash
#SBATCH --job-name=sft2-grid-dino-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:30:00

set -euo pipefail
: "${REPO:?}" "${SFT1_CHECKPOINT:?}" "${TRAIN_JSONL:?}" "${VAL_JSONL:?}" "${OUTPUT:?}" "${WANDB_RUN_NAME:?}"
ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
mkdir -p "${OUTPUT}"
set -a; source /project/peilab/atst/flower/.env; set +a
export WANDB_PROJECT=nimloth-sft2 WANDB_RUN_NAME
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" OMP_NUM_THREADS=8
cat >"${OUTPUT}/README.md" <<EOF
# LeWM-grid decoded-DINO SFT2 world2 smoke

- Status: RUNNING
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- W&B: nimloth-sft2 / ${WANDB_RUN_NAME}
- SFT1: ${SFT1_CHECKPOINT}
- Data: all train/val, max1 trajectory each
- Trainable: LeWM MLP encoder, one bidirectional spatial AdaLN-zero WM, LeWM MLP DINO decoder, value head
- Frozen: SFT1 Qwen/projector, DINOv2 teacher; EMA target encoder decay0.99
- Loss: latent MSE + 0.5 decoded DINO MSE + 0.1 SIGReg + value MSE
- Budget: 2 GPUs, 1 epoch, batch1/GA1, 30m
EOF
cd "${REPO}"
srun "${PY}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/training/sft2/train_grid.py \
  --sft1-checkpoint "${SFT1_CHECKPOINT}" \
  --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output "${OUTPUT}" --dino-model facebook/dinov2-large \
  --epochs 1 --batch-size 1 --grad-accum 1 --max-train-records 1 --max-val-records 1 \
  --max-length 12000 --max-pixels 100352 \
  --latent-weight 1.0 --dino-weight 0.5 --sigreg-weight 0.1 --value-weight 1.0 \
  --ema-decay 0.99 --sigreg-num-proj 64 --wandb-run-name "${WANDB_RUN_NAME}"
