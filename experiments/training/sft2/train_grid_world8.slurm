#!/usr/bin/env bash
#SBATCH --job-name=sft2-lewm-grid-world8
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=48:00:00

set -euo pipefail
: "${REPO:?}" "${SFT1_CHECKPOINT:?}" "${TRAIN_JSONL:?}" "${VAL_JSONL:?}"
: "${PREPROCESS_CACHE:?}" "${DINO_CACHE:?}" "${OUTPUT:?}" "${WANDB_RUN_NAME:?}"
ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
mkdir -p "${OUTPUT}"
set -a
# shellcheck disable=SC1091
source /project/peilab/atst/flower/.env
set +a
export WANDB_PROJECT=nimloth-sft2 WANDB_RUN_NAME WANDB_MODE=online
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1

COMMIT=$(git -C "${REPO}" rev-parse HEAD)
if [[ ! -f "${OUTPUT}/README.md" ]]; then
  cat >"${OUTPUT}/README.md" <<EOF
# Formal LeWM-grid decoded-DINO SFT2

- Status: RUNNING
- Git commit: ${COMMIT}
- W&B: nimloth-sft2 / ${WANDB_RUN_NAME}
- SFT1: ${SFT1_CHECKPOINT}
- Data: all 3,217 train trajectories / 59,389 transitions; all 355 val trajectories / 6,054 transitions
- Caches: required compact Qwen cache ${PREPROCESS_CACHE}; required exact float32 DINO 4x4 cache ${DINO_CACHE}; fallback forbidden
- Trainable: LeWM MLP encoder, one bidirectional spatial AdaLN-zero WM, LeWM MLP DINO decoder, value head
- Frozen: SFT1 Qwen/projector and DINOv2 teacher; EMA target encoder decay 0.99
- Loss: latent MSE + 0.5 decoded DINO MSE + 0.1 SIGReg + value MSE
- Budget: 8 GPUs, 10 epochs, batch2/GA4, effective batch64, SIGReg projections1024, 48h
- Resume: automatically load latest completed epoch when present; ID31 checkpoints are forbidden
EOF
fi

resume_args=()
if [[ -f "${OUTPUT}/latest/training_state.pt" ]]; then
  resume_args=(--resume "${OUTPUT}/latest")
fi

cd "${REPO}"
srun "${PY}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/training/sft2/train_grid.py \
  --sft1-checkpoint "${SFT1_CHECKPOINT}" \
  --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output "${OUTPUT}" --dino-model facebook/dinov2-large \
  --preprocess-cache-dir "${PREPROCESS_CACHE}" --require-prebuilt-cache \
  --dino-cache-root "${DINO_CACHE}" --require-dino-cache \
  --epochs 10 --batch-size 2 --grad-accum 4 --num-workers 0 \
  --max-length 12000 --max-pixels 100352 \
  --encoder-lr 1e-4 --wm-lr 3e-4 --decoder-lr 1e-4 --value-lr 1e-4 \
  --weight-decay 0.01 \
  --latent-weight 1.0 --dino-weight 0.5 --sigreg-weight 0.1 --value-weight 1.0 \
  --ema-decay 0.99 --sigreg-num-proj 1024 --sigreg-knots 17 \
  --encoder-hidden-dim 2048 --decoder-hidden-dim 2048 \
  --wm-depth 6 --wm-heads 16 --wm-dim-head 64 --wm-mlp-dim 2048 --wm-dropout 0.1 \
  --seed 42 --wandb-run-name "${WANDB_RUN_NAME}" \
  "${resume_args[@]}"

touch "${OUTPUT}/sft2_done.flag"
