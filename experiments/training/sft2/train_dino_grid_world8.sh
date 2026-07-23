#!/usr/bin/env bash

# Run inside an already-held one-node/eight-GPU Slurm allocation.  This script
# intentionally contains only launch/runtime wiring; objective semantics live in
# nimloth.training.sft2.dino_grid and grid modules live in nimloth.wm.grid.

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth-dino-grid}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/dino_grid_k16_h4.yaml}
TRAIN_JSONL=${TRAIN_JSONL:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c/train_all.jsonl}
VAL_JSONL=${VAL_JSONL:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c/val_all.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new SFT2 run directory}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

mkdir -p "${OUTPUT_DIR}"
LOG=${OUTPUT_DIR}/train_${SLURM_JOB_ID:-local}.log

export HOME=${REPO}/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export UV_CACHE_DIR=${REPO}/.cache/uv
export VIRTUAL_ENV=${PYTHON_ENV}
export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:${PATH}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=${WANDB_PROJECT:-nimloth-sft2}
export WANDB_MODE=${WANDB_MODE:-online}

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi

for required in "${CONFIG}" "${TRAIN_JSONL}" "${VAL_JSONL}"; do
  if [ ! -s "${required}" ]; then
    echo "ERROR required input missing: ${required}" | tee -a "${LOG}"
    exit 1
  fi
done

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARGS=(--resume)
  if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
  fi
fi

{
  echo "=== DINO-grid SFT2 start $(date --iso-8601=seconds) ==="
  echo "commit: $(git -C "${REPO}" rev-parse HEAD)"
  echo "job/node: ${SLURM_JOB_ID:-local}/${SLURM_JOB_NODELIST:-local}"
  echo "config: ${CONFIG}"
  echo "train/val: ${TRAIN_JSONL} / ${VAL_JSONL}"
  echo "output: ${OUTPUT_DIR}"
  echo "world size: ${NPROC_PER_NODE}; per-rank B=1; grad_accum=8"
  echo "objective: one current-step CE + latent WM + 0.5 decoded DINO grid + value; global SIGReg=0.1"
  echo "trainable: Qwen vision, online grid encoder, H4 temporal-spatial WM, DINO decoder, value head"
  echo "frozen: Qwen LLM, SFT1 slot projector, DINO cache/teacher, EMA target encoder, detached old history"
  echo "initialization: ID33 auxiliary warm start plus zero temporal_position; new optimizer, not resume"
} | tee -a "${LOG}"

PYTHONUNBUFFERED=1 "${PYTHON_ENV}/bin/python3" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO}/experiments/training/sft2/train.py" \
  --config "${CONFIG}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  "${RESUME_ARGS[@]}" \
  ${EXTRA_TRAIN_ARGS:-} \
  2>&1 | tee -a "${LOG}"

echo "SFT2_DONE" > "${OUTPUT_DIR}/sft2_done.flag"
echo "=== DINO-grid SFT2 finished $(date --iso-8601=seconds) ===" | tee -a "${LOG}"
