#!/usr/bin/env bash

# Run on one node or once per node inside an already-held Slurm allocation.  This script
# intentionally contains only launch/runtime wiring; objective semantics live in
# nimloth.training.sft2.dino_grid and grid modules live in nimloth.wm.grid.

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth-dino-grid}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-/project/peilab/atst/nimloth/.cache}
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/dino_grid_k16_h1_t4.yaml}
MODEL_PATH=${MODEL_PATH:-/project/peilab/atst/nimloth/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged}
TRAIN_JSONL=${TRAIN_JSONL:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data/train_terminal_cot_migrated.jsonl}
VAL_JSONL=${VAL_JSONL:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data/val_terminal_cot_migrated.jsonl}
PREPROCESS_CACHE_DIR_OVERRIDE=${PREPROCESS_CACHE_DIR_OVERRIDE:-}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new SFT2 run directory}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-nimloth-sft2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}

mkdir -p "${OUTPUT_DIR}"
LOG=${OUTPUT_DIR}/train_${SLURM_JOB_ID:-local}_node${NODE_RANK}.log

export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export UV_CACHE_DIR=${RUNTIME_CACHE_ROOT}/uv
export XDG_CACHE_HOME=${RUNTIME_CACHE_ROOT}/xdg
export WANDB_DIR=${RUNTIME_CACHE_ROOT}/wandb
export VIRTUAL_ENV=${PYTHON_ENV}
export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:${PATH}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=true
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi

# The shared credential file also carries defaults for other projects. Restore
# the caller-selected SFT2 identity after loading credentials so runs cannot be
# silently written to that unrelated project.
export WANDB_PROJECT=${WANDB_PROJECT_NAME}
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p "${XDG_CACHE_HOME}" "${WANDB_DIR}"

for required in "${CONFIG}" "${TRAIN_JSONL}" "${VAL_JSONL}"; do
  if [ ! -s "${required}" ]; then
    echo "ERROR required input missing: ${required}" | tee -a "${LOG}"
    exit 1
  fi
done
if [ ! -s "${MODEL_PATH}/config.json" ]; then
  echo "ERROR required model missing: ${MODEL_PATH}" | tee -a "${LOG}"
  exit 1
fi

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARGS=(--resume)
  if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
  fi
fi

CACHE_ARGS=()
if [ -n "${PREPROCESS_CACHE_DIR_OVERRIDE}" ]; then
  CACHE_ARGS=(--preprocess-cache-dir "${PREPROCESS_CACHE_DIR_OVERRIDE}")
fi

{
  echo "=== DINO-grid SFT2 start $(date --iso-8601=seconds) ==="
  echo "commit: $(git -C "${REPO}" rev-parse HEAD)"
  echo "job/node: ${SLURM_JOB_ID:-local}/${SLURM_JOB_NODELIST:-local}"
  echo "config: ${CONFIG}"
  echo "train/val: ${TRAIN_JSONL} / ${VAL_JSONL}"
  echo "preprocess cache override: ${PREPROCESS_CACHE_DIR_OVERRIDE:-config default}"
  echo "output: ${OUTPUT_DIR}"
  echo "model: ${MODEL_PATH}"
  echo "topology: nnodes=${NNODES}; node_rank=${NODE_RANK}; local_ranks=${NPROC_PER_NODE}; world_size=$((NNODES * NPROC_PER_NODE))"
  echo "per-rank B=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM}"
  echo "objective: one current-step CE + four-step recorded-action WM/DINO/value rollout; global SIGReg=0.1"
  echo "trainable: Qwen vision, SFT1 DINO-grid projector, H1 temporal-spatial WM, value head"
  echo "frozen: Qwen LLM, DINO cache/teacher, detached old history"
  echo "initialization: SFT1 Qwen and DINO-grid projector; new H1 WM predictor, ValueHead and optimizer"
} | tee -a "${LOG}"

TORCHRUN_ARGS=(
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
)
if [[ "${NNODES}" == "1" ]]; then
  TORCHRUN_ARGS+=(--standalone)
else
  : "${MASTER_ADDR:?set MASTER_ADDR for multi-node SFT2}"
  : "${MASTER_PORT:?set MASTER_PORT for multi-node SFT2}"
  TORCHRUN_ARGS+=(
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

PYTHONUNBUFFERED=1 "${PYTHON_ENV}/bin/python3" -m torch.distributed.run \
  "${TORCHRUN_ARGS[@]}" \
  "${REPO}/experiments/training/sft2/train.py" \
  --config "${CONFIG}" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  "${CACHE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  ${EXTRA_TRAIN_ARGS:-} \
  2>&1 | tee -a "${LOG}"

if [[ "${NODE_RANK}" == "0" ]]; then
  echo "SFT2_DONE" > "${OUTPUT_DIR}/sft2_done.flag"
fi
echo "=== DINO-grid SFT2 finished $(date --iso-8601=seconds) ===" | tee -a "${LOG}"
