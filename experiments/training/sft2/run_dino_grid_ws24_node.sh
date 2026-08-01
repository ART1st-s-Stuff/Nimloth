#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?}"
: "${EXPECTED_COMMIT:?}"
: "${RUN_ROOT:?}"
: "${RUN_OUTPUT:?}"
: "${MODEL_PATH:?}"
: "${TRAIN_JSONL:?}"
: "${VAL_JSONL:?}"
: "${PREPROCESS_CACHE:?}"
: "${WANDB_ENTITY:?}"
: "${WANDB_PROJECT_NAME:?}"
: "${WANDB_RUN_NAME:?}"
: "${WANDB_RUN_ID:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"
: "${SLURM_PROCID:?}"

NODE_RANK=${SLURM_PROCID}
test "${SLURM_NTASKS}" -eq 6
test "${NODE_RANK}" -ge 0
test "${NODE_RANK}" -lt 6
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
test -f "${PREPROCESS_CACHE}/cache_done.flag"

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_COUNT=$(printf '%s\n' "${GPU_NAMES}" | sed '/^[[:space:]]*$/d' | wc -l)
test "${GPU_COUNT}" -eq 4
test "$(printf '%s\n' "${GPU_NAMES}" | grep -c 'H800')" -eq 4
GLOBAL_FIRST=$((NODE_RANK * 4))
GLOBAL_LAST=$((GLOBAL_FIRST + 3))
{
  echo "time=$(date --iso-8601=seconds)"
  echo "job=${SLURM_JOB_ID} host=$(hostname) node_rank=${NODE_RANK}"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-missing}"
  echo "local_ranks=0-3 global_ranks=${GLOBAL_FIRST}-${GLOBAL_LAST}"
  echo "gpu_count=${GPU_COUNT}"
  printf 'gpu_name=%s\n' "${GPU_NAMES}"
} | tee "${RUN_ROOT}/allocation_${SLURM_JOB_ID}_node${NODE_RANK}.log"

export PYTHON_ENV=/project/peilab/atst/nimloth/.venv-vagen-main
export RUNTIME_CACHE_ROOT=/project/peilab/atst/nimloth/.cache
export CONFIG="${REPO}/configs/training/sft2/dino_grid_k16_h1_t4.yaml"
export PREPROCESS_CACHE_DIR_OVERRIDE="${PREPROCESS_CACHE}"
export OUTPUT_DIR="${RUN_OUTPUT}"
export WANDB_MODE=online
export NPROC_PER_NODE=4
export NNODES=6
export NODE_RANK
export BATCH_SIZE=1
export GRAD_ACCUM=4
export RESUME=0
unset EXTRA_TRAIN_ARGS

exec bash "${REPO}/experiments/training/sft2/train_dino_grid_world8.sh"
