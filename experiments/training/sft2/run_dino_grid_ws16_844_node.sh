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

AGENT_RANK=${SLURM_PROCID}
test "${SLURM_NTASKS}" -eq 16
test "${AGENT_RANK}" -ge 0
test "${AGENT_RANK}" -lt 16
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
test -f "${PREPROCESS_CACHE}/cache_done.flag"

test "${CUDA_VISIBLE_DEVICES:?}" != "NoDevFiles"
test "${CUDA_VISIBLE_DEVICES}" != ""
test "${CUDA_VISIBLE_DEVICES}" != *,*
GPU_ROWS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=uuid,name --format=csv,noheader,nounits)
GPU_NAMES=$(printf '%s\n' "${GPU_ROWS}" | cut -d, -f2- | sed 's/^[[:space:]]*//')
GPU_COUNT=$(printf '%s\n' "${GPU_ROWS}" | sed '/^[[:space:]]*$/d' | wc -l)
test "${GPU_COUNT}" -eq 1
test "$(printf '%s\n' "${GPU_NAMES}" | grep -c 'H800')" -eq 1
{
  echo "time=$(date --iso-8601=seconds)"
  echo "job=${SLURM_JOB_ID} host=$(hostname) agent_rank=${AGENT_RANK}"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-missing}"
  echo "local_ranks=0 global_ranks=${AGENT_RANK}"
  echo "gpu_count=${GPU_COUNT}"
  printf '%s\n' "${GPU_ROWS}" | sed 's/^/gpu=/'
} | tee "${RUN_ROOT}/allocation_${SLURM_JOB_ID}_agent${AGENT_RANK}.log"

if [ "${NODE_MODE:-train}" = "probe" ]; then
  exit 0
fi

export PYTHON_ENV=/project/peilab/atst/nimloth/.venv-vagen-main
export RUNTIME_CACHE_ROOT=/project/peilab/atst/nimloth/.cache
export CONFIG="${REPO}/configs/training/sft2/dino_grid_k16_h1_t4.yaml"
export PREPROCESS_CACHE_DIR_OVERRIDE="${PREPROCESS_CACHE}"
export OUTPUT_DIR="${RUN_OUTPUT}"
export WANDB_MODE=online
export NPROC_PER_NODE=1
export NNODES=16
export NODE_RANK=${AGENT_RANK}
export BATCH_SIZE=1
export GRAD_ACCUM=4
export RESUME=0
unset EXTRA_TRAIN_ARGS

exec bash "${REPO}/experiments/training/sft2/train_dino_grid_world8.sh"
