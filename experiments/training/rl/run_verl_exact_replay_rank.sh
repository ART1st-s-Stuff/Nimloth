#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${MODEL:?set MODEL}"
: "${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"
: "${MASTER_ADDR:?set MASTER_ADDR}"
: "${MASTER_PORT:?set MASTER_PORT}"
: "${WANDB_PROJECT:?set WANDB_PROJECT}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME}"
: "${WANDB_RUN_ID:?set WANDB_RUN_ID}"

export RANK=${RANK:-${SLURM_PROCID:?missing RANK/SLURM_PROCID}}
export WORLD_SIZE=${WORLD_SIZE:-${SLURM_NTASKS:?missing WORLD_SIZE/SLURM_NTASKS}}
export LOCAL_RANK=${LOCAL_RANK:-${SLURM_LOCALID:?missing LOCAL_RANK/SLURM_LOCALID}}
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-${SLURM_NTASKS_PER_NODE:-1}}

VERL_ROOT=${REPO}/external/VAGEN/verl
if [[ ! -f "${VERL_ROOT}/verl/__init__.py" ]]; then
  VERL_ROOT=/project/peilab/atst/nimloth/.worktree/vagen-legacy-wm-k8/external/VAGEN/verl
fi
[[ -f "${VERL_ROOT}/verl/__init__.py" ]] || {
  echo "missing VERL source tree" >&2
  exit 2
}

export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${VERL_ROOT}:${REPO}/external/le-wm
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker0,virbr0'
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_DIR=${OUTPUT_DIR}/wandb
mkdir -p "${WANDB_DIR}"

exec /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/rl/run_verl_exact_replay_worker_gate.py" \
  --repo "${REPO}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --model "${MODEL}" \
  --trajectory-jsonl "${TRAJECTORY_JSONL}" \
  --trajectory-index "${TRAJECTORY_INDEX:-0}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-token-length "${MAX_TOKEN_LENGTH:-8192}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"
