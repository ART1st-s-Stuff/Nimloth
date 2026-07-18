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
export SLURM_TASK_LOCAL_RANK=${SLURM_LOCALID:?missing SLURM_LOCALID}
# srun --gpus-per-task=1 remaps each task's sole GPU to CUDA ordinal zero.
# SLURM_LOCALID remains 0..7 and must not be passed to torch.cuda.set_device.
export LOCAL_RANK=0
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
export HOME=/project/peilab/atst/nimloth/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export WANDB_DIR=${OUTPUT_DIR}/wandb
_REQUESTED_WANDB_PROJECT=${WANDB_PROJECT}
_REQUESTED_WANDB_RUN_NAME=${WANDB_RUN_NAME}
_REQUESTED_WANDB_RUN_ID=${WANDB_RUN_ID}
export TRITON_CACHE_DIR=/tmp/triton_verl_exact_${SLURM_JOB_ID:-local}_$(hostname)
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_verl_exact_${SLURM_JOB_ID:-local}_$(hostname)
mkdir -p "${HOME}" "${WANDB_DIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi
# The shared secret file may contain defaults for another project. Experiment
# identity is an explicit launcher contract and must win over those defaults.
export WANDB_PROJECT=${_REQUESTED_WANDB_PROJECT}
export WANDB_RUN_NAME=${_REQUESTED_WANDB_RUN_NAME}
export WANDB_RUN_ID=${_REQUESTED_WANDB_RUN_ID}
unset _REQUESTED_WANDB_PROJECT _REQUESTED_WANDB_RUN_NAME _REQUESTED_WANDB_RUN_ID

EXTRA_ARGS=(--save-global-step "${SAVE_GLOBAL_STEP:-1}")
if [[ "${WM_AUX_MECHANICS:-0}" = 1 ]]; then
  EXTRA_ARGS+=(--enable-wm-aux-mechanics)
fi
if [[ -n "${RESUME_CHECKPOINT_ROOT:-}" || -n "${RESUME_RESULT:-}" ]]; then
  : "${RESUME_CHECKPOINT_ROOT:?set both resume paths}"
  : "${RESUME_RESULT:?set both resume paths}"
  EXTRA_ARGS+=(
    --resume-checkpoint-root "${RESUME_CHECKPOINT_ROOT}"
    --resume-result "${RESUME_RESULT}"
  )
fi

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
  --wandb-run-id "${WANDB_RUN_ID}" \
  "${EXTRA_ARGS[@]}"
