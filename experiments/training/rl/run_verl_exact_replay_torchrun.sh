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

VERL_ROOT=${REPO}/external/VAGEN/verl
if [[ ! -f "${VERL_ROOT}/verl/__init__.py" ]]; then
  VERL_ROOT=/project/peilab/atst/nimloth/.worktree/vagen-legacy-wm-k8/external/VAGEN/verl
fi
[[ -f "${VERL_ROOT}/verl/__init__.py" ]] || { echo "missing VERL source" >&2; exit 2; }

export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${VERL_ROOT}:${REPO}/external/le-wm
export HOME=/project/peilab/atst/nimloth/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=true
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker0,virbr0'
export NCCL_DEBUG=WARN
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_DIR=${OUTPUT_DIR}/wandb
_REQUESTED_WANDB_PROJECT=${WANDB_PROJECT}
_REQUESTED_WANDB_RUN_NAME=${WANDB_RUN_NAME}
_REQUESTED_WANDB_RUN_ID=${WANDB_RUN_ID}
mkdir -p "${HOME}" "${WANDB_DIR}"
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi
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

# One Slurm task owns all eight GPUs. torchrun children see CUDA ordinals0..7,
# which is required by NCCL peer access on this cluster.
exec /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=8 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
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
