#!/usr/bin/env bash
#SBATCH --job-name=sft1-v2-gated
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Partition/GRES/CPU/memory/walltime/output are supplied only by the separately
# approved sbatch command; this file does not reserve resources by itself.
set -euo pipefail

: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PY:?PY must be the exact .venv-vagen-main/bin/python3}"
: "${CONFIG:?resolved launch-locked CONFIG is required}"
: "${PHASE:?PHASE must be cache|smoke|resume-smoke|formal|validate-report}"
: "${CONTROLLER_LOG_DIR:?controller logs must be outside protected run directories}"
: "${CONTROLLER_DIR:?durable controller transaction directory is required}"
: "${MASTER_PORT:?unique MASTER_PORT is required}"
: "${TMPDIR:?unique TMPDIR is required}"
: "${HF_HOME:?explicit offline HF_HOME is required}"
: "${TRANSFORMERS_CACHE:?explicit offline TRANSFORMERS_CACHE is required}"
: "${WORLD_SIZE:?approved one-node world size is required}"
: "${SEED:?approved experiment seed is required}"
: "${SLURM_JOB_ID:?this wrapper must run inside the approved Slurm allocation}"
[[ "${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}" == 1 ]] || {
  echo "SFT1-v2 currently requires one physical node" >&2; exit 64;
}

case "${PHASE}" in
  cache|smoke|resume-smoke|formal|validate-report) ;;
  *) echo "invalid PHASE=${PHASE}" >&2; exit 64 ;;
esac
[[ "${PY}" == */.venv-vagen-main/bin/python3 ]] || {
  echo "PY must be the explicit .venv-vagen-main/bin/python3" >&2; exit 64;
}
[[ "$(${PY} -c 'import sys; print(sys.executable)')" == "${PY}" ]] || {
  echo "interpreter identity mismatch" >&2; exit 64;
}
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "EXPECTED_COMMIT mismatch" >&2; exit 64;
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)" ]] || {
  echo "dirty parent worktree" >&2; exit 64;
}
git -C "${REPO}" submodule foreach --recursive --quiet \
  'test -z "$(git status --porcelain --untracked-files=all)"' || {
  echo "dirty initialized submodule" >&2; exit 64;
}
[[ "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid WORLD_SIZE" >&2; exit 64; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CUDA_VISIBLE_DEVICES is empty" >&2; exit 64; }
VISIBLE_COUNT=$(${PY} - <<'PY'
import os
print(len([item for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]))
PY
)
[[ "${VISIBLE_COUNT}" == "${WORLD_SIZE}" ]] || {
  echo "CUDA visibility/world-size mismatch ${VISIBLE_COUNT} != ${WORLD_SIZE}" >&2; exit 64;
}
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ -n "${GPU_NAMES}" ]] && ! grep -vq 'H800' <<<"${GPU_NAMES}" || {
  echo "approved SFT1-v2 topology requires only H800 GPUs" >&2; exit 64;
}

mkdir -p "${CONTROLLER_LOG_DIR}" "${CONTROLLER_DIR}" "${TMPDIR}"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "world_size=${WORLD_SIZE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  scontrol show job "${SLURM_JOB_ID}" -dd
} >"${CONTROLLER_LOG_DIR}/${PHASE}.allocation.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  >"${CONTROLLER_LOG_DIR}/${PHASE}.gpu_map.csv"
(
  while true; do
    date --iso-8601=seconds
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
    sleep 30
  done
) >"${CONTROLLER_LOG_DIR}/${PHASE}.nvidia_smi.csv" 2>&1 &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1

CONTROLLER=(
  "${PY}" "${REPO}/experiments/training/sft1/state_interface_v2_controller.py"
  --config "${CONFIG}" --phase "${PHASE}" --repo-root "${REPO}"
  --controller-dir "${CONTROLLER_DIR}"
)
if [[ "${PHASE}" == cache && "${RESUME_CACHE:-0}" == 1 ]] || \
   [[ "${PHASE}" == formal && -n "${RESUME_CHECKPOINT:-}" ]]; then
  CONTROLLER+=(--resume)
fi
TORCHRUN=(
  "${PY}" -m torch.distributed.run
  --nnodes=1 --nproc-per-node="${WORLD_SIZE}"
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}"
)

case "${PHASE}" in
  cache)
    export WANDB_MODE=disabled
    if [[ "${CACHE_EVIDENCE_ONLY:-0}" == 1 ]]; then
      [[ "${RESUME_CACHE:-0}" == 1 ]] || {
        echo "CACHE_EVIDENCE_ONLY requires RESUME_CACHE=1" >&2; exit 64;
      }
      COMMAND=("${PY}" "${REPO}/experiments/training/sft1/state_interface_v2_cache.py" --config "${CONFIG}" --repo-root "${REPO}" --finalize-evidence)
    else
      COMMAND=("${TORCHRUN[@]}" "${REPO}/experiments/training/sft1/state_interface_v2_cache.py" --config "${CONFIG}" --repo-root "${REPO}")
    fi
    ;;
  smoke)
    export WANDB_MODE=disabled
    COMMAND=("${TORCHRUN[@]}" "${REPO}/experiments/training/sft1/state_interface_v2_train.py" --config "${CONFIG}" --repo-root "${REPO}" --mode smoke --seed "${SEED}")
    ;;
  resume-smoke)
    : "${RESUME_CHECKPOINT:?resume-smoke requires RESUME_CHECKPOINT}"
    export WANDB_MODE=disabled
    COMMAND=("${TORCHRUN[@]}" "${REPO}/experiments/training/sft1/state_interface_v2_train.py" --config "${CONFIG}" --repo-root "${REPO}" --mode resume-smoke --seed "${SEED}" --resume-checkpoint "${RESUME_CHECKPOINT}")
    ;;
  formal)
    : "${WANDB_ENV_FILE:?formal phase requires WANDB_ENV_FILE}"
    # shellcheck disable=SC1090
    set -a; source "${WANDB_ENV_FILE}"; set +a
    export WANDB_MODE=online
    COMMAND=("${TORCHRUN[@]}" "${REPO}/experiments/training/sft1/state_interface_v2_train.py" --config "${CONFIG}" --repo-root "${REPO}" --mode formal --seed "${SEED}")
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
      COMMAND+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
    fi
    ;;
  validate-report)
    COMMAND=()
    ;;
esac

# The controller performs the same fail-closed preflight immediately before
# executing the exact command. A separate login/CPU --preflight-only invocation
# is required and recorded before sbatch allocation.
if ((${#COMMAND[@]})); then
  "${CONTROLLER[@]}" -- "${COMMAND[@]}" \
    >"${CONTROLLER_LOG_DIR}/${PHASE}.controller.out" \
    2>"${CONTROLLER_LOG_DIR}/${PHASE}.controller.err"
else
  "${CONTROLLER[@]}" \
    >"${CONTROLLER_LOG_DIR}/${PHASE}.controller.out" \
    2>"${CONTROLLER_LOG_DIR}/${PHASE}.controller.err"
fi
