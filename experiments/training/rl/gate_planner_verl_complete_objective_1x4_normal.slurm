#!/bin/bash
#SBATCH --job-name=planner-complete-fsdp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/planner-complete-fsdp_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/planner-complete-fsdp_%j.err

set -euo pipefail

: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${MODEL:?MODEL is required}"
: "${WM_CHECKPOINT:?WM_CHECKPOINT is required}"
: "${STATE_PROJ_CHECKPOINT:?STATE_PROJ_CHECKPOINT is required}"
: "${VALUE_HEAD_CHECKPOINT:?VALUE_HEAD_CHECKPOINT is required}"
: "${PLANNER_POLICY_HEAD_CHECKPOINT:?PLANNER_POLICY_HEAD_CHECKPOINT is required}"
: "${TRAJECTORY_JSONL:?TRAJECTORY_JSONL is required}"
: "${FRESH_ROLLOUT_MANIFEST:?FRESH_ROLLOUT_MANIFEST is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"
GATE_MINIMUM_STATE_TOKENS=${GATE_MINIMUM_STATE_TOKENS:-6000}
GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX=${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX:-false}
HOLD_READY_FILE=${HOLD_READY_FILE:-}
HOLD_GO_FILE=${HOLD_GO_FILE:-}
HOLD_AUTO_RELEASE_SECONDS=${HOLD_AUTO_RELEASE_SECONDS:-}
if [[ -n "${HOLD_READY_FILE}" || -n "${HOLD_GO_FILE}" ]]; then
  : "${HOLD_READY_FILE:?both hold handshake files are required}"
  : "${HOLD_GO_FILE:?both hold handshake files are required}"
  [[ ! -e "${HOLD_READY_FILE}" && ! -e "${HOLD_GO_FILE}" ]]
  mkdir -p "$(dirname "${HOLD_READY_FILE}")"
  printf '%s\n' "job=${SLURM_JOB_ID} node=$(hostname) state=held" \
    > "${HOLD_READY_FILE}.tmp"
  mv "${HOLD_READY_FILE}.tmp" "${HOLD_READY_FILE}"
  trap 'rm -f "${HOLD_READY_FILE}"' EXIT
  if [[ -n "${HOLD_AUTO_RELEASE_SECONDS}" ]]; then
    [[ "${HOLD_AUTO_RELEASE_SECONDS}" =~ ^[0-9]+$ ]]
    [[ "$(nvidia-smi -L | wc -l)" -eq 4 ]]
    sleep "${HOLD_AUTO_RELEASE_SECONDS}"
    printf '%s\n' "job=${SLURM_JOB_ID} state=auto_released" \
      > "${HOLD_GO_FILE}.tmp"
    mv "${HOLD_GO_FILE}.tmp" "${HOLD_GO_FILE}"
  fi
  while [[ ! -e "${HOLD_GO_FILE}" ]]; do
    sleep 1
  done
  rm -f "${HOLD_GO_FILE}" "${HOLD_READY_FILE}"
  trap - EXIT
fi
GATE_EXPECTED_MANIFEST_SHA256=${GATE_EXPECTED_MANIFEST_SHA256:-}
GATE_EXPECTED_TRAJECTORY_COUNT=${GATE_EXPECTED_TRAJECTORY_COUNT:-}
GATE_EXPECTED_SELECTED_RECORD_ID=${GATE_EXPECTED_SELECTED_RECORD_ID:-}
GATE_EXPECTED_SELECTED_STATE_TOKENS=${GATE_EXPECTED_SELECTED_STATE_TOKENS:-}
GATE_EXPECTED_CURRENT_POLICY_SHA256=${GATE_EXPECTED_CURRENT_POLICY_SHA256:-}
GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256=${GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256:-}
GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256=${GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256:-}
GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256=${GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256:-}
GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256=${GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256:-}
if [[ "${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX}" == true ]]; then
  (( GATE_MINIMUM_STATE_TOKENS >= 14000 )) || {
    echo "nonbehavior diagnostic requires >=14000 real state tokens" >&2
    exit 2
  }
  : "${GATE_EXPECTED_MANIFEST_SHA256:?diagnostic manifest SHA256 is required}"
  : "${GATE_EXPECTED_TRAJECTORY_COUNT:?diagnostic trajectory count is required}"
  : "${GATE_EXPECTED_SELECTED_RECORD_ID:?diagnostic selected record is required}"
  : "${GATE_EXPECTED_SELECTED_STATE_TOKENS:?diagnostic selected tokens are required}"
  : "${GATE_EXPECTED_CURRENT_POLICY_SHA256:?current policy SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256:?current StateProjector SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256:?current WM SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256:?current ValueHead SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256:?current PlannerPolicyHead SHA256 is required}"
elif [[ "${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX}" != false ]]; then
  echo "GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX must be true or false" >&2
  exit 2
fi

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
CONFIG=${REPO}/configs/training/rl/planner_policy_h1_complete_objective_fsdp_gate_1x4.yaml
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm

REQUESTED_WANDB_PROJECT="${WANDB_PROJECT}"
REQUESTED_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
REQUESTED_WANDB_RUN_ID="${WANDB_RUN_ID}"
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
export WANDB_PROJECT="${REQUESTED_WANDB_PROJECT}"
export WANDB_RUN_NAME="${REQUESTED_WANDB_RUN_NAME}"
export WANDB_RUN_ID="${REQUESTED_WANDB_RUN_ID}"

cd "${REPO}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git status --short)" ]]
[[ "$(git -C external/VAGEN rev-parse HEAD)" == "192c35a91f3941b72d5e1272af6603ef7a7d93e0" ]]
[[ "$(git -C external/VAGEN/verl rev-parse HEAD)" == "65316156d1011d71d62e0542e4b954f9499e872e" ]]
[[ "$(git -C external/le-wm rev-parse HEAD)" == "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac" ]]
[[ ! -e "${OUTPUT_DIR}" ]]
[[ -x "${PYTHON}" ]]
[[ -f "${CONFIG}" ]]
[[ -d "${MODEL}" ]]
[[ -d "${WM_CHECKPOINT}" ]]
[[ -f "${STATE_PROJ_CHECKPOINT}" ]]
[[ -d "${VALUE_HEAD_CHECKPOINT}" ]]
[[ -d "${PLANNER_POLICY_HEAD_CHECKPOINT}" ]]
[[ -f "${TRAJECTORY_JSONL}" ]]
[[ -f "${FRESH_ROLLOUT_MANIFEST}" ]]
[[ ! -e "${FRESH_ROLLOUT_MANIFEST}.consumption.json" ]]

export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

echo "PLANNER_COMPLETE_FSDP_START commit=${EXPECTED_COMMIT} job=${SLURM_JOB_ID} node=$(hostname) diagnostic_nonbehavior=${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX} minimum_tokens=${GATE_MINIMUM_STATE_TOKENS}"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader

GATE_EXTRA_ARGS=()
if [[ "${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX}" == true ]]; then
  GATE_EXTRA_ARGS+=(
    --diagnostic-nonbehavior-prefix
    --expected-manifest-sha256 "${GATE_EXPECTED_MANIFEST_SHA256}"
    --expected-trajectory-count "${GATE_EXPECTED_TRAJECTORY_COUNT}"
    --expected-selected-record-id "${GATE_EXPECTED_SELECTED_RECORD_ID}"
    --expected-selected-state-tokens "${GATE_EXPECTED_SELECTED_STATE_TOKENS}"
    --expected-current-policy-sha256 "${GATE_EXPECTED_CURRENT_POLICY_SHA256}"
    --expected-current-state-projector-sha256 "${GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256}"
    --expected-current-wm-predictor-sha256 "${GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256}"
    --expected-current-value-head-sha256 "${GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256}"
    --expected-current-planner-policy-head-sha256 "${GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256}"
  )
fi

srun --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 \
  "${PYTHON}" experiments/training/rl/gate_planner_verl_complete_objective.py \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --state-proj-checkpoint "${STATE_PROJ_CHECKPOINT}" \
  --value-head-checkpoint "${VALUE_HEAD_CHECKPOINT}" \
  --planner-policy-head-checkpoint "${PLANNER_POLICY_HEAD_CHECKPOINT}" \
  --trajectory-jsonl "${TRAJECTORY_JSONL}" \
  --fresh-rollout-manifest "${FRESH_ROLLOUT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --world-size 4 \
  --minimum-state-tokens "${GATE_MINIMUM_STATE_TOKENS}" \
  "${GATE_EXTRA_ARGS[@]}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"

echo "PLANNER_COMPLETE_FSDP_ALL_OK output=${OUTPUT_DIR}"
