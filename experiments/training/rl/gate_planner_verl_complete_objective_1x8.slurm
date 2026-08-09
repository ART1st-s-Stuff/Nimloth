#!/bin/bash
#SBATCH --job-name=planner-complete-fsdp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
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

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
CONFIG=${REPO}/configs/training/rl/planner_policy_h1_complete_objective_fsdp_gate.yaml
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

echo "PLANNER_COMPLETE_FSDP_START commit=${EXPECTED_COMMIT} job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader

srun --nodes=1 --ntasks=1 --gpus-per-task=8 --cpus-per-task=64 \
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
  --world-size 8 \
  --minimum-state-tokens 6000 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"

echo "PLANNER_COMPLETE_FSDP_ALL_OK output=${OUTPUT_DIR}"
