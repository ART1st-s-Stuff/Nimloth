#!/bin/bash
#SBATCH --job-name=planner-fsdp-ray-gate
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60

set -euo pipefail

: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
[[ -x "${PYTHON}" ]]
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --short)" ]]
[[ "$(git -C "${REPO}/external/VAGEN" rev-parse HEAD)" == \
  "192c35a91f3941b72d5e1272af6603ef7a7d93e0" ]]
[[ "$(git -C "${REPO}/external/VAGEN/verl" rev-parse HEAD)" == \
  "65316156d1011d71d62e0542e4b954f9499e872e" ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == \
  "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac" ]]
[[ ! -e "${OUTPUT_DIR}" ]]
[[ "${SLURM_JOB_NUM_NODES}" == "1" ]]
[[ "${SLURM_GPUS_ON_NODE}" == "8" ]]

mkdir -p /project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm
REQUESTED_WANDB_PROJECT="${WANDB_PROJECT}"
REQUESTED_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
REQUESTED_WANDB_RUN_ID="${WANDB_RUN_ID}"
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
fi
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
export WANDB_PROJECT="${REQUESTED_WANDB_PROJECT}"
export WANDB_RUN_NAME="${REQUESTED_WANDB_RUN_NAME}"
export WANDB_RUN_ID="${REQUESTED_WANDB_RUN_ID}"

export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export RAY_DEDUP_LOGS=0

cd "${REPO}"
echo "PLANNER_FSDP_RAY_GATE_START commit=${EXPECTED_COMMIT} job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader

srun --nodes=1 --ntasks=1 --gpus-per-task=8 --cpus-per-task=64 \
  --cpu-bind=none --kill-on-bad-exit=1 \
  "${PYTHON}" experiments/training/rl/gate_planner_verl_fsdp_ray.py \
  --output-dir "${OUTPUT_DIR}" \
  --world-size 8 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"

test -s "${OUTPUT_DIR}/result.json"
grep -Fq '"status": "ALL_OK"' "${OUTPUT_DIR}/result.json"
echo "PLANNER_FSDP_RAY_GATE_ALL_OK output=${OUTPUT_DIR}"
