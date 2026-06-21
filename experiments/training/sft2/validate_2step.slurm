#!/bin/bash
#SBATCH --job-name=sft2-val2step
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft2
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
export REPO SCRIPTDIR SFT1_RUN BASE_HF RECORDS_ROOT SFT2_OUTPUTS_ROOT

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "${MODEL_PATH}" ]; then
  bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh"
  # shellcheck disable=SC1091
  source "${SFT1_RUN}/.sft2_init_env"
  MODEL_PATH="${SFT1_BEST}"
fi

LOG_DIR="${SFT2_OUTPUTS_ROOT}/smoke"
mkdir -p "${LOG_DIR}" "${REPO}/outputs/experiments/training/sft2/slurm"
LOG="${LOG_DIR}/validate_2step_${SLURM_JOB_ID:-local}.log"

export PATH="${REPO}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO}/src"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

{
  echo "=== SFT2 trajectory-once 2-step validation at $(date) ==="
  echo "Model: ${MODEL_PATH}"
  echo "Train jsonl: ${RECORDS_ROOT}/train_all.jsonl"
} | tee "${LOG}"

PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/validate_trajectory_once_2step.py" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --record-index "${RECORD_INDEX:-0}" \
  --max-length "${MAX_LENGTH:-12000}" \
  2>&1 | tee -a "${LOG}"

echo "=== validate_2step finished at $(date) ===" | tee -a "${LOG}"
