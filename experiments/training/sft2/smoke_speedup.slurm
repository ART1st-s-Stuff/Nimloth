#!/bin/bash
#SBATCH --job-name=sft2-smoke
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft2
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "${MODEL_PATH}" ]; then
  bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh"
  # shellcheck disable=SC1091
  source "${SFT1_RUN}/.sft2_init_env"
  MODEL_PATH="${SFT1_BEST}"
fi

LOG_DIR="${SFT2_OUTPUTS_ROOT}/smoke"
mkdir -p "${LOG_DIR}" "${REPO}/outputs/experiments/training/sft2/slurm"
LOG="${LOG_DIR}/smoke_speedup_${SLURM_JOB_ID:-local}.log"

COMMON_ENV='export PATH=/project/peilab/atst/nimloth/.venv/bin:$PATH; export PYTHONPATH=/project/peilab/atst/nimloth/src; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; source /project/peilab/atst/nimloth/.venv/bin/activate'

eval "${COMMON_ENV}"

{
  echo "=== SFT2 speedup smoke at $(date) ==="
  echo "Model: ${MODEL_PATH}"
  echo "Train jsonl: ${RECORDS_ROOT}/train_all.jsonl"
} | tee "${LOG}"

PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/smoke_speedup.py" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --max-train-records "${MAX_TRAIN_RECORDS:-4}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --max-micro-steps "${MAX_MICRO_STEPS:-1}" \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "=== smoke finished at $(date) ===" | tee -a "${LOG}"
