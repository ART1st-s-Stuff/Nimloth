#!/bin/bash
#SBATCH --job-name=qwen-imgfeat-probe
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft2
source "${SCRIPTDIR}/common_env.sh"
export REPO SCRIPTDIR SFT1_RUN BASE_HF RECORDS_ROOT SFT2_OUTPUTS_ROOT

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "${MODEL_PATH}" ]; then
  bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh"
  source "${SFT1_RUN}/.sft2_init_env"
  MODEL_PATH="${SFT1_BEST}"
fi

LOG_DIR="${SFT2_OUTPUTS_ROOT}/smoke"
mkdir -p "${LOG_DIR}" "${REPO}/outputs/experiments/training/sft2/slurm"
LOG="${LOG_DIR}/probe_qwen_image_feature_structure_${SLURM_JOB_ID:-local}.log"

export PATH="${REPO}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO}/src"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
source "${REPO}/.venv/bin/activate"

{
  echo "=== qwen image feature structure probe at $(date) ==="
  echo "Model: ${MODEL_PATH}"
  echo "Train jsonl: ${RECORDS_ROOT}/train_all.jsonl"
  echo "Attn implementation: ${ATTN_IMPLEMENTATION:-sdpa}"
} | tee "${LOG}"

PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/probe_qwen_image_feature_structure.py" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --record-index "${RECORD_INDEX:-0}" \
  --max-length "${MAX_LENGTH:-12000}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  2>&1 | tee -a "${LOG}"

echo "=== qwen image feature structure probe finished at $(date) ===" | tee -a "${LOG}"
