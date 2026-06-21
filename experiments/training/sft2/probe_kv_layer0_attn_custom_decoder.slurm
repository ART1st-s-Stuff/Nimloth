#!/bin/bash
#SBATCH --job-name=qwen-kv-attn0
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
REPO=/project/peilab/atst/nimloth; SCRIPTDIR=${REPO}/experiments/training/sft2
source "${SCRIPTDIR}/common_env.sh"; export REPO SCRIPTDIR SFT1_RUN BASE_HF RECORDS_ROOT SFT2_OUTPUTS_ROOT
MODEL_PATH="${MODEL_PATH:-}"; if [ -z "${MODEL_PATH}" ]; then bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh"; source "${SFT1_RUN}/.sft2_init_env"; MODEL_PATH="${SFT1_BEST}"; fi
LOG_DIR="${SFT2_OUTPUTS_ROOT}/smoke"; mkdir -p "${LOG_DIR}" "${REPO}/outputs/experiments/training/sft2/slurm"
LOG="${LOG_DIR}/probe_kv_layer0_attn_custom_decoder_${SLURM_JOB_ID:-local}.log"
export PATH="${REPO}/.venv/bin:${PATH}" PYTHONPATH="${REPO}/src"
export HF_HOME=/project/peilab/atst/.cache/huggingface TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TRITON_CACHE_DIR=/project/peilab/atst/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/peilab/atst/.cache/torchinductor
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
source "${REPO}/.venv/bin/activate"
echo "=== KV layer0 attn custom decoder probe at $(date) ===" | tee "${LOG}"
PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/probe_kv_layer0_attn_custom_decoder.py" \
  --model "${MODEL_PATH}" --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" --record-index "${RECORD_INDEX:-0}" \
  --max-length "${MAX_LENGTH:-12000}" --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  2>&1 | tee -a "${LOG}"
echo "=== KV layer0 attn custom decoder probe finished at $(date) ===" | tee -a "${LOG}"
