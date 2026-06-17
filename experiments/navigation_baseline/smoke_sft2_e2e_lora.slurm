#!/bin/bash
#SBATCH --job-name=sft2-smoke-lora
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen79_nimloth_format
SFT1_RUN=${SCRIPTDIR}/runs/sft1_train_vagen79_qwen25vl_alltrain_8gpu_lora
BASE_HF=${SCRIPTDIR}/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface
SMOKE_OUT=${SCRIPTDIR}/runs/sft2_smoke_lora_test
LOG=${SMOKE_OUT}/smoke.log

mkdir -p "${SMOKE_OUT}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; export PYTHONPATH=/project/peilab/atst/nimloth/src; source /project/peilab/atst/nimloth/.venv/bin/activate; export TOKENIZERS_PARALLELISM=true'

EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-alltrain_8gpu_lora_cache}

resolve_sft1_init() {
  bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh"
  # shellcheck disable=SC1091
  source "${SFT1_RUN}/.sft2_init_env"
}

{
  echo "=== SFT2 LoRA smoke (SFT1 hf_merged base + --lora) at $(date) ==="
  echo "out=${SMOKE_OUT}"
} | tee "${LOG}"

eval "${COMMON_ENV}"

if [ ! -s "${SCRIPTDIR}/train_sft2_qwen25vl.py" ]; then
  echo "ERROR train_sft2_qwen25vl.py is empty; restore from git" | tee -a "${LOG}"
  exit 1
fi

resolve_sft1_init >>"${LOG}" 2>&1
# shellcheck disable=SC1091
source "${SFT1_RUN}/.sft2_init_env"
SFT1_INIT="${SFT1_BEST}"
echo "init_hf=${SFT1_INIT} adapter=${SFT1_ADAPTER}" | tee -a "${LOG}"

PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/train_sft2_qwen25vl.py" \
  --llm-tune lora \
  --vision-tune freeze \
  --lora-r 8 \
  --lora-alpha 16 \
  --model "${SFT1_INIT}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${SMOKE_OUT}" \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-train-records 2 \
  --max-val-records 2 \
  --max-val-batches 1 \
  --max-length 8000 \
  --max-pixels 602112 \
  --gradient-checkpointing \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

if [ ! -f "${SMOKE_OUT}/best/adapter_config.json" ]; then
  echo "ERROR LoRA smoke missing adapter checkpoint" | tee -a "${LOG}"
  exit 1
fi
echo "SMOKE_LORA_OK" | tee -a "${LOG}"
