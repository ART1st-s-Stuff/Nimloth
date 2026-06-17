#!/bin/bash
#SBATCH --job-name=sft2-smoke
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
SMOKE_OUT=${SCRIPTDIR}/runs/sft2_smoke_alltrain_init_test
LOG=${SMOKE_OUT}/smoke.log

mkdir -p "${SMOKE_OUT}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; export PYTHONPATH=/project/peilab/atst/nimloth/src; source /project/peilab/atst/nimloth/.venv/bin/activate; export TOKENIZERS_PARALLELISM=true'

find_sft1_init_hf() {
  local merged="${SFT1_RUN}/best/hf_merged"
  if [ -f "${merged}/config.json" ]; then
    echo "${merged}"
    return 0
  fi
  local ep dir
  for ep in $(ls -d "${SFT1_RUN}"/epoch_* 2>/dev/null | sort -r); do
    dir="${ep}/hf_merged"
    if [ -f "${dir}/config.json" ]; then
      echo "${dir}"
      return 0
    fi
    if [ -f "${ep}/adapter_config.json" ]; then
      echo "=== smoke merge ${ep} -> hf_merged ===" >>"${LOG}"
      python3 "${SCRIPTDIR}/merge_sft1_lora_ckpt.py" \
        --base-model "${BASE_HF}" \
        --adapter-dir "${ep}" \
        --out-dir "${dir}" >>"${LOG}" 2>&1
      echo "${dir}"
      return 0
    fi
  done
  return 1
}

{
  echo "=== SFT2 smoke (SFT1 LoRA init + e2e) at $(date) ==="
  echo "out=${SMOKE_OUT}"
} | tee "${LOG}"

eval "${COMMON_ENV}"

if [ ! -s "${SCRIPTDIR}/train_sft2_qwen25vl.py" ]; then
  echo "ERROR train_sft2_qwen25vl.py is empty; restore from git HEAD" | tee -a "${LOG}"
  exit 1
fi

SFT1_INIT=$(find_sft1_init_hf) || {
  echo "ERROR no SFT1 hf_merged or LoRA epoch checkpoint found under ${SFT1_RUN}" | tee -a "${LOG}"
  exit 1
}
echo "init_hf=${SFT1_INIT}" | tee -a "${LOG}"

PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/train_sft2_qwen25vl.py" \
  --llm-tune lora \
  --vision-tune freeze \
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

echo "SMOKE_OK" | tee -a "${LOG}"
