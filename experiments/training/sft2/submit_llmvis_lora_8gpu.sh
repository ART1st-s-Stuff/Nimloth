#!/bin/bash
# LLM LoRA + Vision LoRA variant (overrides yaml defaults).
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/sft2
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft2/slurm"

SFT1_EPOCH=${SFT1_EPOCH:-2}
SFT2_NODE=${SFT2_NODE:-}
NGPUS=${NGPUS:-8}
PARTITION=${PARTITION:-normal}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft2_latentwm_llmvis_lora_8gpu}
TRAIN_OUT=${TRAIN_OUT:-${REPO}/outputs/experiments/training/sft2/${RUN_DATE}/${EXPERIMENT_NAME}}

echo "=== Submit SFT2 LLM+Vision LoRA (${NGPUS} GPU, ${PARTITION}) ==="
echo "output: ${TRAIN_OUT}"

SBATCH_ARGS=(--account=peilab --partition="${PARTITION}" --job-name="sft2-lmvis-lora")
if [ -n "${SFT2_NODE}" ]; then
  SBATCH_ARGS+=(--nodelist="${SFT2_NODE}")
fi

J=$($SLURM "${SBATCH_ARGS[@]}" \
  --export=ALL,SFT2_LLM_TUNE=lora,SFT2_VISION_TUNE=lora,NGPUS="${NGPUS}",SFT1_EPOCH="${SFT1_EPOCH}",SKIP_SFT1_DONE=1,TRAIN_OUT_OVERRIDE="${TRAIN_OUT}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",RUN_DATE="${RUN_DATE}" \
  "${ROOT}/train_vagen79_default.slurm" | awk '{print $NF}')
echo "sft2 job: ${J}"
echo "log: ${TRAIN_OUT}/sft2_train_${J}.log"
