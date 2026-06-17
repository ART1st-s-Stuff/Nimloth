#!/bin/bash
# LLM LoRA + Vision LoRA variant (overrides yaml defaults).
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/sft2
NB=${REPO}/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

SFT1_EPOCH=${SFT1_EPOCH:-2}
SFT2_NODE=${SFT2_NODE:-dgx-52}
NGPUS=${NGPUS:-8}
TRAIN_OUT=${TRAIN_OUT:-${NB}/runs/sft2_latentwm_llmvis_lora_8gpu}
PARTITION=${PARTITION:-normal}

echo "=== Submit SFT2 LLM+Vision LoRA (${NGPUS} GPU, ${PARTITION}) ==="

J=$($SLURM --account=peilab --partition="${PARTITION}" --nodelist="${SFT2_NODE}" \
  --job-name="sft2-lmvis-lora" \
  --export=ALL,SFT2_LLM_TUNE=lora,SFT2_VISION_TUNE=lora,NGPUS="${NGPUS}",SFT1_EPOCH="${SFT1_EPOCH}",SKIP_SFT1_DONE=1,TRAIN_OUT_OVERRIDE="${TRAIN_OUT}" \
  "${ROOT}/train_vagen79_default.slurm" | awk '{print $NF}')
echo "sft2 job: ${J}"
echo "log: ${TRAIN_OUT}/sft2_train_${J}.log"
