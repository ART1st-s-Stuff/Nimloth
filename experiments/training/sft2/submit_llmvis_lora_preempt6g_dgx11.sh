#!/bin/bash
# Preempt queue, 6 GPU — LLM+Vision LoRA.
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/sft2
NB=${REPO}/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

SFT1_EPOCH=${SFT1_EPOCH:-2}
SFT2_NODE=${SFT2_NODE:-dgx-11}
NGPUS=${NGPUS:-6}
TRAIN_OUT=${TRAIN_OUT:-${NB}/runs/sft2_latentwm_llmvis_lora_preempt6g_dgx11}

echo "=== Submit SFT2 preempt 6GPU dgx-11 ==="

J=$($SLURM --account=peilab --partition=preempt --nodelist="${SFT2_NODE}" \
  --gres=gpu:${NGPUS} --mem=600G \
  --job-name="sft2-lmvis-pre6g" \
  --export=ALL,SFT2_LLM_TUNE=lora,SFT2_VISION_TUNE=lora,NGPUS="${NGPUS}",SFT1_EPOCH="${SFT1_EPOCH}",SKIP_SFT1_DONE=1,TRAIN_OUT_OVERRIDE="${TRAIN_OUT}" \
  "${ROOT}/train_vagen79_default.slurm" | awk '{print $NF}')
echo "sft2 job: ${J}"
echo "log: ${TRAIN_OUT}/sft2_train_${J}.log"
