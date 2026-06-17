#!/bin/bash
# SFT2 LoRA e2e from SFT1 epoch_002 (does not wait for full SFT1 training).
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

SFT1_EPOCH=${SFT1_EPOCH:-2}
SFT2_NODE=${SFT2_NODE:-dgx-46}
TRAIN_OUT=${TRAIN_OUT:-${ROOT}/runs/sft2_e2e_vagen79_from_epoch${SFT1_EPOCH}_lora}

echo "=== Submit SFT2 LoRA from SFT1 epoch_${SFT1_EPOCH} ==="
echo "init: epoch_$(printf '%03d' "${SFT1_EPOCH}")/hf_merged"
echo "output: ${TRAIN_OUT}"
echo "node: ${SFT2_NODE}"

J=$($SLURM --account=peilab --nodelist="${SFT2_NODE}" \
  --job-name="sft2-ep${SFT1_EPOCH}-lora" \
  --export=ALL,SFT2_MODE=lora,SFT1_EPOCH="${SFT1_EPOCH}",SKIP_SFT1_DONE=1,TRAIN_OUT_OVERRIDE="${TRAIN_OUT}" \
  "${ROOT}/train_sft2_vagen79_e2e_alltrain8gpu.slurm" | awk '{print $NF}')
echo "sft2 job: ${J}"
echo "log: ${TRAIN_OUT}/sft2_e2e_alltrain8gpu_${J}.log"
