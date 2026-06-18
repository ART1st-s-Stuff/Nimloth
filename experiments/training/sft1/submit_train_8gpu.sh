#!/usr/bin/env bash
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft1
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft1/slurm"

SFT1_TUNE_MODE=${SFT1_TUNE_MODE:-lora}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft1-alltrain-8gpu-${SFT1_TUNE_MODE}}
NODELIST=${NODELIST:-}

SBATCH_ARGS=(--parsable --export=ALL,SFT1_TUNE_MODE="${SFT1_TUNE_MODE}",WANDB_RUN_NAME="${WANDB_RUN_NAME}")
if [ -n "${NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

jobid=$("$SLURM" "${SBATCH_ARGS[@]}" "${SCRIPTDIR}/train_8gpu.slurm")
echo "Submitted SFT1 train (${SFT1_TUNE_MODE}) job ${jobid}"
