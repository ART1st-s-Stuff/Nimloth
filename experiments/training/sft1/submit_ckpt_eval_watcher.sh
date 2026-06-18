#!/usr/bin/env bash
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft1
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft1/slurm"

: "${TRAIN_OUT:?TRAIN_OUT required}"
: "${BASE_MODEL:?BASE_MODEL required}"

EVAL_NODE=${EVAL_NODE:-}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-alltrain_8gpu_lora}
TRAIN_JOB_ID=${TRAIN_JOB_ID:-}

SBATCH_ARGS=(
  --parsable
  --partition=cpu
  --export=ALL,TRAIN_OUT="${TRAIN_OUT}",BASE_MODEL="${BASE_MODEL}",EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX}",TRAIN_JOB_ID="${TRAIN_JOB_ID}"
)
if [ -n "${EVAL_NODE}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODE}")
fi

jobid=$("$SLURM" "${SBATCH_ARGS[@]}" "${SCRIPTDIR}/ckpt_eval_watcher.slurm")
echo "Submitted ckpt_eval_watcher job ${jobid}"
