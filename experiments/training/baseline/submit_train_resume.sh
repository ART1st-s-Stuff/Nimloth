#!/usr/bin/env bash
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
SLURM_BIN=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm"

# Resume training with external env already running (or submit env first).
# Legacy retry2 example:
#   RUN_DIR=${REPO}/experiments/navigation_baseline/runs/vagen_nav_..._retry2 \
#   EXPERIMENT_NAME=vagen_nav_..._retry2 \
#   NODELIST=dgx-32,dgx-37 bash submit_train_resume.sh
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
NODELIST=${NODELIST:-}
TOTAL_STEPS=${TOTAL_STEPS:-100}

SBATCH_ARGS=(--parsable --export=ALL,RUN_DIR="${RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",TOTAL_STEPS="${TOTAL_STEPS}")
if [ -n "${NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

jobid=$("$SLURM_BIN"/sbatch "${SBATCH_ARGS[@]}" "${SCRIPTDIR}/train_resume.slurm")
echo "Submitted train_resume job ${jobid} for RUN_DIR=${RUN_DIR}"
