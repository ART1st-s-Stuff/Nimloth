#!/usr/bin/env bash
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
SLURM_BIN=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm"

# Example: submit external 4-GPU env for an existing or new run.
#   RUN_DIR=.../retry2 sbatch --nodelist=dgx-12 env_external_4gpu.slurm
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
NODELIST=${NODELIST:-}

SBATCH_ARGS=(--parsable --export=ALL,RUN_DIR="${RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}")
if [ -n "${NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

jobid=$("$SLURM_BIN"/sbatch "${SBATCH_ARGS[@]}" "${SCRIPTDIR}/env_external_4gpu.slurm")
echo "Submitted env_external_4gpu job ${jobid} for RUN_DIR=${RUN_DIR}"
