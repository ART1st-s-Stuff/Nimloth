#!/usr/bin/env bash
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft1
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft1/slurm"

ENV_NODE=${ENV_NODE:-}
PORT_BASE=${PORT_BASE:-8500}

SBATCH_ARGS=(--parsable --export=ALL,PORT_BASE="${PORT_BASE}")
if [ -n "${ENV_NODE}" ]; then
  SBATCH_ARGS+=(--nodelist="${ENV_NODE}")
fi

jobid=$("$SLURM" "${SBATCH_ARGS[@]}" "${SCRIPTDIR}/env_external_4gpu.slurm")
echo "Submitted env_external_4gpu job ${jobid}"
