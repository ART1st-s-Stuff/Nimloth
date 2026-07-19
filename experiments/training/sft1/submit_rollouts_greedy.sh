#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/sft1
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft1/slurm"

# Greedy rollout collection (array 0-3). Start env first:
#   ENV_NODE=dgx-13 bash submit_env_external_4gpu.sh
ENV_NODE=${ENV_NODE:-}
NODELIST=${NODELIST:-}

if [ -n "${ENV_NODE}" ]; then
  bash "${SCRIPTDIR}/submit_env_external_4gpu.sh"
fi

SBATCH_ARGS=(--parsable)
if [ "${ROLLOUT_SMOKE:-0}" = "1" ]; then
  SBATCH_ARGS+=(--array=0 --job-name=sft1-rollout-smoke)
elif [ "${ROLLOUT_TRAIN120:-0}" = "1" ]; then
  SBATCH_ARGS+=(--array=0 --job-name=rollout-train120)
fi
if [ -n "${NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

jobid=$("$SLURM" "${SBATCH_ARGS[@]}" "${SCRIPTDIR}/rollouts_greedy_parallel.slurm")
echo "Submitted rollouts_greedy_parallel array job ${jobid}"
