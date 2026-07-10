#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
ROOT=${REPO}/experiments/training/sft2
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

: "${PREPROCESS_CACHE_DIR:?PREPROCESS_CACHE_DIR required}"
: "${SFT1_RUN:?SFT1_RUN required}"
: "${BASE_HF:?BASE_HF required}"
: "${RECORDS_ROOT:?RECORDS_ROOT required}"

mkdir -p "${REPO}/outputs/experiments/training/sft2/slurm"
SBATCH_ARGS=(--parsable --export=ALL)
if [ -n "${CACHE_DEPENDENCY:-}" ]; then
  SBATCH_ARGS+=(--dependency="${CACHE_DEPENDENCY}")
fi
jobid=$(${SLURM} "${SBATCH_ARGS[@]}" "${ROOT}/build_compact_cache.slurm")
echo "Submitted compact cache job ${jobid}"
