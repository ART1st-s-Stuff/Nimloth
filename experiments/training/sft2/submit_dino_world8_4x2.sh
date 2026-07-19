#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-query-align}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
OUT_ROOT=${OUT_ROOT:-${ROOT}/sft2/${RUN_NAME}}
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export REPO RUN_NAME ROOT OUT_ROOT
mkdir -p "${OUT_ROOT}/logs" "${ROOT}/sft2" \
  /project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm

# Put every resource and walltime option in every heterogeneous component.
# This avoids later components silently inheriting the partition's 8h default.
component=(
  --job-name=sft2-dino2-w8
  --account=peilab
  --partition=normal
  --nodes=1
  --ntasks=1
  --gres=gpu:2
  --cpus-per-task=16
  --mem=96G
  --time=2-00:00:00
  --requeue
)

job_id=$(
  "${SBATCH}" --parsable \
    "${component[@]}" : \
    "${component[@]}" : \
    "${component[@]}" : \
    "${component[@]}" \
    "${REPO}/experiments/training/sft2/train_dino_world8_4x2.slurm"
)
printf 'dino SFT2 heterogeneous job: %s\n' "${job_id}"
printf 'run: %s\n' "${OUT_ROOT}"
