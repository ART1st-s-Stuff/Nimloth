#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-query-align}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
OUT_ROOT=${OUT_ROOT:-${ROOT}/sft2/${RUN_NAME}}
SLURM_BIN=/cm/shared/apps/slurm/current/bin
SBATCH=${SLURM_BIN}/sbatch
SCONTROL=${SLURM_BIN}/scontrol
SCANCEL=${SLURM_BIN}/scancel
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
  --hold
)

job_id=$(
  "${SBATCH}" --parsable \
    "${component[@]}" : \
    "${component[@]}" : \
    "${component[@]}" : \
    "${component[@]}" \
    "${REPO}/experiments/training/sft2/train_dino_world8_4x2.slurm"
)
job_id=${job_id%%;*}
trap '"${SCANCEL}" "${job_id}" >/dev/null 2>&1 || true' ERR
component_ids=()
for group in 0 1 2 3; do
  component="${job_id}+${group}"
  details=$("${SCONTROL}" show job "${component}" -o)
  actual_id=$(tr " " "\n" <<<"${details}" | awk -F= '$1=="JobId" {print $2; exit}')
  "${SCONTROL}" update JobId="${actual_id}" TimeLimit=2-00:00:00
  details=$("${SCONTROL}" show job "${actual_id}" -o)
  actual_limit=$(tr " " "\n" <<<"${details}" | awk -F= '$1=="TimeLimit" {print $2; exit}')
  if [[ "${actual_limit}" != "2-00:00:00" ]]; then
    echo "component ${component} has unsafe TimeLimit=${actual_limit}; cancelling ${job_id}" >&2
    "${SCANCEL}" "${job_id}" || true
    exit 5
  fi
  component_ids+=("${actual_id}")
done
for component_id in "${component_ids[@]}"; do
  "${SCONTROL}" release "${component_id}"
done
trap - ERR
printf 'dino SFT2 heterogeneous job: %s (components %s; verified 48h and released)\n' \
  "${job_id}" "${component_ids[*]}"
printf 'run: %s\n' "${OUT_ROOT}"
