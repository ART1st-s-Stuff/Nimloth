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

# Equal 2-GPU fragments use one regular four-node batch job. The site submit
# plugin caps later heterogeneous components at 8h, so do not use `sbatch :`.
job_id=$(
  "${SBATCH}" --parsable --hold \
    --job-name=sft2-dino2-w8 \
    --account=peilab --partition=normal \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --gres=gpu:2 --cpus-per-task=16 --mem=96G \
    --time=2-00:00:00 --requeue \
    "${REPO}/experiments/training/sft2/train_dino_world8_4x2.slurm"
)
job_id=${job_id%%;*}
trap '"${SCANCEL}" "${job_id}" >/dev/null 2>&1 || true' ERR

details=$("${SCONTROL}" show job "${job_id}" -o)
time_limit=$(tr " " "\n" <<<"${details}" | awk -F= '$1=="TimeLimit" {print $2; exit}')
num_nodes=$(tr " " "\n" <<<"${details}" | awk -F= '$1=="NumNodes" {print $2; exit}')
req_tres=$(tr " " "\n" <<<"${details}" | awk -F= '$1=="ReqTRES" {print $2; exit}')
if [[ "${time_limit}" != "2-00:00:00" || "${num_nodes}" != "4" || "${req_tres}" != *"gres/gpu=8"* ]]; then
  echo "unsafe allocation request: TimeLimit=${time_limit} NumNodes=${num_nodes} ReqTRES=${req_tres}" >&2
  "${SCANCEL}" "${job_id}" || true
  exit 5
fi
"${SCONTROL}" release "${job_id}"
trap - ERR
printf 'dino SFT2 job: %s (4 nodes x 2 GPU; verified 48h and released)\n' "${job_id}"
printf 'run: %s\n' "${OUT_ROOT}"
