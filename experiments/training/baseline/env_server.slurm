#!/bin/bash
#SBATCH --job-name=vagen-nav-env
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --exclude=dgx-03,dgx-16,dgx-20,dgx-23,dgx-31,dgx-34,dgx-47
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G

set -e

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
RUNTIME_DIR=${RUNTIME_DIR:-${REPO}/outputs/experiments/training/baseline/runtime}

PORT=${1:-8000}
MAX_ENVS=${2:-64}
THREAD_POOL=${3:-64}

echo "=== Env Server Job starting at $(date) ==="
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

mkdir -p "${RUNTIME_DIR}"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/setup_ai2thor_env.sh"

cd "${REPO}/external/VAGEN"

python -m vagen.envs.navigation.pre_download_scenes 2>&1 || echo "pre-download (may have already been done)"

HOSTFILE="${RUNTIME_DIR}/env_server_host.txt"
(
  flock -x 200
  echo "$(hostname):${PORT}" >> "$HOSTFILE"
) 200>"${HOSTFILE}.lock"
echo "Env server host appended to: $HOSTFILE"

DEVICES_ARG=$(python - <<'PY'
import os
vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if vis:
    vals = list(range(len([x for x in vis.split(',') if x.strip()])))
else:
    vals = [0]
print(str(vals))
PY
)
echo "Using navigation devices: ${DEVICES_ARG}"

python -m vagen.envs.navigation.serve \
    --port "${PORT}" \
    --devices="${DEVICES_ARG}" \
    --max_envs "${MAX_ENVS}" \
    --max_inflight "${THREAD_POOL}" \
    --thread_pool_size "${THREAD_POOL}" \
    --session_timeout 7200.0

echo "=== Env Server Job finished at $(date) ==="
