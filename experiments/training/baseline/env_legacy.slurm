#!/bin/bash
#SBATCH --job-name=vagen-env-legacy
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:4
#SBATCH --mem=512G

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
BASEDIR=${REPO}/external/VAGEN
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
RUNTIME_DIR=${RUNTIME_DIR:-${REPO}/outputs/experiments/training/baseline/runtime}
PORT=${PORT:-5000}

mkdir -p "${RUN_DIR}" "${RUNTIME_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

NODE=$(hostname)
NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${NODE_IP}" ]; then NODE_IP=$(hostname -I | awk '{print $1}'); fi

echo "=== Legacy env server starting at $(date) ==="
echo "job=${SLURM_JOB_ID} node=${NODE} ip=${NODE_IP} port=${PORT}"
echo "RUN_DIR=${RUN_DIR}"

# Write host info for training job
echo "${NODE_IP}:${PORT}" > "${RUNTIME_DIR}/env_legacy_host.txt"

# Environment setup
export TMPDIR=/tmp/vagen-env-legacy-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

export UV_CACHE_DIR="${REPO}/.cache/uv"
export XDG_CACHE_HOME="${REPO}/.cache"
export HOME="${REPO}/.home"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export PYTHONPATH="${REPO}/src:${BASEDIR}:${BASEDIR}/verl:${PYTHONPATH:-}"
export PATH="${REPO}/.local/bin:$PATH"

mkdir -p "$HOME"

# Source env file for WANDB etc.
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi

# AI2-THOR setup
# shellcheck disable=SC1091
source "${SCRIPTDIR}/setup_ai2thor_env.sh"

cd "${BASEDIR}"

# Figure out which GPUs to use
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Smoke test AI2-THOR on each GPU
GOOD_GPUS=()
for idx in "${!GPU_IDS[@]}"; do
  phys="${GPU_IDS[$idx]}"; phys=$(echo "$phys" | xargs)
  echo "=== AI2-THOR smoke test gpu=${phys} ==="
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$phys" timeout 150s python3 - <<'PY'
import time, ai2thor.controller
from ai2thor.platform import CloudRendering
c = ai2thor.controller.Controller(
    agentMode='default', gridSize=0.1, visibilityDistance=10,
    renderDepthImage=False, renderInstanceSegmentation=False,
    width=255, height=255, fieldOfView=100,
    platform=CloudRendering, gpu_device=0,
    server_timeout=60, server_start_timeout=60,
)
try:
    ev = c.reset(scene='FloorPlan1')
    print(f'OK frame={ev.frame.shape}', flush=True)
finally:
    c.stop()
PY
  rc=$?; set -e
  if [ "$rc" -eq 0 ]; then GOOD_GPUS+=("$phys"); fi
done

if [ "${#GOOD_GPUS[@]}" -lt 4 ]; then
  echo "ERROR: need 4 AI2-THOR-ready GPUs, got ${#GOOD_GPUS[@]}"
  exit 2
fi

# Clean up any leftover ray
ray stop 2>/dev/null || true

echo "=== Starting legacy env server on GPUs ${GOOD_GPUS[*]} ==="

# Start the VAGEN server
# navigation.devices specifies which GPUs to use for AI2-THOR
PYTHONUNBUFFERED=1 python3 -m vagen.server.server \
  server.host=0.0.0.0 \
  server.port="${PORT}" \
  navigation.devices="[${GOOD_GPUS[0]},${GOOD_GPUS[1]},${GOOD_GPUS[2]},${GOOD_GPUS[3]}]" \
  navigation.max_workers=48 \
  use_state_reward=False \
  2>&1 | tee "${RUN_DIR}/env_server_${SLURM_JOB_ID}.log"

echo "=== Env server stopped at $(date) ==="
