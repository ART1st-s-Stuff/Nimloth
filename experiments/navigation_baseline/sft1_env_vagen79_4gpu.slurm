#!/bin/bash
#SBATCH --job-name=sft1-env-v79-4gpu
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
# Node: pass at submit time, e.g. sbatch --nodelist=dgx-46 (default in submit script).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:4
#SBATCH --mem=512G

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
EXPERIMENT_NAME=sft1_rollouts_vagen79_greedy_parallel
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CONTROL_DIR=${RUN_DIR}/external_env_4gpu
mkdir -p "$RUN_DIR" "$CONTROL_DIR"
: > "${CONTROL_DIR}/env_hosts.txt"
: > "${CONTROL_DIR}/env_urls.txt"
rm -f "${CONTROL_DIR}/ready" "${CONTROL_DIR}/failed"

export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export HOME=${ROOT}/.home
export FLASHINFER_WORKSPACE_DIR=${ROOT}/.cache/flashinfer
mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"
export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/vagen-env-sft1v79-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi
source ${ROOT}/.venv/bin/activate
source ${SCRIPTDIR}/setup_ai2thor_env.sh
cd "$BASEDIR"

NODE=$(hostname)
NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${NODE_IP}" ]; then NODE_IP=$(hostname -I | awk '{print $1}'); fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PORT_BASE=${PORT_BASE:-8400}

echo "=== SFT1 vagen79 4-GPU env job starting ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
echo "job=${SLURM_JOB_ID} node=${NODE} ip=${NODE_IP}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

declare -a GOOD_PHYS=()
for idx in "${!GPU_IDS[@]}"; do
  phys="${GPU_IDS[$idx]}"; phys=$(echo "$phys" | xargs)
  echo "=== AI2-THOR smoke ordinal=${idx} physical=${phys} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$phys" timeout 150s python - <<'PY'
import time
import ai2thor.controller
from ai2thor.platform import CloudRendering
c = ai2thor.controller.Controller(
    agentMode='default', gridSize=0.1, visibilityDistance=10,
    renderDepthImage=False, renderInstanceSegmentation=False,
    width=255, height=255, fieldOfView=100,
    platform=CloudRendering, gpu_device=0,
    server_timeout=60, server_start_timeout=60,
)
try:
    c.reset(scene='FloorPlan1')
finally:
    c.stop()
print('OK', flush=True)
PY
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then GOOD_PHYS+=("$phys"); fi
done

if [ "${#GOOD_PHYS[@]}" -lt 4 ]; then
  echo "ERROR need 4 good GPUs, got ${#GOOD_PHYS[@]}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  touch "${CONTROL_DIR}/failed"
  exit 2
fi

: > "/tmp/vagen_env_sft1v79_${SLURM_JOB_ID}_pids"
cleanup() {
  set +e
  if [ -f "/tmp/vagen_env_sft1v79_${SLURM_JOB_ID}_pids" ]; then
    xargs -r kill < "/tmp/vagen_env_sft1v79_${SLURM_JOB_ID}_pids" 2>/dev/null || true
  fi
  # Do not remove shared ready/urls: other eval jobs may still depend on them.
}
trap cleanup EXIT

for env_i in 0 1 2 3; do
  phys="${GOOD_PHYS[$env_i]}"
  port=$((PORT_BASE + env_i))
  echo "http://${NODE_IP}:${port}" >> "${CONTROL_DIR}/env_urls.txt"
  CUDA_VISIBLE_DEVICES="$phys" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
    --port "$port" --devices='[0]' --max_envs 48 --max_inflight 48 \
    --thread_pool_size 48 --session_timeout 7200.0 \
    > "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${NODE}_${env_i}.log" 2>&1 &
  echo $! >> "/tmp/vagen_env_sft1v79_${SLURM_JOB_ID}_pids"
done

for env_i in 0 1 2 3; do
  port=$((PORT_BASE + env_i))
  ok=0
  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done
  if [ "$ok" -ne 1 ]; then
    touch "${CONTROL_DIR}/failed"
    exit 4
  fi
done

touch "${CONTROL_DIR}/ready"
echo "${SLURM_JOB_ID}" > "${CONTROL_DIR}/ready_job_id"
echo "=== env ready ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
cat "${CONTROL_DIR}/env_urls.txt" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
wait $(cat "/tmp/vagen_env_sft1v79_${SLURM_JOB_ID}_pids")
