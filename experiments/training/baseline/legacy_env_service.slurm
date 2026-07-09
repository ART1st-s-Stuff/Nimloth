#!/bin/bash
#SBATCH --job-name=vagen-legacy-env
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:2
#SBATCH --mem=512G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/baseline
BASEDIR=${REPO}/external/VAGEN
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_legacy_env_service}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
CONTROL_DIR=${CONTROL_DIR:-${RUN_DIR}/legacy_env_service}
LOG_FILE=${LOG_FILE:-${RUN_DIR}/${EXPERIMENT_NAME}.log}
ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-2}
SERVICE_PORT=${SERVICE_PORT:-5000}
LEGACY_ENV_MAX_WORKERS=${LEGACY_ENV_MAX_WORKERS:-48}

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"
rm -f "${CONTROL_DIR}/ready" "${CONTROL_DIR}/failed"

# shellcheck disable=SC1091
source "${PYTHON_ENV}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/setup_ai2thor_env.sh"
cd "${BASEDIR}"

NODE=$(hostname)
NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${NODE_IP}" ]; then NODE_IP=$(hostname -I | awk '{print $1}'); fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

{
  echo "=== legacy external env job starting $(date) ==="
  echo "job=${SLURM_JOB_ID} node=${NODE} ip=${NODE_IP} run_dir=${RUN_DIR}"
  echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
} | tee -a "${LOG_FILE}"

pkill -u "$USER" -f 'vagen.server.server|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true

declare -a GOOD_PHYS=()
for idx in "${!GPU_IDS[@]}"; do
  phys="${GPU_IDS[$idx]}"
  phys=$(echo "$phys" | xargs)
  [ -z "$phys" ] && continue
  echo "=== AI2-THOR smoke ordinal=${idx} physical=${phys} ===" | tee -a "${LOG_FILE}"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$phys" timeout 150s python - <<'PY'
import time
import ai2thor.controller
from ai2thor.platform import CloudRendering
print('create', flush=True)
t0=time.time()
c=ai2thor.controller.Controller(
    agentMode='default', gridSize=0.1, visibilityDistance=10,
    renderDepthImage=False, renderInstanceSegmentation=False,
    width=255, height=255, fieldOfView=100,
    platform=CloudRendering, gpu_device=0,
    server_timeout=60, server_start_timeout=60,
)
print(f'created {time.time()-t0:.1f}s', flush=True)
try:
    t1=time.time(); ev=c.reset(scene='FloorPlan1')
    print(f'reset {time.time()-t1:.1f}s frame={None if ev.frame is None else ev.frame.shape}', flush=True)
finally:
    c.stop()
print('OK', flush=True)
PY
  rc=$?
  set -e
  echo "AI2-THOR rc=${rc} physical=${phys}" | tee -a "${LOG_FILE}"
  if [ "$rc" -eq 0 ]; then GOOD_PHYS+=("$phys"); fi
done

if [ "${#GOOD_PHYS[@]}" -lt "${ENV_GPUS_PER_NODE}" ]; then
  echo "ERROR need ${ENV_GPUS_PER_NODE} AI2-THOR-good GPUs, got ${#GOOD_PHYS[@]}" | tee -a "${LOG_FILE}"
  touch "${CONTROL_DIR}/failed"
  exit 2
fi

ENV_PHYS=("${GOOD_PHYS[@]:0:${ENV_GPUS_PER_NODE}}")
ENV_CUDA=$(IFS=,; echo "${ENV_PHYS[*]}")
ENV_DEVICES=$(python - <<PY
n=${ENV_GPUS_PER_NODE}
print('[' + ','.join(str(i) for i in range(n)) + ']')
PY
)
SERVICE_BASE_URL="http://${NODE_IP}:${SERVICE_PORT}"

echo "${NODE}" > "${CONTROL_DIR}/node.txt"
echo "${NODE_IP}" > "${CONTROL_DIR}/ip.txt"
echo "${SERVICE_BASE_URL}" > "${CONTROL_DIR}/base_url.txt"
echo "${ENV_CUDA}" > "${CONTROL_DIR}/env_cuda.txt"

cleanup() {
  set +e
  echo "=== legacy env cleanup $(date) ===" | tee -a "${LOG_FILE}"
  rm -f "${CONTROL_DIR}/ready"
  if [ -n "${ENV_PID:-}" ]; then
    kill "${ENV_PID}" >/dev/null 2>&1 || true
    wait "${ENV_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "=== Starting legacy env service CUDA=${ENV_CUDA} devices=${ENV_DEVICES} url=${SERVICE_BASE_URL} ===" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${ENV_CUDA}" PYTHONUNBUFFERED=1 python -m vagen.server.server \
  server.host=0.0.0.0 \
  server.port=${SERVICE_PORT} \
  use_state_reward=False \
  navigation.devices="${ENV_DEVICES}" \
  navigation.max_workers=${LEGACY_ENV_MAX_WORKERS} \
  > "${RUN_DIR}/legacy_env_server_${SLURM_JOB_ID}_${NODE}.log" 2>&1 &
ENV_PID=$!

for i in $(seq 1 180); do
  if curl -fsS "${SERVICE_BASE_URL}/health" >/dev/null 2>&1; then
    echo "legacy env service health OK after ${i} checks" | tee -a "${LOG_FILE}"
    touch "${CONTROL_DIR}/ready"
    wait "${ENV_PID}"
    exit $?
  fi
  kill -0 "${ENV_PID}" >/dev/null 2>&1 || {
    echo "ERROR legacy env service exited" | tee -a "${LOG_FILE}"
    tail -200 "${RUN_DIR}/legacy_env_server_${SLURM_JOB_ID}_${NODE}.log" | tee -a "${LOG_FILE}" || true
    touch "${CONTROL_DIR}/failed"
    exit 4
  }
  sleep 5
  if [ "$i" -eq 180 ]; then
    echo "ERROR timed out waiting for legacy env service" | tee -a "${LOG_FILE}"
    tail -200 "${RUN_DIR}/legacy_env_server_${SLURM_JOB_ID}_${NODE}.log" | tee -a "${LOG_FILE}" || true
    touch "${CONTROL_DIR}/failed"
    exit 4
  fi
done
