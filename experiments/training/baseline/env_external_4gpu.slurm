#!/bin/bash
#SBATCH --job-name=vagen-env-ext4gpu
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
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
CONTROL_DIR=${CONTROL_DIR:-${RUN_DIR}/external_env_4gpu}
ENV_COUNT=${ENV_COUNT:-4}
MAX_ENVS_PER_PROC=${MAX_ENVS_PER_PROC:-48}
PORT_BASE=${PORT_BASE:-8400}

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"
: > "${CONTROL_DIR}/env_hosts.txt"
: > "${CONTROL_DIR}/env_urls.txt"
rm -f "${CONTROL_DIR}/ready" "${CONTROL_DIR}/failed"

export TMPDIR=/tmp/vagen-env-ext4-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/setup_ai2thor_env.sh"
cd "$BASEDIR"

NODE=$(hostname)
NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${NODE_IP}" ]; then NODE_IP=$(hostname -I | awk '{print $1}'); fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== external ${ENV_COUNT}-GPU env job starting ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
echo "job=${SLURM_JOB_ID} node=${NODE} ip=${NODE_IP} RUN_DIR=${RUN_DIR}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

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
  echo "AI2-THOR rc=${rc} physical=${phys}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  if [ "$rc" -eq 0 ]; then GOOD_PHYS+=("$phys"); fi
done

if [ "${#GOOD_PHYS[@]}" -lt "${ENV_COUNT}" ]; then
  echo "ERROR need ${ENV_COUNT} AI2-THOR-good GPUs, got ${#GOOD_PHYS[@]}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  touch "${CONTROL_DIR}/failed"
  exit 2
fi

PIDFILE="/tmp/vagen_env_ext4_${SLURM_JOB_ID}_pids"
: > "$PIDFILE"
cleanup() {
  set +e
  echo "=== external env cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  xargs -r kill < "$PIDFILE" 2>/dev/null || true
  rm -f "${CONTROL_DIR}/ready"
}
trap cleanup EXIT

for env_i in $(seq 0 $((ENV_COUNT - 1))); do
  phys="${GOOD_PHYS[$env_i]}"
  port=$((PORT_BASE + env_i))
  echo "${NODE_IP}:${port}" >> "${CONTROL_DIR}/env_hosts.txt"
  echo "http://${NODE_IP}:${port}" >> "${CONTROL_DIR}/env_urls.txt"
  echo "starting env_i=${env_i} physical=${phys} port=${port}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  CUDA_VISIBLE_DEVICES="$phys" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
    --port "$port" \
    --devices='[0]' \
    --max_envs "${MAX_ENVS_PER_PROC}" \
    --max_inflight "${MAX_ENVS_PER_PROC}" \
    --thread_pool_size "${MAX_ENVS_PER_PROC}" \
    --session_timeout 7200.0 \
    > "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${NODE}_${env_i}.log" 2>&1 &
  echo $! >> "$PIDFILE"
done

for env_i in $(seq 0 $((ENV_COUNT - 1))); do
  port=$((PORT_BASE + env_i))
  ok=0
  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "health OK port=${port} tries=${i}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
      ok=1; break
    fi
    sleep 3
  done
  if [ "$ok" -ne 1 ]; then
    echo "ERROR env health failed port=${port}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    tail -200 "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${NODE}_${env_i}.log" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true
    touch "${CONTROL_DIR}/failed"
    exit 4
  fi
done

touch "${CONTROL_DIR}/ready"
echo "=== external env ready: $(tr '\n' ' ' < "${CONTROL_DIR}/env_urls.txt") ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
wait $(cat "$PIDFILE")
