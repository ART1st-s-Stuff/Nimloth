#!/usr/bin/env bash
set -euo pipefail

NODE_NAME=$1
NODE_IDX=$2
CONTROL_DIR=$3
RUN_DIR=$4
JOB_ID=$5
ENV_GPUS_PER_NODE=$6
TRAIN_GPUS_PER_NODE=$7
ENV_PORT_BASE=$8

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN

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
export TMPDIR=/tmp/vagen-2node-${USER}-${JOB_ID}-${NODE_NAME}
mkdir -p "$TMPDIR" "$CONTROL_DIR" "$RUN_DIR"

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi
source ${ROOT}/.venv/bin/activate
source ${SCRIPTDIR}/setup_ai2thor_env.sh
cd "$BASEDIR"

NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${NODE_IP}" ]; then
  NODE_IP=$(hostname -I | awk '{print $1}')
fi
IFS=',' read -r -a ALLOC_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
echo "[${NODE_NAME}] Node IP: ${NODE_IP}"
echo "[${NODE_NAME}] Allocated physical GPU ids: ${ALLOC_GPU_IDS[*]}"

GOOD_ORD=()
for idx in "${!ALLOC_GPU_IDS[@]}"; do
  phys="${ALLOC_GPU_IDS[$idx]}"
  phys=$(echo "$phys" | xargs)
  [ -z "$phys" ] && continue
  echo "[${NODE_NAME}] Testing AI2-THOR ordinal ${idx} physical GPU ${phys} at $(date)"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$phys" timeout 120s python - <<'PY'
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
  if [ "$rc" -eq 0 ]; then
    echo "[${NODE_NAME}] ordinal ${idx} physical ${phys}: AI2-THOR OK"
    GOOD_ORD+=("$idx")
  else
    echo "[${NODE_NAME}] ordinal ${idx} physical ${phys}: AI2-THOR FAILED/TIMEOUT rc=${rc}"
  fi
done

echo "[${NODE_NAME}] AI2-THOR-good ordinals: ${GOOD_ORD[*]:-NONE}"
if [ "${#GOOD_ORD[@]}" -lt "$ENV_GPUS_PER_NODE" ]; then
  echo "ERROR: [${NODE_NAME}] need ${ENV_GPUS_PER_NODE} AI2-THOR-good GPUs; got ${#GOOD_ORD[@]}. Not downgrading."
  exit 2
fi

ENV_ORD=("${GOOD_ORD[@]:0:${ENV_GPUS_PER_NODE}}")
TRAIN_ORD=()
for idx in "${!ALLOC_GPU_IDS[@]}"; do
  skip=0
  for env_idx in "${ENV_ORD[@]}"; do
    [ "$idx" = "$env_idx" ] && skip=1
  done
  [ "$skip" -eq 0 ] && TRAIN_ORD+=("$idx")
done
if [ "${#TRAIN_ORD[@]}" -lt "$TRAIN_GPUS_PER_NODE" ]; then
  echo "ERROR: [${NODE_NAME}] need ${TRAIN_GPUS_PER_NODE} train GPUs after env reservation; got ${#TRAIN_ORD[@]}"
  exit 3
fi
TRAIN_ORD=("${TRAIN_ORD[@]:0:${TRAIN_GPUS_PER_NODE}}")

TRAIN_CUDA=$(python - <<PY
alloc="${ALLOC_GPU_IDS[*]}".split()
ords=[int(x) for x in "${TRAIN_ORD[*]}".split()]
print(','.join(alloc[i] for i in ords))
PY
)
echo "$TRAIN_CUDA" > "${CONTROL_DIR}/train_cuda_${NODE_NAME}.txt"
echo "[${NODE_NAME}] Env ordinals: ${ENV_ORD[*]}"
echo "[${NODE_NAME}] Train ordinals: ${TRAIN_ORD[*]} physical CUDA_VISIBLE_DEVICES=${TRAIN_CUDA}"

: > "/tmp/vagen_${JOB_ID}_env_pids"
for env_i in "${!ENV_ORD[@]}"; do
  ord="${ENV_ORD[$env_i]}"
  phys="${ALLOC_GPU_IDS[$ord]}"
  port=$(( ENV_PORT_BASE + (NODE_IDX * ENV_GPUS_PER_NODE) + env_i ))
  echo "${NODE_IP}:${port}" >> "${CONTROL_DIR}/env_hosts.txt"
  echo "[${NODE_NAME}] Starting env server ${env_i} on physical GPU ${phys} port ${port}"
  CUDA_VISIBLE_DEVICES="$phys" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
    --port "$port" \
    --devices='[0]' \
    --max_envs 48 \
    --max_inflight 48 \
    --thread_pool_size 48 \
    --session_timeout 7200.0 \
    > "${RUN_DIR}/env_server_${JOB_ID}_${NODE_NAME}_${env_i}.log" 2>&1 &
  echo $! >> "/tmp/vagen_${JOB_ID}_env_pids"
done

for env_i in "${!ENV_ORD[@]}"; do
  port=$(( ENV_PORT_BASE + (NODE_IDX * ENV_GPUS_PER_NODE) + env_i ))
  ok=0
  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[${NODE_NAME}] Env server port ${port} health OK after ${i} tries"
      ok=1
      break
    fi
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "ERROR: [${NODE_NAME}] env server port ${port} did not become healthy"
    tail -200 "${RUN_DIR}/env_server_${JOB_ID}_${NODE_NAME}_${env_i}.log" || true
    exit 4
  fi
  curl -fsS "http://127.0.0.1:${port}/health"
done

touch "${CONTROL_DIR}/ready_${NODE_NAME}"
echo "[${NODE_NAME}] Env servers ready; keeping this Slurm step alive."
wait $(cat "/tmp/vagen_${JOB_ID}_env_pids")
