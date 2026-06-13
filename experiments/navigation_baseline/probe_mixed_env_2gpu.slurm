#!/bin/bash
#SBATCH --job-name=vagen-probe-mixed-env2
#SBATCH --output=%x_%j_%N.out
#SBATCH --error=%x_%j_%N.err
#SBATCH --time=00:30:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --mem=256G

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
RUN_DIR=${SCRIPTDIR}/runs/probe_mixed_env2_${SLURM_JOB_ID}_${SLURM_JOB_NODELIST}
mkdir -p "$RUN_DIR"

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
export TMPDIR=/tmp/vagen-probe-mixed-env2-${USER}-${SLURM_JOB_ID}
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
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

echo "=== Mixed env probe ==="
echo "date=$(date)"
echo "job=${SLURM_JOB_ID} node=${NODE} node_ip=${NODE_IP}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "allocated_gpu_ids=${GPU_IDS[*]}"
echo "run_dir=${RUN_DIR}"

GOOD=()
for idx in "${!GPU_IDS[@]}"; do
  phys="${GPU_IDS[$idx]}"; phys=$(echo "$phys" | xargs)
  echo "=== AI2-THOR smoke ordinal=${idx} physical=${phys} ==="
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
  echo "AI2-THOR rc=${rc} ordinal=${idx} physical=${phys}"
  if [ "$rc" -eq 0 ]; then GOOD+=("$idx:$phys"); fi
done

echo "GOOD=${GOOD[*]:-NONE}"
if [ "${#GOOD[@]}" -lt 2 ]; then
  echo "ERROR need 2 good env GPUs, got ${#GOOD[@]}"
  exit 2
fi

: > "$RUN_DIR/env_hosts.txt"
: > /tmp/vagen_probe_mixed_env2_${SLURM_JOB_ID}_pids
PORT_BASE=${PORT_BASE:-8300}
for env_i in 0 1; do
  item="${GOOD[$env_i]}"
  phys="${item#*:}"
  port=$((PORT_BASE + env_i))
  echo "${NODE_IP}:${port}" >> "$RUN_DIR/env_hosts.txt"
  echo "=== starting env server env_i=${env_i} physical=${phys} port=${port} ==="
  CUDA_VISIBLE_DEVICES="$phys" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
    --port "$port" \
    --devices='[0]' \
    --max_envs 48 \
    --max_inflight 48 \
    --thread_pool_size 48 \
    --session_timeout 7200.0 \
    > "$RUN_DIR/env_server_${env_i}.log" 2>&1 &
  echo $! >> /tmp/vagen_probe_mixed_env2_${SLURM_JOB_ID}_pids
done

cleanup() {
  set +e
  xargs -r kill < /tmp/vagen_probe_mixed_env2_${SLURM_JOB_ID}_pids 2>/dev/null || true
}
trap cleanup EXIT

for env_i in 0 1; do
  port=$((PORT_BASE + env_i))
  ok=0
  for i in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "health OK port=${port} tries=${i}"
      ok=1; break
    fi
    sleep 3
  done
  if [ "$ok" -ne 1 ]; then
    echo "ERROR health failed port=${port}"
    tail -200 "$RUN_DIR/env_server_${env_i}.log" || true
    exit 4
  fi
  curl -fsS "http://127.0.0.1:${port}/health" || true
  echo
  tail -20 "$RUN_DIR/env_server_${env_i}.log" || true
done

echo "=== mixed env 2gpu probe SUCCESS ==="
echo "env_hosts=$(cat $RUN_DIR/env_hosts.txt | tr '\n' ' ')"
