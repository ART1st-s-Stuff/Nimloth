#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline

export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export HOME=${ROOT}/.home
export FLASHINFER_WORKSPACE_DIR=${ROOT}/.cache/flashinfer
mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"
export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH

source ${ROOT}/.venv/bin/activate
source ${SCRIPTDIR}/setup_ai2thor_env.sh

printf 'NODE=%s IP=%s\n' "$(hostname)" "$(hostname -I)"

for gpu in 0 1 2 3 4 5 6 7; do
  echo "--- GPU ${gpu} start $(date) ---"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES=${gpu} timeout 120s python - <<'PY'
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
  echo "--- GPU ${gpu} rc=${rc} end $(date) ---"
done
