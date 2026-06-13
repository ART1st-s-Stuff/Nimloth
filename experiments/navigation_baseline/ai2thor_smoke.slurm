#!/bin/bash
#SBATCH --job-name=vagen-ai2thor-smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --exclude=dgx-03,dgx-16,dgx-20,dgx-23,dgx-31,dgx-34,dgx-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

echo "=== AI2-THOR smoke at $(date) ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

ROOT=/project/peilab/atst/nimloth
export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH
source ${ROOT}/.venv/bin/activate
source ${ROOT}/experiments/navigation_baseline/setup_ai2thor_env.sh
cd ${ROOT}/external/VAGEN

python - <<'PY'
import time
import ai2thor.controller
from ai2thor.platform import CloudRendering

print("Creating controller...", flush=True)
t0 = time.time()
controller = ai2thor.controller.Controller(
    agentMode="default",
    gridSize=0.1,
    visibilityDistance=10,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=255,
    height=255,
    fieldOfView=100,
    platform=CloudRendering,
    gpu_device=0,
    server_timeout=120,
    server_start_timeout=120,
)
print(f"Controller created in {time.time()-t0:.1f}s", flush=True)
try:
    t1 = time.time()
    event = controller.reset(scene="FloorPlan1")
    print(f"Reset ok in {time.time()-t1:.1f}s; frame={event.frame.shape if event.frame is not None else None}", flush=True)
finally:
    controller.stop()
print("AI2-THOR smoke OK", flush=True)
PY

echo "=== AI2-THOR smoke finished at $(date) ==="
