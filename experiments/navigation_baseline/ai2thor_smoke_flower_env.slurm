#!/bin/bash
#SBATCH --job-name=flower-ai2thor-smoke
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --exclude=dgx-03,dgx-16,dgx-20,dgx-23,dgx-31,dgx-34,dgx-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

echo "=== flower AI2-THOR smoke at $(date) ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
cd /project/peilab/atst/flower
source .venv/bin/activate
source scripts/running/setup_ai2thor_env.sh
python - <<'PY'
import time
import ai2thor.controller
from ai2thor.platform import CloudRendering
print('Creating controller...', flush=True)
t0=time.time()
controller=ai2thor.controller.Controller(agentMode='default',gridSize=0.1,visibilityDistance=10,renderDepthImage=False,renderInstanceSegmentation=False,width=255,height=255,fieldOfView=100,platform=CloudRendering,gpu_device=0,server_timeout=120,server_start_timeout=120)
print(f'created {time.time()-t0:.1f}s', flush=True)
try:
    t1=time.time(); ev=controller.reset(scene='FloorPlan1'); print(f'reset {time.time()-t1:.1f}s frame={None if ev.frame is None else ev.frame.shape}', flush=True)
finally:
    controller.stop()
print('OK', flush=True)
PY
