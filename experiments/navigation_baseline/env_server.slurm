#!/bin/bash
#SBATCH --job-name=vagen-nav-env
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --exclude=dgx-03,dgx-16,dgx-20,dgx-23,dgx-31,dgx-34,dgx-47
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G

# ============================================================================
# VAGEN Navigation Environment Server
# ============================================================================
# Serves AI2-THOR navigation environments to training nodes via HTTP.
# One env server GPU can handle many concurrent env instances.
# ============================================================================

set -e

echo "=== Env Server Job starting at $(date) ==="
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Config ---
PORT=${1:-8000}
MAX_ENVS=${2:-64}
THREAD_POOL=${3:-64}

# --- uv venv setup ---
export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv
export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python
export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache
export HOME=/project/peilab/atst/nimloth/.home
export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/vagen-nav-env-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi
source /project/peilab/atst/nimloth/.venv/bin/activate
source /project/peilab/atst/nimloth/experiments/navigation_baseline/setup_ai2thor_env.sh

# cd to VAGEN
cd /project/peilab/atst/nimloth/external/VAGEN

# --- Pre-download scenes (first time only) ---
python -m vagen.envs.navigation.pre_download_scenes 2>&1 || echo "pre-download (may have already been done)"

# --- Start env server ---
echo "=== Starting Navigation Env Server on port ${PORT} ==="
echo "Server host: $(hostname)"

# Append hostname to a shared location so the training job can discover all env servers
HOSTFILE="/project/peilab/atst/nimloth/experiments/navigation_baseline/env_server_host.txt"
(
  flock -x 200
  echo "$(hostname):${PORT}" >> "$HOSTFILE"
) 200>"${HOSTFILE}.lock"
echo "Env server host appended to: $HOSTFILE"

DEVICES_ARG=$(python - <<'PY'
import os
vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if vis:
    # Slurm may expose physical GPU ids (e.g. "5,6") while CUDA remaps them
    # to visible ordinals 0..N-1 inside the process. AI2-THOR should receive
    # the visible ordinal ids.
    vals = list(range(len([x for x in vis.split(',') if x.strip()])))
else:
    vals = [0]
print(str(vals))
PY
)
echo "Using navigation devices: ${DEVICES_ARG}"

python -m vagen.envs.navigation.serve \
    --port ${PORT} \
    --devices="${DEVICES_ARG}" \
    --max_envs ${MAX_ENVS} \
    --max_inflight ${THREAD_POOL} \
    --thread_pool_size ${THREAD_POOL} \
    --session_timeout 7200.0

echo "=== Env Server Job finished at $(date) ==="
