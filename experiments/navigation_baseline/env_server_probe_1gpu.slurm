#!/bin/bash
#SBATCH --job-name=vagen-nav-env-probe1g
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --exclude=dgx-03,dgx-16,dgx-20,dgx-23,dgx-31,dgx-34,dgx-47
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

echo "=== Probe 1-GPU Env Server starting at $(date) ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

PORT=${1:-8000}
MAX_ENVS=${2:-16}
THREAD_POOL=${3:-16}

export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv
export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python
export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache
export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/vagen-nav-env-probe-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi
source /project/peilab/atst/nimloth/.venv/bin/activate
source /project/peilab/atst/nimloth/experiments/navigation_baseline/setup_ai2thor_env.sh

cd /project/peilab/atst/nimloth/external/VAGEN

HOSTFILE=/project/peilab/atst/nimloth/experiments/navigation_baseline/env_server_probe_host.txt
rm -f "$HOSTFILE" "${HOSTFILE}.lock"
echo "$(hostname):${PORT}" > "$HOSTFILE"
echo "Probe env server host written to $HOSTFILE"

python -m vagen.envs.navigation.serve \
  --port ${PORT} \
  --devices='[0]' \
  --max_envs ${MAX_ENVS} \
  --max_inflight ${THREAD_POOL} \
  --thread_pool_size ${THREAD_POOL} \
  --session_timeout 7200.0
