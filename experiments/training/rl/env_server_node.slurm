#!/bin/bash
#SBATCH --job-name=rl-env
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Dedicated env server for RL training.
# After it starts, note the IP:PORT and pass as ENV_URL to the training job.

set -euo pipefail
: "${REPO:?export REPO as the pinned Nimloth worktree before sbatch}"
EXPECTED_VAGEN_COMMIT=${EXPECTED_VAGEN_COMMIT:-316d9d7}
[[ "$(git -C "${REPO}/external/VAGEN" rev-parse --short=7 HEAD)" == "${EXPECTED_VAGEN_COMMIT}" ]] || {
    echo "unexpected VAGEN revision; expected ${EXPECTED_VAGEN_COMMIT}" >&2
    exit 1
}

export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:$PATH
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"

ENV_PORT=5000
HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[ -z "${HEAD_IP}" ] && HEAD_IP=$(hostname -I | awk '{print $1}')

echo "=== Env server on $(hostname) ${HEAD_IP}:${ENV_PORT} ==="
echo "ENV_URL=http://${HEAD_IP}:${ENV_PORT}"

cd "${REPO}/external/VAGEN"
exec python3 -m vagen.envs.navigation.serve \
    --host=0.0.0.0 \
    --port="${ENV_PORT}" \
    --devices='[0]' \
    --max_envs=4 \
    --max_inflight=4 \
    --thread_pool_size=4
