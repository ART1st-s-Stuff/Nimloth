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
REPO=/project/peilab/atst/nimloth-feat-rl

export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:$PATH
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"

ENV_PORT=5000
HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[ -z "${HEAD_IP}" ] && HEAD_IP=$(hostname -I | awk '{print $1}')

echo "=== Env server on $(hostname) ${HEAD_IP}:${ENV_PORT} ==="
echo "ENV_URL=http://${HEAD_IP}:${ENV_PORT}"

cd "${REPO}/external/VAGEN"
exec python3 -m vagen.server.server \
    server.host=0.0.0.0 \
    server.port=${ENV_PORT} \
    use_state_reward=False \
    navigation.devices=[0] \
    navigation.max_workers=4
