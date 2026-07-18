#!/usr/bin/env bash
#SBATCH --job-name=hold-verl-exact8
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err

set -euo pipefail
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm
echo "HOLD_READY job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} gpus=${SLURM_GPUS:-unknown}"
exec bash -c 'trap "exit 0" TERM INT; while true; do sleep 300 & wait $!; done'
