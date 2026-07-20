#!/bin/bash
#SBATCH --job-name=rl-k8wm-smoke
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=180G

set -euo pipefail
: "${REPO:?REPO is required}"
: "${RUN_OUT:?RUN_OUT is required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"

exec bash "${REPO}/experiments/training/rl/run_k8_wm_fastpath_smoke.sh"
