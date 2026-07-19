#!/bin/bash
#SBATCH --job-name=dino-sft2-prof
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err

set -euo pipefail
REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-profile}
cd "${REPO}"
exec bash experiments/training/sft2/run_dino_world8_profile.sh
