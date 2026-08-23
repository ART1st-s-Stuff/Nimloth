#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id60-id75-probe-t1
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/atst/nimloth/logs/slurm-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/logs/slurm-%j.err

set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
exec "${REPO}/experiments/training/sft2/run_id60_goal_probe_id75_residual_t1_canary.sh"
