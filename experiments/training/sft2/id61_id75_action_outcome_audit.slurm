#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id61-outcome-audit
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/project/peilab/atst/nimloth/logs/slurm-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/logs/slurm-%j.err

set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
exec "${REPO}/experiments/training/sft2/run_id61_id75_action_outcome_audit.sh"
