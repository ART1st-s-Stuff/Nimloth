#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id55-id189-cfm-all
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/atst/nimloth/logs/slurm-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/logs/slurm-%j.err

set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
exec "${REPO}/experiments/training/reconstruction/run_id55_id189_source20_all120_id45cfm.sh"
