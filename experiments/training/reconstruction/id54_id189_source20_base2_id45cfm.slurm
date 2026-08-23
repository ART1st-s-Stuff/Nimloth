#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id54-id189-cfm
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --exclude=dgx-[09,13,32,51]
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/slurm/id54_id189_source20_base2_id45cfm_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/slurm/id54_id189_source20_base2_id45cfm_%j.err

set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
exec "${REPO}/experiments/training/reconstruction/run_id54_id189_source20_base2_id45cfm.sh"
