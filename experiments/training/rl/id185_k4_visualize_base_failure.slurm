#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id185-k4-visualize-base-r1
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51

set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
module load slurm
printf 'id185_visualize_base_failure_job=%s node=%s partition=%s cuda_visible_devices=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_JOB_PARTITION}" \
  "${CUDA_VISIBLE_DEVICES:-}"
exec "${REPO}/experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh" \
  "${SLURM_JOB_ID}"
