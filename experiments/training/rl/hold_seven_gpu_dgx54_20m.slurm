#!/usr/bin/env bash
#SBATCH --job-name=nimloth-seven-gpu-dgx54-hold
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-54
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:7
#SBATCH --mem=256G

set -euo pipefail
printf 'hold_job=%s node=%s partition=%s cuda_visible_devices=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_JOB_PARTITION}" \
  "${CUDA_VISIBLE_DEVICES:-}"
sleep infinity
