#!/bin/bash
#SBATCH --job-name=sft2-hold-8g
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --requeue

set -euo pipefail
echo "=== Holding node for SFT2 8-GPU train ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
date
while true; do sleep 300; done
