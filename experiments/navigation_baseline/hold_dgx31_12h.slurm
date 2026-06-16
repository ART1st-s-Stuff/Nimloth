#!/bin/bash
#SBATCH --job-name=vagen-hold-dgx31
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --requeue

set -euo pipefail
echo "=== Holding dgx-31 for VAGEN 2env4train ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
date
while true; do sleep 300; done
