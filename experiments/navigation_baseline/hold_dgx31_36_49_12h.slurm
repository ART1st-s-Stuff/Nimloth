#!/bin/bash
#SBATCH --job-name=vagen-hold-3nodes-envprobe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-31,dgx-36,dgx-49
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --requeue

set -euo pipefail

echo "=== Holding nodes for VAGEN env probe ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
date

# Keep the allocation alive so separate srun --jobid probes can be launched.
while true; do
  sleep 300
done
