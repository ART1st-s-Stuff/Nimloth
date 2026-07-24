#!/usr/bin/env bash
# Resource-only allocation. Supply nodes/tasks/GPUs on the sbatch command line.
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/hold-%j.out

set -euo pipefail

echo "hold job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
while true; do
  sleep 300
done
