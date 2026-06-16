#!/bin/bash
set -euo pipefail
ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

echo "=== Submit SFT1 train vagen79 at $(date) ==="
echo "1 node x 4 GPU DDP | init global_step_79 HF | 677 train_success records"
J=$($SLURM "${ROOT}/train_sft1_vagen79_1node4gpu.slurm" | awk '{print $NF}')
echo "train job: ${J}"
echo "log: ${ROOT}/runs/sft1_train_vagen79_qwen25vl/sft1_train_vagen79.log"
