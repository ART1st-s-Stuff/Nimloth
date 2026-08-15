#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id176-action-head
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --requeue

set -euo pipefail

: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
module load slurm

CONTROL_ROOT=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm
mkdir -p "${CONTROL_ROOT}"
ALLOCATION=${CONTROL_ROOT}/nimloth-id176-action-head_${SLURM_JOB_ID}.allocation.txt
scontrol show job -dd "${SLURM_JOB_ID}" -o >"${ALLOCATION}"
grep -Eq 'AllocTRES=.*cpu=64.*gres/gpu=8|AllocTRES=.*gres/gpu=8.*cpu=64' "${ALLOCATION}"
grep -q 'MinMemoryNode=256G' "${ALLOCATION}"

MONITOR=${CONTROL_ROOT}/nimloth-id176-action-head_${SLURM_JOB_ID}.nvidia_smi.csv
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 2
  done
) >"${MONITOR}" 2>&1 &
MONITOR_PID=$!
cleanup() {
  status=$?
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

export RUN_NAME=176_id74_action_head_repair_balanced271x8_val40x8
export TRAIN_EXAMPLES_PER_ACTION=271
export VALIDATION_EXAMPLES_PER_ACTION=40
export SELECTION_SEED=42002
export FIT_LEARNING_RATE=0.0001
export FIT_WEIGHT_DECAY=0.0
export FIT_MAX_EPOCHS=500
export FIT_EARLY_STOPPING_PATIENCE=50
export MINIMUM_VALIDATION_NLL_IMPROVEMENT=0.05
export MINIMUM_BF16_MEDIAN_SPREAD=0.001
export EXPECTED_WORLD_SIZE=8
export OUTPUT_ROOT=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-08-15

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:8 --exclusive \
  bash "${REPO}/experiments/training/sft2/run_id74_action_head_repair.sh"
