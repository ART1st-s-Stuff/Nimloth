#!/bin/bash
#SBATCH --job-name=val-wandb-watch
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --requeue

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline

RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
VAL_RUN_DIR=${VAL_RUN_DIR:-${TRAIN_RUN_DIR}/val_wandb_watcher}
ENV_CONTROL_DIR=${ENV_CONTROL_DIR:-${TRAIN_RUN_DIR}/external_env_val_wandb}
ENV_URL_FILE=${ENV_URL_FILE:-${ENV_CONTROL_DIR}/env_urls.txt}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-300}
VAL_EVERY=${VAL_EVERY:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-50}

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm" "${VAL_RUN_DIR}"

echo "=== val wandb watcher slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "TRAIN_RUN_DIR=${TRAIN_RUN_DIR}"
echo "ENV_URL_FILE=${ENV_URL_FILE}"

for i in $(seq 1 360); do
  if [ -f "${ENV_CONTROL_DIR}/failed" ]; then
    echo "ERROR external env failed"
    exit 4
  fi
  if [ -f "${ENV_CONTROL_DIR}/ready" ] && [ -s "${ENV_URL_FILE}" ]; then
    break
  fi
  sleep 5
done
if [ ! -s "${ENV_URL_FILE}" ]; then
  echo "ERROR timed out waiting for external env"
  exit 4
fi

export TRAIN_RUN_DIR VAL_RUN_DIR ENV_URL_FILE EXPERIMENT_NAME
export POLL_INTERVAL_SEC VAL_EVERY MAX_TRAIN_STEPS PRUNE_CHECKPOINTS
bash "${SCRIPTDIR}/val_wandb_watcher.sh"
