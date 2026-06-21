#!/usr/bin/env bash
# Submit external env + val wandb watcher for an in-progress training run.
#
# Example:
#   TRAIN_RUN_DIR=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-20/vagen_nav_wm_fresh \
#   EXPERIMENT_NAME=vagen_nav_wm_fresh \
#   NODELIST_ENV=dgx-40 NODELIST_VAL=dgx-47 \
#   bash experiments/training/baseline/launch_val_wandb_watcher.sh
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
SLURM_BIN=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
ENV_CONTROL_DIR=${TRAIN_RUN_DIR}/external_env_val_wandb
VAL_EVERY=${VAL_EVERY:-1}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-300}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-50}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-0}
NODELIST_ENV=${NODELIST_ENV:-}
NODELIST_VAL=${NODELIST_VAL:-}

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm" "${TRAIN_RUN_DIR}/val_wandb_watcher"

ENV_ARGS=(--parsable --partition=preempt --export=ALL,RUN_DIR="${TRAIN_RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",CONTROL_DIR="${ENV_CONTROL_DIR}")
if [ -n "${NODELIST_ENV}" ]; then
  ENV_ARGS+=(--nodelist="${NODELIST_ENV}")
fi
ENV_JOB=$("$SLURM_BIN"/sbatch "${ENV_ARGS[@]}" "${SCRIPTDIR}/env_external_4gpu.slurm" | awk '{print $NF}')
echo "env job: ${ENV_JOB}"

VAL_ARGS=(--parsable --partition=preempt --export=ALL,TRAIN_RUN_DIR="${TRAIN_RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",ENV_CONTROL_DIR="${ENV_CONTROL_DIR}",VAL_EVERY="${VAL_EVERY}",POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC}",MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}",PRUNE_CHECKPOINTS="${PRUNE_CHECKPOINTS}")
if [ -n "${NODELIST_VAL}" ]; then
  VAL_ARGS+=(--nodelist="${NODELIST_VAL}")
fi
VAL_JOB=$("$SLURM_BIN"/sbatch "${VAL_ARGS[@]}" "${SCRIPTDIR}/val_wandb_watcher.slurm" | awk '{print $NF}')
echo "val watcher job: ${VAL_JOB}"

cat >> "${TRAIN_RUN_DIR}/val_wandb_watcher/README_launch.txt" <<EOF
launch_time=$(date -Iseconds)
env_job=${ENV_JOB}
val_watcher_job=${VAL_JOB}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR}
VAL_EVERY=${VAL_EVERY}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC}
wandb_run_name=${EXPERIMENT_NAME}_val_curve
EOF

echo "Submitted val wandb watcher for ${TRAIN_RUN_DIR}"
