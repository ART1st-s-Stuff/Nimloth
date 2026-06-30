#!/usr/bin/env bash
# Generic: hold 2 preempt nodes and run co-located VAGEN PPO training.
# Set EXPERIMENT_NAME (required), optional NODELIST / hyperparams via env.
# Record the exact command in outputs/experiments/training/baseline/<date>/<run>/README.md
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
SQUEUE=/cm/shared/apps/slurm/current/bin/squeue
SRUN=/cm/shared/apps/slurm/current/bin/srun
SCONTROL=/cm/shared/apps/slurm/current/bin/scontrol
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME (record in outputs README)}"

NODELIST=${NODELIST:-}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
TOTAL_STEPS=${TOTAL_STEPS:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
TEST_FREQ=${TEST_FREQ:-10}
ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-2}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
RUN_DIR=${RUN_DIR:-}
RESUME_FROM_STEP=${RESUME_FROM_STEP:-}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-0}

echo "=== Preempt co-located VAGEN training ==="
echo "experiment=${EXPERIMENT_NAME} run_date=${RUN_DATE}"
echo "run_dir=${RUN_DIR:-auto}"
echo "resume_from_step=${RESUME_FROM_STEP:-fresh}"
echo "prune_checkpoints=${PRUNE_CHECKPOINTS}"
echo "steps=${TOTAL_STEPS} batch=${TRAIN_BATCH_SIZE}/${PPO_MINI_BATCH_SIZE} test_freq=${TEST_FREQ}"
echo "topology: ${ENV_GPUS_PER_NODE} env + ${TRAIN_GPUS_PER_NODE} train GPU per node"
echo "nodelist: ${NODELIST:-slurm auto}"

HOLD_ARGS=(--account=peilab --partition=preempt --job-name="hold-preempt")
[ -n "${NODELIST}" ] && HOLD_ARGS+=(--nodelist="${NODELIST}")
HOLD_JOB=$($SLURM "${HOLD_ARGS[@]}" "${ROOT}/hold_preempt.slurm" | awk '{print $NF}')
echo "hold job: ${HOLD_JOB}"

for _ in $(seq 1 120); do
  state=$($SQUEUE -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
  [ "${state}" = "RUNNING" ] && break
  [ -z "${state}" ] && { echo "ERROR hold ${HOLD_JOB} disappeared"; exit 1; }
  sleep 5
done
[ "$($SQUEUE -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)" = "RUNNING" ] || { echo "ERROR hold not RUNNING"; exit 1; }

mapfile -t HOLD_NODES < <($SCONTROL show hostnames "$($SQUEUE -j "${HOLD_JOB}" -h -o "%N")")
if [ "${#HOLD_NODES[@]}" -lt 2 ]; then
  echo "ERROR: hold ${HOLD_JOB} has ${#HOLD_NODES[@]} nodes, need 2"
  exit 1
fi
echo "hold nodes: ${HOLD_NODES[*]}"

set +e
$SRUN --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HOLD_NODES[0]}" bash -lc "
  export HOLD_JOB=${HOLD_JOB}
  export RUN_DATE=${RUN_DATE}
  export EXPERIMENT_NAME=${EXPERIMENT_NAME}
  export TOTAL_STEPS=${TOTAL_STEPS}
  export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
  export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}
  export TEST_FREQ=${TEST_FREQ}
  export ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE}
  export TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE}
  export RESUME_FROM_STEP=${RESUME_FROM_STEP}
  export PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS}
  export ENABLE_WANDB=${ENABLE_WANDB}
  export RUN_DIR=${RUN_DIR}
  bash ${ROOT}/run_preempt_training.sh
"
RC=$?
set -e
$SCANCEL "${HOLD_JOB}" || true
exit ${RC}
