#!/usr/bin/env bash
# Hold 1x8GPU + external env, then srun train resume (SERVER.md pattern).
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/baseline
SLURM=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

: "${RUN_DIR:?RUN_DIR required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME required}"

RESUME_FROM_STEP=${RESUME_FROM_STEP:-}
TOTAL_STEPS=${TOTAL_STEPS:-50}
TEST_FREQ=${TEST_FREQ:-1}
ENABLE_WANDB=${ENABLE_WANDB:-1}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-0}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-8}
TRAIN_NODES=${TRAIN_NODES:-1}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-6}
ENV_CONTROL_DIR=${ENV_CONTROL_DIR:-${RUN_DIR}/external_env_hold}
NODELIST_HOLD=${NODELIST_HOLD:-}
NODELIST_ENV=${NODELIST_ENV:-}

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm"

HOLD_ARGS=(--parsable --account=peilab --partition=preempt)
[ -n "${NODELIST_HOLD}" ] && HOLD_ARGS+=(--nodelist="${NODELIST_HOLD}")
HOLD_JOB=$("$SLURM"/sbatch "${HOLD_ARGS[@]}" "${ROOT}/hold_preempt_1n8g.slurm" | awk '{print $NF}')
echo "hold job: ${HOLD_JOB}"

for _ in $(seq 1 360); do
  state=$("$SLURM"/squeue -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
  [ "${state}" = "RUNNING" ] && break
  [ -z "${state}" ] && { echo "ERROR hold ${HOLD_JOB} disappeared"; exit 1; }
  sleep 5
done
HOLD_NODE=$("$SLURM"/squeue -j "${HOLD_JOB}" -h -o '%N')
echo "hold node: ${HOLD_NODE}"

rm -f "${ENV_CONTROL_DIR}/failed" "${ENV_CONTROL_DIR}/ready"
mkdir -p "${ENV_CONTROL_DIR}"
: > "${ENV_CONTROL_DIR}/env_urls.txt"
: > "${ENV_CONTROL_DIR}/env_hosts.txt"

ENV_ARGS=(--parsable --partition=preempt --export=ALL,RUN_DIR="${RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",CONTROL_DIR="${ENV_CONTROL_DIR}")
[ -n "${NODELIST_ENV}" ] && ENV_ARGS+=(--nodelist="${NODELIST_ENV}")
ENV_JOB=$("$SLURM"/sbatch "${ENV_ARGS[@]}" "${ROOT}/env_external_4gpu.slurm" | awk '{print $NF}')
echo "env job: ${ENV_JOB}"

for _ in $(seq 1 360); do
  if [ -f "${ENV_CONTROL_DIR}/failed" ]; then
    echo "ERROR external env failed"
    "$SLURM"/scancel "${HOLD_JOB}" || true
    exit 4
  fi
  if [ -f "${ENV_CONTROL_DIR}/ready" ] && [ -s "${ENV_CONTROL_DIR}/env_urls.txt" ]; then
    break
  fi
  sleep 5
done
if [ ! -s "${ENV_CONTROL_DIR}/env_urls.txt" ]; then
  echo "ERROR timed out waiting for env"
  "$SLURM"/scancel "${HOLD_JOB}" "${ENV_JOB}" || true
  exit 4
fi

echo "=== srun train resume on hold ${HOLD_JOB} ==="
set +e
"$SLURM"/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HOLD_NODE}" bash -lc "
  export RUN_DIR='${RUN_DIR}'
  export EXPERIMENT_NAME='${EXPERIMENT_NAME}'
  export ENV_CONTROL_DIR='${ENV_CONTROL_DIR}'
  export RESUME_FROM_STEP='${RESUME_FROM_STEP}'
  export TOTAL_STEPS='${TOTAL_STEPS}'
  export TEST_FREQ='${TEST_FREQ}'
  export ENABLE_WANDB='${ENABLE_WANDB}'
  export PRUNE_CHECKPOINTS='${PRUNE_CHECKPOINTS}'
  export TRAIN_GPUS_PER_NODE='${TRAIN_GPUS_PER_NODE}'
  export TRAIN_NODES='${TRAIN_NODES}'
  export AGENT_NUM_WORKERS='${AGENT_NUM_WORKERS}'
  export AGENT_MAX_CONCURRENT_TRAJECTORIES='${AGENT_MAX_CONCURRENT_TRAJECTORIES}'
  bash '${ROOT}/train_resume.slurm'
"
RC=$?
set -e

"$SLURM"/scancel "${HOLD_JOB}" "${ENV_JOB}" 2>/dev/null || true
exit "${RC}"
