#!/usr/bin/env bash
# Hold 1x8GPU + external env, then srun train resume (.local/SERVER.md pattern).
#
# Hold 占住节点后默认不释放（KEEP_HOLD=1），env/训练失败时可换 NODELIST_ENV 重试：
#   USE_EXISTING_HOLD=<hold_job_id> RUN_DIR=... NODELIST_ENV=dgx-12 bash launch_hold_train_resume.sh
#
# 仅占节点：
#   RUN_DIR=... HOLD_ONLY=1 bash launch_hold_train_resume.sh
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/baseline
SLURM=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

: "${RUN_DIR:?RUN_DIR required}"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_wm_fresh}

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
ENV_PARTITION=${ENV_PARTITION:-preempt}
KEEP_HOLD=${KEEP_HOLD:-1}
HOLD_ONLY=${HOLD_ONLY:-0}
USE_EXISTING_HOLD=${USE_EXISTING_HOLD:-}

ENV_JOB=""

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm" "${RUN_DIR}"

cancel_env_job() {
  if [ -n "${ENV_JOB}" ]; then
    "$SLURM"/scancel "${ENV_JOB}" 2>/dev/null || true
  fi
}

release_hold_if_requested() {
  if [ "${KEEP_HOLD}" = "0" ] && [ -n "${HOLD_JOB:-}" ]; then
    echo "=== releasing hold job ${HOLD_JOB} (KEEP_HOLD=0) ==="
    "$SLURM"/scancel "${HOLD_JOB}" 2>/dev/null || true
  fi
}

print_hold_info() {
  {
    echo "=== hold job ${HOLD_JOB} on ${HOLD_NODE} ==="
    echo "hold_job_id=${HOLD_JOB}"
    echo "hold_node=${HOLD_NODE}"
    echo "keep_hold=${KEEP_HOLD}"
    echo ""
    echo "Resume train on existing hold:"
    echo "  USE_EXISTING_HOLD=${HOLD_JOB} RUN_DIR=${RUN_DIR} EXPERIMENT_NAME=${EXPERIMENT_NAME} \\"
    echo "    RESUME_FROM_STEP=${RESUME_FROM_STEP:-<step>} NODELIST_ENV=<env_node> \\"
    echo "    bash ${ROOT}/launch_hold_train_resume.sh"
    echo ""
    echo "Manual srun:"
    echo "  srun --jobid=${HOLD_JOB} --overlap --nodes=1 --ntasks=1 -w ${HOLD_NODE} bash -lc '...'"
  } | tee -a "${RUN_DIR}/hold_job.txt"
}

wait_for_hold_running() {
  for _ in $(seq 1 360); do
    state=$("$SLURM"/squeue -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
    [ "${state}" = "RUNNING" ] && return 0
    [ -z "${state}" ] && { echo "ERROR hold ${HOLD_JOB} disappeared"; return 1; }
    sleep 5
  done
  echo "ERROR timed out waiting for hold ${HOLD_JOB} to run"
  return 1
}

resolve_hold_node() {
  HOLD_NODE=$("$SLURM"/squeue -j "${HOLD_JOB}" -h -o '%N' 2>/dev/null || true)
  if [ -z "${HOLD_NODE}" ]; then
    echo "ERROR hold job ${HOLD_JOB} not in queue"
    return 1
  fi
}

if [ -n "${USE_EXISTING_HOLD}" ]; then
  HOLD_JOB="${USE_EXISTING_HOLD}"
  echo "reuse hold job: ${HOLD_JOB}"
  if ! resolve_hold_node; then
    exit 1
  fi
  echo "hold node: ${HOLD_NODE}"
else
  HOLD_ARGS=(--parsable --account=peilab --partition=preempt)
  [ -n "${NODELIST_HOLD}" ] && HOLD_ARGS+=(--nodelist="${NODELIST_HOLD}")
  HOLD_JOB=$("$SLURM"/sbatch "${HOLD_ARGS[@]}" "${ROOT}/hold_preempt_1n8g.slurm" | awk '{print $NF}')
  echo "hold job: ${HOLD_JOB}"
  if ! wait_for_hold_running; then
    exit 1
  fi
  resolve_hold_node
  echo "hold node: ${HOLD_NODE}"
fi

print_hold_info

if [ "${HOLD_ONLY}" = "1" ]; then
  echo "HOLD_ONLY=1; hold kept, exiting."
  exit 0
fi

rm -f "${ENV_CONTROL_DIR}/failed" "${ENV_CONTROL_DIR}/ready"
mkdir -p "${ENV_CONTROL_DIR}"
: > "${ENV_CONTROL_DIR}/env_urls.txt"
: > "${ENV_CONTROL_DIR}/env_hosts.txt"

ENV_ARGS=(--parsable --partition="${ENV_PARTITION}" --export=ALL,RUN_DIR="${RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",CONTROL_DIR="${ENV_CONTROL_DIR}")
[ -n "${NODELIST_ENV}" ] && ENV_ARGS+=(--nodelist="${NODELIST_ENV}")
ENV_JOB=$("$SLURM"/sbatch "${ENV_ARGS[@]}" "${ROOT}/env_external_4gpu.slurm" | awk '{print $NF}')
echo "env job: ${ENV_JOB}"

ENV_READY=0
for _ in $(seq 1 360); do
  if [ -f "${ENV_CONTROL_DIR}/failed" ]; then
    echo "ERROR external env failed; hold ${HOLD_JOB} on ${HOLD_NODE} kept (KEEP_HOLD=${KEEP_HOLD})"
    cancel_env_job
    print_hold_info
    exit 4
  fi
  if [ -f "${ENV_CONTROL_DIR}/ready" ] && [ -s "${ENV_CONTROL_DIR}/env_urls.txt" ]; then
    ENV_READY=1
    break
  fi
  sleep 5
done
if [ "${ENV_READY}" != "1" ]; then
  echo "ERROR timed out waiting for env; hold ${HOLD_JOB} on ${HOLD_NODE} kept (KEEP_HOLD=${KEEP_HOLD})"
  cancel_env_job
  print_hold_info
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

cancel_env_job
release_hold_if_requested
if [ "${KEEP_HOLD}" = "1" ]; then
  print_hold_info
fi
exit "${RC}"
