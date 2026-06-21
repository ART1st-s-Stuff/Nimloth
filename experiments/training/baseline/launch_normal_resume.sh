#!/usr/bin/env bash
# Resume baseline training on normal partition (external env + train).
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
SLURM_BIN=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

RUN_DIR=${RUN_DIR:?RUN_DIR required}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
ENV_NODELIST=${ENV_NODELIST:-}
TRAIN_NODELIST=${TRAIN_NODELIST:-}
ENV_CONTROL_DIR=${ENV_CONTROL_DIR:-${RUN_DIR}/external_env_normal_resume}
TOTAL_STEPS=${TOTAL_STEPS:-50}
TEST_FREQ=${TEST_FREQ:-10}
RESUME_FROM_STEP=${RESUME_FROM_STEP:-}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-0}
TRAIN_NODES=${TRAIN_NODES:-1}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-8}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-6}

AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-6}

mkdir -p "${REPO}/outputs/experiments/training/baseline/slurm"

ENV_ARGS=(--parsable --partition=normal --export=ALL,RUN_DIR="${RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",CONTROL_DIR="${ENV_CONTROL_DIR}")
if [ -n "${ENV_NODELIST}" ]; then
  ENV_ARGS+=(--nodelist="${ENV_NODELIST}")
fi
ENV_JOB=$("$SLURM_BIN"/sbatch "${ENV_ARGS[@]}" "${SCRIPTDIR}/env_external_4gpu.slurm" | awk '{print $NF}')
echo "env job: ${ENV_JOB} CONTROL_DIR=${ENV_CONTROL_DIR}"

TRAIN_ARGS=(
  --parsable --partition=normal --nodes="${TRAIN_NODES}"
  --export=ALL,RUN_DIR="${RUN_DIR}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",ENV_CONTROL_DIR="${ENV_CONTROL_DIR}",TOTAL_STEPS="${TOTAL_STEPS}",TEST_FREQ="${TEST_FREQ}",TRAIN_NODES="${TRAIN_NODES}",TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE}",AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS}",AGENT_MAX_CONCURRENT_TRAJECTORIES="${AGENT_MAX_CONCURRENT_TRAJECTORIES}",RESUME_FROM_STEP="${RESUME_FROM_STEP}",PRUNE_CHECKPOINTS="${PRUNE_CHECKPOINTS}"
)
if [ -n "${TRAIN_NODELIST}" ]; then
  TRAIN_ARGS+=(--nodelist="${TRAIN_NODELIST}")
fi
TRAIN_JOB=$("$SLURM_BIN"/sbatch "${TRAIN_ARGS[@]}" "${SCRIPTDIR}/train_resume.slurm" | awk '{print $NF}')
echo "train resume job: ${TRAIN_JOB} nodes=${TRAIN_NODES}x${TRAIN_GPUS_PER_NODE}GPU"

cat >> "${RUN_DIR}/README.md" <<EOF

## normal resume $(date -Iseconds)
- env_job=${ENV_JOB} nodelist=${ENV_NODELIST:-auto} control=${ENV_CONTROL_DIR}
- train_job=${TRAIN_JOB} nodelist=${TRAIN_NODELIST:-auto} ${TRAIN_NODES}x${TRAIN_GPUS_PER_NODE}GPU
- from checkpoint: ${RUN_DIR}/checkpoints (resume_mode=${RESUME_FROM_STEP:+resume_path step ${RESUME_FROM_STEP}}${RESUME_FROM_STEP:-auto})
- prune_checkpoints=${PRUNE_CHECKPOINTS}
- total_steps=${TOTAL_STEPS} test_freq=${TEST_FREQ}
EOF

echo "Submitted normal resume for ${RUN_DIR}"
