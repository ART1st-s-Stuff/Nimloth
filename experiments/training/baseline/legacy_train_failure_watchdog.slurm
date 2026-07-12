#!/bin/bash
#SBATCH --job-name=vagen-restart-watch
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Submit this job with --dependency=afternotok:<train-job-id>. It only restarts
# the train when the failed run contains the exact none_dealloc signature.
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/baseline
TRAIN_SLURM=${TRAIN_SLURM:-${SCRIPTDIR}/legacy_train_external_service.slurm}
WATCHDOG_SLURM=${WATCHDOG_SLURM:-${SCRIPTDIR}/legacy_train_failure_watchdog.slurm}
SBATCH_BIN=${SBATCH_BIN:-sbatch}
SACCT_BIN=${SACCT_BIN:-sacct}

PREVIOUS_JOB_ID=${PREVIOUS_JOB_ID:?set PREVIOUS_JOB_ID}
PREVIOUS_RUN_DIR=${PREVIOUS_RUN_DIR:?set PREVIOUS_RUN_DIR}
RETRY_ROOT_DIR=${RETRY_ROOT_DIR:?set RETRY_ROOT_DIR}
RETRY_BASENAME=${RETRY_BASENAME:-vagen_legacy_auto_resume}
RETRY_INDEX=${RETRY_INDEX:-1}
MAX_AUTO_RETRIES=${MAX_AUTO_RETRIES:-8}
TARGET_STEP=${TARGET_STEP:-330}

SERVICE_BASE_URL=${SERVICE_BASE_URL:?set SERVICE_BASE_URL}
TRAIN_CONFIG=${TRAIN_CONFIG:?set TRAIN_CONFIG}
VAL_CONFIG=${VAL_CONFIG:?set VAL_CONFIG}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:?set ACTOR_MODEL_PATH}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-${ACTOR_MODEL_PATH}}
TRAIN_PARTITION=${TRAIN_PARTITION:-preempt}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-8}
TRAIN_NODES=${TRAIN_NODES:-1}
TRAIN_CPUS_PER_TASK=${TRAIN_CPUS_PER_TASK:-128}
TRAIN_MEM=${TRAIN_MEM:-512G}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-128}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-1}
ENABLE_WANDB=${ENABLE_WANDB:-1}
ROLLOUT_MAX_TURNS=${ROLLOUT_MAX_TURNS:-20}
VLLM_MAX_RESPONSE_PER_TURN=${VLLM_MAX_RESPONSE_PER_TURN:-1024}
VAGEN_NON_RESPONSE_TOKEN_RESERVE=${VAGEN_NON_RESPONSE_TOKEN_RESERVE:-11000}
ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-32000}
VAGEN_TRAJECTORY_TRUNCATION=${VAGEN_TRAJECTORY_TRUNCATION:-error}
RESTORE_DATALOADER_STATE=${RESTORE_DATALOADER_STATE:-true}
REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-false}

PREVIOUS_LOG=${PREVIOUS_LOG:-${PREVIOUS_RUN_DIR}/legacy_train.log}
PREVIOUS_CHECKPOINT_DIR=${PREVIOUS_CHECKPOINT_DIR:-${PREVIOUS_RUN_DIR}/checkpoints}
STATE_LOG=${STATE_LOG:-${RETRY_ROOT_DIR}/auto_resume_state.log}
mkdir -p "${RETRY_ROOT_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${STATE_LOG}"
}

if [ ! -f "${PREVIOUS_LOG}" ]; then
  log "STOP previous log missing: ${PREVIOUS_LOG}"
  exit 10
fi
PREVIOUS_STATE=$("${SACCT_BIN}" -X -j "${PREVIOUS_JOB_ID}" -n -o State 2>/dev/null | awk 'NF {print $1; exit}' || true)
IS_NONE_DEALLOC=false
IS_PREEMPTED=false
grep -Fq 'Fatal Python error: none_dealloc: deallocating None' "${PREVIOUS_LOG}" && IS_NONE_DEALLOC=true
[[ "${PREVIOUS_STATE}" == PREEMPTED* ]] && IS_PREEMPTED=true
if [ "${IS_NONE_DEALLOC}" != true ] && [ "${IS_PREEMPTED}" != true ]; then
  log "STOP job=${PREVIOUS_JOB_ID} state=${PREVIOUS_STATE:-unknown} failed without exact none_dealloc signature and was not preempted"
  exit 11
fi
RESTART_REASON=none_dealloc
[ "${IS_PREEMPTED}" = true ] && RESTART_REASON=preempted
if [ ! -f "${PREVIOUS_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
  log "STOP latest checkpoint marker missing: ${PREVIOUS_CHECKPOINT_DIR}"
  exit 12
fi
LATEST_STEP=$(tr -d '[:space:]' < "${PREVIOUS_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
if ! [[ "${LATEST_STEP}" =~ ^[0-9]+$ ]]; then
  log "STOP invalid checkpoint marker: ${LATEST_STEP}"
  exit 13
fi
if [ "${LATEST_STEP}" -ge "${TARGET_STEP}" ]; then
  log "DONE target already reached: latest=${LATEST_STEP} target=${TARGET_STEP}"
  exit 0
fi
if [ "${RETRY_INDEX}" -gt "${MAX_AUTO_RETRIES}" ]; then
  log "STOP retry limit reached: index=${RETRY_INDEX} max=${MAX_AUTO_RETRIES} latest=${LATEST_STEP}"
  exit 14
fi

SOURCE_STEP_DIR=${PREVIOUS_CHECKPOINT_DIR}/global_step_${LATEST_STEP}
for role in actor critic; do
  for ((rank=0; rank<TRAIN_GPUS_PER_NODE; rank++)); do
    for kind in model optim extra_state; do
      shard=${SOURCE_STEP_DIR}/${role}/${kind}_world_size_${TRAIN_GPUS_PER_NODE}_rank_${rank}.pt
      if [ ! -f "${shard}" ]; then
        log "STOP strict checkpoint shard missing: ${shard}"
        exit 15
      fi
    done
  done
done
if [ "${RESTORE_DATALOADER_STATE}" = true ] && [ ! -f "${SOURCE_STEP_DIR}/data.pt" ]; then
  log "STOP dataloader restore requested but data.pt is missing: ${SOURCE_STEP_DIR}"
  exit 16
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RETRY_TAG=$(printf 'retry%02d_from%s_%s' "${RETRY_INDEX}" "${LATEST_STEP}" "${STAMP}")
RUN_DIR=${RETRY_ROOT_DIR}/${RETRY_BASENAME}_${RETRY_TAG}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
LOG_FILE=${RUN_DIR}/legacy_train.log
EXPERIMENT_NAME=${RETRY_BASENAME}_${RETRY_TAG}
mkdir -p "${SAVE_CHECKPOINT_DIR}"
ln -s "${SOURCE_STEP_DIR}" "${SAVE_CHECKPOINT_DIR}/global_step_${LATEST_STEP}"
printf '%s\n' "${LATEST_STEP}" > "${SAVE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

cat > "${RUN_DIR}/README.md" <<EOF
# Automatic retry ${RETRY_INDEX}

- watchdog job: ${SLURM_JOB_ID:-manual}
- failed predecessor job: ${PREVIOUS_JOB_ID}
- failed predecessor run: ${PREVIOUS_RUN_DIR}
- recognized restart reason: ${RESTART_REASON}
- strict ws${TRAIN_GPUS_PER_NODE} source: ${SOURCE_STEP_DIR}
- source step: ${LATEST_STEP}
- target step: ${TARGET_STEP}
- restore dataloader state: ${RESTORE_DATALOADER_STATE}
- fresh retry directory: ${RUN_DIR}
EOF

EXPORTS="ALL,REPO=${REPO},EXPERIMENT_NAME=${EXPERIMENT_NAME},RUN_DIR=${RUN_DIR},LOG_FILE=${LOG_FILE},SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR},SERVICE_BASE_URL=${SERVICE_BASE_URL},TRAIN_CONFIG=${TRAIN_CONFIG},VAL_CONFIG=${VAL_CONFIG},ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH},CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH},SOURCE_CHECKPOINT_STEP=${LATEST_STEP},TOTAL_STEPS=${TARGET_STEP},TOTAL_EPOCHS=${TOTAL_EPOCHS},TEST_FREQ=${TEST_FREQ},SAVE_FREQ=${SAVE_FREQ},TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE},TRAIN_NODES=${TRAIN_NODES},ENABLE_WANDB=${ENABLE_WANDB},RAY_NUM_CPUS=${RAY_NUM_CPUS},RESTORE_DATALOADER_STATE=${RESTORE_DATALOADER_STATE},REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE},ROLLOUT_MAX_TURNS=${ROLLOUT_MAX_TURNS},VLLM_MAX_RESPONSE_PER_TURN=${VLLM_MAX_RESPONSE_PER_TURN},VAGEN_NON_RESPONSE_TOKEN_RESERVE=${VAGEN_NON_RESPONSE_TOKEN_RESERVE},ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH},VAGEN_TRAJECTORY_TRUNCATION=${VAGEN_TRAJECTORY_TRUNCATION}"
TRAIN_JOB_ID=$("${SBATCH_BIN}" --parsable \
  --partition="${TRAIN_PARTITION}" \
  --nodes="${TRAIN_NODES}" --ntasks=1 \
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}" \
  --gres="gpu:${TRAIN_GPUS_PER_NODE}" --mem="${TRAIN_MEM}" \
  --export="${EXPORTS}" \
  "${TRAIN_SLURM}")
TRAIN_JOB_ID=${TRAIN_JOB_ID%%;*}
printf '%s\n' "${TRAIN_JOB_ID}" > "${RUN_DIR}/train_job_id.txt"
log "RESTART predecessor=${PREVIOUS_JOB_ID} reason=${RESTART_REASON} retry_job=${TRAIN_JOB_ID} step=${LATEST_STEP} run=${RUN_DIR}"

NEXT_INDEX=$((RETRY_INDEX + 1))
NEXT_EXPORTS="ALL,REPO=${REPO},PREVIOUS_JOB_ID=${TRAIN_JOB_ID},PREVIOUS_RUN_DIR=${RUN_DIR},RETRY_ROOT_DIR=${RETRY_ROOT_DIR},RETRY_BASENAME=${RETRY_BASENAME},RETRY_INDEX=${NEXT_INDEX},MAX_AUTO_RETRIES=${MAX_AUTO_RETRIES},TARGET_STEP=${TARGET_STEP},SERVICE_BASE_URL=${SERVICE_BASE_URL},TRAIN_CONFIG=${TRAIN_CONFIG},VAL_CONFIG=${VAL_CONFIG},ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH},CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH},TRAIN_PARTITION=${TRAIN_PARTITION},TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE},TRAIN_NODES=${TRAIN_NODES},TRAIN_CPUS_PER_TASK=${TRAIN_CPUS_PER_TASK},TRAIN_MEM=${TRAIN_MEM},RAY_NUM_CPUS=${RAY_NUM_CPUS},TOTAL_EPOCHS=${TOTAL_EPOCHS},TEST_FREQ=${TEST_FREQ},SAVE_FREQ=${SAVE_FREQ},ENABLE_WANDB=${ENABLE_WANDB},ROLLOUT_MAX_TURNS=${ROLLOUT_MAX_TURNS},VLLM_MAX_RESPONSE_PER_TURN=${VLLM_MAX_RESPONSE_PER_TURN},VAGEN_NON_RESPONSE_TOKEN_RESERVE=${VAGEN_NON_RESPONSE_TOKEN_RESERVE},ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH},VAGEN_TRAJECTORY_TRUNCATION=${VAGEN_TRAJECTORY_TRUNCATION},RESTORE_DATALOADER_STATE=${RESTORE_DATALOADER_STATE},REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE},SBATCH_BIN=${SBATCH_BIN},SACCT_BIN=${SACCT_BIN}"
NEXT_WATCHDOG_JOB_ID=$("${SBATCH_BIN}" --parsable \
  --dependency="afternotok:${TRAIN_JOB_ID}" --kill-on-invalid-dep=yes \
  --export="${NEXT_EXPORTS}" \
  "${WATCHDOG_SLURM}")
NEXT_WATCHDOG_JOB_ID=${NEXT_WATCHDOG_JOB_ID%%;*}
printf '%s\n' "${NEXT_WATCHDOG_JOB_ID}" > "${RUN_DIR}/watchdog_job_id.txt"
log "ARMED watchdog=${NEXT_WATCHDOG_JOB_ID} dependency=afternotok:${TRAIN_JOB_ID} next_index=${NEXT_INDEX}"
