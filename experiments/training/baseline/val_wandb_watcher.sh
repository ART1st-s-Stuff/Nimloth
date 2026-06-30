#!/usr/bin/env bash
# Long-running watcher: poll TRAIN_RUN_DIR for new checkpoints, run val_only, log to wandb.
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline

: "${TRAIN_RUN_DIR:?TRAIN_RUN_DIR required}"
: "${ENV_URL_FILE:?ENV_URL_FILE required}"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
VAL_RUN_DIR=${VAL_RUN_DIR:-${TRAIN_RUN_DIR}/val_wandb_watcher}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-300}
VAL_EVERY=${VAL_EVERY:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-50}
STATE_FILE=${VAL_RUN_DIR}/last_val_step.txt

mkdir -p "${VAL_RUN_DIR}" "${VAL_RUN_DIR}/logs"
CKPT_DIR="${TRAIN_RUN_DIR}/checkpoints"

# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"

{
  echo "=== val wandb watcher starting at $(date) ==="
  echo "TRAIN_RUN_DIR=${TRAIN_RUN_DIR}"
  echo "VAL_RUN_DIR=${VAL_RUN_DIR}"
  echo "ENV_URL_FILE=${ENV_URL_FILE}"
  echo "POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC} VAL_EVERY=${VAL_EVERY}"
} | tee "${VAL_RUN_DIR}/val_wandb_watcher.log"

wait_for_env() {
  for _ in $(seq 1 360); do
    if [ ! -s "${ENV_URL_FILE}" ]; then
      sleep 5
      continue
    fi
    local ok=1 url
    while read -r url; do
      [ -z "$url" ] && continue
      if ! curl -fsS --max-time 10 "${url}/health" >/dev/null 2>&1; then
        ok=0
        break
      fi
    done < "${ENV_URL_FILE}"
    if [ "$ok" -eq 1 ]; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR timed out waiting for env URLs" | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"
  exit 4
}

checkpoint_ready() {
  local step=$1
  local actor_dir="${CKPT_DIR}/global_step_${step}/actor"
  local critic_dir="${CKPT_DIR}/global_step_${step}/critic"
  [ -d "${actor_dir}" ] && [ -d "${critic_dir}" ] || return 1
  compgen -G "${actor_dir}/model_world_size_*_rank_*.pt" >/dev/null || \
    [ -d "${actor_dir}/huggingface" ] || return 1
  compgen -G "${critic_dir}/model_world_size_*_rank_*.pt" >/dev/null || return 1
  return 0
}

should_val_step() {
  local step=$1
  [ "${step}" -ge 1 ] || return 1
  if [ "$VAL_EVERY" -le 1 ]; then
    return 0
  fi
  [ $((step % VAL_EVERY)) -eq 0 ]
}

wait_for_env

LAST_VAL=0
if [ -f "${STATE_FILE}" ]; then
  LAST_VAL=$(tr -d '[:space:]' < "${STATE_FILE}" || echo 0)
fi

while true; do
  latest=$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [[ "${latest}" =~ ^[0-9]+$ ]]; then
    step=$((LAST_VAL + 1))
    while [ "${step}" -le "${latest}" ]; do
      if ! checkpoint_ready "${step}"; then
        ckpt_path="${CKPT_DIR}/global_step_${step}"
        if [ -d "${ckpt_path}" ]; then
          # Checkpoint still being written; retry on next poll.
          break
        fi
        if [ "${step}" -le "${latest}" ]; then
          echo "WARNING checkpoint step ${step} missing (likely pruned before val); skipping" \
            | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"
        fi
        step=$((step + 1))
        continue
      fi
      if ! should_val_step "${step}"; then
        step=$((step + 1))
        continue
      fi
      if [ "${step}" -le "${LAST_VAL}" ]; then
        step=$((step + 1))
        continue
      fi
        echo "=== checkpoint ${step}; running val ===" | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"
        set +e
        TRAIN_RUN_DIR="${TRAIN_RUN_DIR}" \
          VAL_RUN_DIR="${VAL_RUN_DIR}" \
          ENV_URL_FILE="${ENV_URL_FILE}" \
          CHECKPOINT_STEP="${step}" \
          EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
          bash "${SCRIPTDIR}/run_val_checkpoint_wandb.sh"
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then
          echo "${step}" > "${STATE_FILE}"
          LAST_VAL="${step}"
          if [ "${PRUNE_CHECKPOINTS:-0}" = "1" ]; then
            KEEP_EVERY="${KEEP_EVERY:-10}" bash "${SCRIPTDIR}/prune_checkpoints_policy.sh" "${TRAIN_RUN_DIR}" \
              2>&1 | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log" || true
          fi
        else
          echo "WARNING val for step ${step} failed rc=${rc}" | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"
        fi
      step=$((step + 1))
    done
    if [ "${latest}" -ge "${MAX_TRAIN_STEPS}" ] && [ "${LAST_VAL}" -ge "${MAX_TRAIN_STEPS}" ]; then
      echo "=== reached MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}; watcher exiting ===" | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"
      exit 0
    fi
  fi
  sleep "${POLL_INTERVAL_SEC}"
done
