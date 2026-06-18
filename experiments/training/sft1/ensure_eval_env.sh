#!/bin/bash
# Ensure a single healthy SFT1 eval env (4 GPU AI2-THOR servers) before rollout eval.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/training/sft1
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
SQUEUE=/cm/shared/apps/slurm/current/bin/squeue
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

ROLLOUT_ENV_RUN=${ROLLOUT_RUN_DIR}
ENV_CONTROL=${ROLLOUT_ENV_RUN}/external_env_4gpu
ENV_URL_FILE=${ENV_CONTROL}/env_urls.txt
ENV_NODE=${ENV_NODE:-}
PORT_BASE=${PORT_BASE:-8500}
WAIT_SEC=${WAIT_SEC:-1800}
LOG=${ENV_CONTROL}/ensure.log

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

running_env_jobs() {
  $SQUEUE -u "${USER:-csejzhang}" -h -o "%i %j %N" 2>/dev/null | awk '/sft1-env-v79/ {print $1, $3}'
}

cancel_extra_env_jobs() {
  local keep=${1:-}
  local j _node
  while read -r j _node; do
    [ -z "${j}" ] && continue
    if [ -n "${keep}" ] && [ "${j}" = "${keep}" ]; then
      continue
    fi
    log "scancel duplicate env job ${j} on ${_node}"
    $SCANCEL "${j}" 2>/dev/null || true
  done < <(running_env_jobs)
}

submit_env() {
  local extra=()
  if [ -n "${ENV_NODE}" ]; then
    extra+=(--nodelist="${ENV_NODE}")
  fi
  $SLURM --account=peilab "${extra[@]}" \
    --export=ALL,PORT_BASE="${PORT_BASE}" \
    "${SCRIPTDIR}/env_external_4gpu.slurm" | awk '{print $NF}'
}

mkdir -p "${ENV_CONTROL}"
touch "${LOG}"

mapfile -t jobs < <(running_env_jobs)
if [ "${#jobs[@]}" -gt 1 ]; then
  keep_id=$(echo "${jobs[0]}" | awk '{print $1}')
  cancel_extra_env_jobs "${keep_id}"
fi

if [ -f "${ENV_CONTROL}/ready" ] && [ -s "${ENV_URL_FILE}" ] && [ "${#jobs[@]}" -ge 1 ]; then
  log "env ready marker present with running job"
  exit 0
fi

if [ "${#jobs[@]}" -ge 1 ]; then
  jid=$(echo "${jobs[0]}" | awk '{print $1}')
  log "waiting for env job ${jid} to publish ready"
  for _ in $(seq 1 $((WAIT_SEC / 5))); do
    if [ -f "${ENV_CONTROL}/failed" ]; then
      log "env failed; will restart"
      $SCANCEL "${jid}" 2>/dev/null || true
      break
    fi
    if [ -f "${ENV_CONTROL}/ready" ] && [ -s "${ENV_URL_FILE}" ]; then
      log "env job ${jid} ready"
      cat "${ENV_URL_FILE}" | tee -a "${LOG}"
      exit 0
    fi
    sleep 5
  done
fi

cancel_extra_env_jobs
J=$(submit_env)
log "submitted env job ${J} on ${ENV_NODE}"

for _ in $(seq 1 $((WAIT_SEC / 5))); do
  if [ -f "${ENV_CONTROL}/failed" ]; then
    log "ERROR env job ${J} failed"
    exit 1
  fi
  if [ -f "${ENV_CONTROL}/ready" ] && [ -s "${ENV_URL_FILE}" ]; then
    log "env job ${J} ready"
    cat "${ENV_URL_FILE}" | tee -a "${LOG}"
    exit 0
  fi
  sleep 5
done

log "ERROR timed out waiting for env job ${J}"
exit 1
