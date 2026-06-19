#!/bin/bash
#SBATCH --job-name=sft2-ckpt-eval-watch
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${TRAIN_OUT:?TRAIN_OUT required}"

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/training/sft2
SFT1=${ROOT}/experiments/training/sft1
EVAL_NODE=${EVAL_NODE:-dgx-32}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-sft2_ckpt}
ROLLOUT_RUN_DIR=${ROLLOUT_RUN_DIR:-${ROOT}/outputs/experiments/training/baseline/2026-06-18/vagen_nav_fresh_4env8train}
TRAIN_JOB_ID=${TRAIN_JOB_ID:-}
POLL_SEC=${POLL_SEC:-120}
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SQUEUE=/cm/shared/apps/slurm/current/bin/squeue
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

log() { echo "[$(date -Iseconds)] $*"; }

ckpt_ready() {
  local d=$1
  [[ -d "${d}" && -f "${d}/config.json" ]]
}

list_ready_epochs() {
  for d in "${TRAIN_OUT}"/epoch_*; do
    [[ -d "${d}" ]] || continue
    ckpt_ready "${d}" || continue
    basename "${d}" | sed 's/epoch_//'
  done | sort -n
}

submit_eval() {
  local tag=$1
  local model_path=$2
  local eval_tag="${EVAL_TAG_PREFIX}_${tag}"
  local stamp="${TRAIN_OUT}/.eval_submitted_${eval_tag}"
  if [[ -f "${stamp}" ]]; then
    return 0
  fi
  log "submit eval tag=${eval_tag} model=${model_path}"
  ${SLURM} --parsable --account=peilab --partition=normal \
    --nodelist="${EVAL_NODE}" --gres=gpu:2 --mem=180G --cpus-per-task=56 \
    --job-name="sft2-eval-${tag}" \
    --export=ALL,EVAL_TAG="${eval_tag}",MODEL_PATH="${model_path}",SFT1_RUNS_ROOT="${TRAIN_OUT}/eval_rollouts",ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR}" \
    "${REPO}/experiments/training/sft2/eval_greedy_valtest.slurm"
  touch "${stamp}"
}

train_still_running() {
  if [[ -n "${TRAIN_JOB_ID}" ]] && ${SQUEUE} -j "${TRAIN_JOB_ID}" -h 2>/dev/null | grep -q .; then
    return 0
  fi
  [[ ! -f "${TRAIN_OUT}/sft2_done.flag" ]]
}

LAST_BEST_MTIME=0
log "watcher start TRAIN_OUT=${TRAIN_OUT} EVAL_NODE=${EVAL_NODE} TRAIN_JOB_ID=${TRAIN_JOB_ID}"

while train_still_running || true; do
  for ep in $(list_ready_epochs); do
    tag="epoch_$(printf '%03d' "${ep}")"
    submit_eval "${tag}" "${TRAIN_OUT}/${tag}"
  done
  if ckpt_ready "${TRAIN_OUT}/best"; then
    best_mtime=$(stat -c %Y "${TRAIN_OUT}/best/config.json" 2>/dev/null || echo 0)
    if (( best_mtime > LAST_BEST_MTIME )); then
      submit_eval "best" "${TRAIN_OUT}/best"
      LAST_BEST_MTIME=${best_mtime}
    fi
  fi
  if ! train_still_running; then
    break
  fi
  sleep "${POLL_SEC}"
done

# Final sweep after training ends.
for ep in $(list_ready_epochs); do
  tag="epoch_$(printf '%03d' "${ep}")"
  submit_eval "${tag}" "${TRAIN_OUT}/${tag}"
done
if ckpt_ready "${TRAIN_OUT}/best"; then
  submit_eval "best" "${TRAIN_OUT}/best"
fi

log "watcher done"
