#!/bin/bash
#SBATCH --job-name=sft1-ckpt-eval-watch
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${TRAIN_OUT:?TRAIN_OUT required}"
: "${BASE_MODEL:?BASE_MODEL required}"

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
EVAL_NODE=${EVAL_NODE:-dgx-09}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-alltrain_8gpu_lora}
TRAIN_JOB_ID=${TRAIN_JOB_ID:-}
POLL_SEC=${POLL_SEC:-90}
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SQUEUE=/cm/shared/apps/slurm/current/bin/squeue
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH
source ${ROOT}/.venv/bin/activate

log() { echo "[$(date -Iseconds)] $*"; }

ckpt_ready() {
  local d=$1
  [[ -f "${d}/adapter_config.json" ]] && [[ -f "${d}/adapter_model.safetensors" || -f "${d}/adapter_model.bin" ]]
}

max_ready_epoch() {
  local best_ep=-1 best_dir=""
  for d in "${TRAIN_OUT}"/epoch_*; do
    [[ -d "${d}" ]] || continue
    ckpt_ready "${d}" || continue
    ep=$(basename "${d}" | sed 's/epoch_//')
    epn=$((10#${ep}))
    if (( epn > best_ep )); then
      best_ep=${epn}
      best_dir=${d}
    fi
  done
  if (( best_ep >= 0 )); then
    echo "${best_ep} ${best_dir}"
  fi
}

submit_eval() {
  local ep=$1
  local model_path=$2
  local tag="${EVAL_TAG_PREFIX}_epoch$(printf '%03d' "${ep}")"
  log "submit eval tag=${tag} model=${model_path}"
  ${SLURM} --account=peilab --nodelist="${EVAL_NODE}" \
    --job-name="sft1-eval-ep${ep}" \
    --export=ALL,EVAL_TAG="${tag}",MODEL_PATH="${model_path}" \
    "${SCRIPTDIR}/sft1_eval_vagen79_greedy_valtest.slurm"
}

train_still_running() {
  if [[ -n "${TRAIN_JOB_ID}" ]] && ${SQUEUE} -j "${TRAIN_JOB_ID}" -h 2>/dev/null | grep -q .; then
    return 0
  fi
  [[ ! -f "${TRAIN_OUT}/train_done.flag" ]]
}

LAST_EVAL_EP=-1
log "watcher start TRAIN_OUT=${TRAIN_OUT} EVAL_NODE=${EVAL_NODE} TRAIN_JOB_ID=${TRAIN_JOB_ID}"

while train_still_running || true; do
  read -r ep dir < <(max_ready_epoch || true)
  if [[ -n "${ep:-}" ]] && (( ep > LAST_EVAL_EP )); then
    stamp="${TRAIN_OUT}/.eval_submitted_epoch_$(printf '%03d' "${ep}")"
    if [[ ! -f "${stamp}" ]]; then
      merged="${dir}/hf_merged"
      if [[ ! -f "${merged}/config.json" ]]; then
        log "merge LoRA epoch ${ep}"
        python3 "${SCRIPTDIR}/merge_sft1_lora_ckpt.py" \
          --base-model "${BASE_MODEL}" \
          --adapter-dir "${dir}" \
          --out-dir "${merged}"
      fi
      submit_eval "${ep}" "${merged}"
      touch "${stamp}"
      LAST_EVAL_EP=${ep}
      log "queued eval for epoch ${ep}"
    else
      LAST_EVAL_EP=${ep}
    fi
  fi
  if ! train_still_running; then
    break
  fi
  sleep "${POLL_SEC}"
done

log "watcher done last_eval_epoch=${LAST_EVAL_EP}"
