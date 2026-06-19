#!/usr/bin/env bash
# Submit greedy VAGEN val/test rollout eval for each SFT2 checkpoint in parallel.
set -euo pipefail

REPO="${REPO:-/project/peilab/atst/nimloth}"
SCRIPTDIR="${REPO}/experiments/training/sft2"
SFT1="${REPO}/experiments/training/sft1"
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft2/slurm"

: "${TRAIN_OUT:?TRAIN_OUT required}"

EVAL_NODE="${EVAL_NODE:-dgx-32}"
EVAL_GPUS="${EVAL_GPUS:-2}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-sft2_ckpt}"
ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR:-${REPO}/outputs/experiments/training/baseline/2026-06-18/vagen_nav_fresh_4env8train}"
SFT1_RUNS_ROOT="${TRAIN_OUT}/eval_rollouts"
INIT_MODEL="${INIT_MODEL:-}"

if [ -z "${INIT_MODEL}" ] && [ -f "${TRAIN_OUT}/sft2_train_456005.log" ]; then
  INIT_MODEL="$(grep -m1 '"init_model"' "${TRAIN_OUT}/sft2_train_456005.log" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['init_model'])" 2>/dev/null || true)"
fi
if [ -z "${INIT_MODEL}" ]; then
  for log in "${TRAIN_OUT}"/sft2_train_*.log; do
    [ -f "${log}" ] || continue
    INIT_MODEL="$(grep -m1 '"init_model"' "${log}" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['init_model'])" 2>/dev/null || true)"
    [ -n "${INIT_MODEL}" ] && break
  done
fi

ckpt_ready() {
  local d=$1
  [[ -d "${d}" && -f "${d}/config.json" ]]
}

declare -a CKPT_TAGS=()
declare -a CKPT_PATHS=()

add_ckpt() {
  local tag=$1 path=$2
  local i
  for i in "${!CKPT_PATHS[@]}"; do
    if [ "${CKPT_PATHS[$i]}" = "${path}" ]; then
      return 0
    fi
  done
  CKPT_TAGS+=("${tag}")
  CKPT_PATHS+=("${path}")
}

if [ -n "${INIT_MODEL}" ] && ckpt_ready "${INIT_MODEL}"; then
  add_ckpt "init" "${INIT_MODEL}"
fi
for d in "${TRAIN_OUT}"/epoch_* "${TRAIN_OUT}"/best; do
  [ -d "${d}" ] || continue
  ckpt_ready "${d}" || continue
  add_ckpt "$(basename "${d}")" "${d}"
done

if [ "${#CKPT_PATHS[@]}" -eq 0 ]; then
  echo "ERROR: no ready checkpoints under ${TRAIN_OUT}" >&2
  exit 1
fi

echo "=== SFT2 parallel ckpt eval ==="
echo "TRAIN_OUT=${TRAIN_OUT}"
echo "EVAL_NODE=${EVAL_NODE} (${EVAL_GPUS} GPU/job)"
echo "ROLLOUT_RUN_DIR=${ROLLOUT_RUN_DIR}"
echo "checkpoints: ${#CKPT_PATHS[@]}"

for i in "${!CKPT_TAGS[@]}"; do
  tag="${CKPT_TAGS[$i]}"
  model="${CKPT_PATHS[$i]}"
  eval_tag="${EVAL_TAG_PREFIX}_${tag}"
  stamp="${TRAIN_OUT}/.eval_submitted_${eval_tag}"
  if [ -f "${stamp}" ]; then
    echo "skip already submitted ${eval_tag}"
    continue
  fi
  jobid=$("${SLURM}" --parsable --account=peilab --partition=normal \
    --nodelist="${EVAL_NODE}" --gres="gpu:${EVAL_GPUS}" --mem=180G \
    --cpus-per-task=56 --job-name="sft2-eval-${tag}" \
    --export=ALL,EVAL_TAG="${eval_tag}",MODEL_PATH="${model}",SFT1_RUNS_ROOT="${SFT1_RUNS_ROOT}",ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR}" \
    "${REPO}/experiments/training/sft2/eval_greedy_valtest.slurm")
  touch "${stamp}"
  echo "submitted ${eval_tag} job=${jobid} model=${model}"
done

echo "eval outputs: ${SFT1_RUNS_ROOT}/sft1_eval_${EVAL_TAG_PREFIX}_*/summary_0.json"
