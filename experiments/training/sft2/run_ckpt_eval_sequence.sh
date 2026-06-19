#!/usr/bin/env bash
# Run greedy rollout eval for each checkpoint sequentially on a hold allocation.
set -euo pipefail

REPO="${REPO:-/project/peilab/atst/nimloth}"
: "${TRAIN_OUT:?TRAIN_OUT required}"
: "${HOLD_JOB:?HOLD_JOB required}"

EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-sft2_ckpt}"
ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR:-${REPO}/outputs/experiments/training/baseline/2026-06-18/vagen_nav_fresh_4env8train}"
INIT_MODEL="${INIT_MODEL:-}"
EVAL_SCRIPT="${REPO}/experiments/training/sft2/eval_greedy_valtest.slurm"
SFT1_RUNS_ROOT="${TRAIN_OUT}/eval_rollouts"
LOG_ROOT="${TRAIN_OUT}/eval_rollouts"
mkdir -p "${LOG_ROOT}" "${REPO}/.home/.ssh"

ckpt_ready() {
  local d=$1
  [[ -d "${d}" && -f "${d}/config.json" && -f "${d}/model.safetensors.index.json" ]]
}

summary_exists() {
  local tag=$1
  [[ -f "${LOG_ROOT}/sft2_eval_${tag}/summary_0.json" ]]
}

run_one() {
  local tag=$1 model=$2 idx=$3
  local eval_tag="${EVAL_TAG_PREFIX}_${tag}"
  if summary_exists "${eval_tag}"; then
    echo "skip ${eval_tag} (summary exists)"
    return 0
  fi
  local log="${LOG_ROOT}/seq_${eval_tag}.log"
  echo "=== eval ${eval_tag} at $(date) ===" | tee -a "${log}"
  export EVAL_TAG="${eval_tag}" MODEL_PATH="${model}" TRAIN_OUT="${TRAIN_OUT}"
  export EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX}"
  export SFT1_RUNS_ROOT="${SFT1_RUNS_ROOT}" ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR}"
  export SLURM_JOB_ID="$((HOLD_JOB * 100 + idx))" SLURM_CPUS_PER_TASK=56
  export CUDA_VISIBLE_DEVICES=0,1
  export RAY_TMPDIR="/tmp/sft2_eval_${eval_tag}"
  export RAY_PORT=$((6450 + idx * 11))
  bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${log}"
}

declare -a TAGS=() PATHS=()
add_ckpt() {
  local tag=$1 path=$2 i
  for i in "${!PATHS[@]}"; do
    [[ "${PATHS[$i]}" == "${path}" ]] && return 0
  done
  TAGS+=("${tag}")
  PATHS+=("${path}")
}

if [ -z "${INIT_MODEL}" ]; then
  for log in "${TRAIN_OUT}"/sft2_train_*.log; do
    [ -f "${log}" ] || continue
    INIT_MODEL="$(grep -m1 '"init_model"' "${log}" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['init_model'])" 2>/dev/null || true)"
    [ -n "${INIT_MODEL}" ] && break
  done
fi
[ -n "${INIT_MODEL}" ] && ckpt_ready "${INIT_MODEL}" && add_ckpt "init" "${INIT_MODEL}"
for d in "${TRAIN_OUT}"/epoch_* "${TRAIN_OUT}"/best; do
  [ -d "${d}" ] || continue
  ckpt_ready "${d}" || continue
  add_ckpt "$(basename "${d}")" "${d}"
done

idx=0
for i in "${!TAGS[@]}"; do
  run_one "${TAGS[$i]}" "${PATHS[$i]}" "${idx}"
  idx=$((idx + 1))
done

python3 "${REPO}/experiments/training/sft2/summarize_ckpt_evals.py" \
  --train-out "${TRAIN_OUT}" --tag-prefix "${EVAL_TAG_PREFIX}" \
  --out "${LOG_ROOT}/summary_all.json"

if [ "${UPLOAD_WANDB:-1}" = "1" ] && [ -n "${WANDB_API_KEY:-}" ]; then
  python3 "${REPO}/experiments/training/sft2/upload_sft2_eval_wandb.py" \
    --train-out "${TRAIN_OUT}" --tag-prefix "${EVAL_TAG_PREFIX}" \
    --summary-all "${LOG_ROOT}/summary_all.json" || true
fi
