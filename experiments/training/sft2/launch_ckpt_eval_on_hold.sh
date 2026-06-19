#!/usr/bin/env bash
# Launch greedy rollout eval jobs inside an existing hold allocation (srun --overlap).
set -euo pipefail

REPO="${REPO:-/project/peilab/atst/nimloth}"
SFT1="${REPO}/experiments/training/sft2"
EVAL_SCRIPT="${REPO}/experiments/training/sft2/eval_greedy_valtest.slurm"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
: "${TRAIN_OUT:?TRAIN_OUT required}"
: "${HOLD_JOB:?HOLD_JOB required}"

EVAL_NODE="${EVAL_NODE:-dgx-32}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-sft2_ckpt}"
ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR:-${REPO}/outputs/experiments/training/baseline/2026-06-18/vagen_nav_fresh_4env8train}"
INIT_MODEL="${INIT_MODEL:-}"
SFT1_RUNS_ROOT="${TRAIN_OUT}/eval_rollouts"
SRUN=/cm/shared/apps/slurm/current/bin/srun
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

ckpt_ready() {
  local d=$1
  [[ -d "${d}" && -f "${d}/config.json" && -f "${d}/model.safetensors.index.json" ]]
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

if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "ERROR: no ready checkpoints" >&2
  exit 1
fi

mkdir -p "${TRAIN_OUT}/eval_rollouts" "${REPO}/outputs/experiments/training/sft2/slurm"
echo "=== launch ${#PATHS[@]} eval jobs on hold ${HOLD_JOB} node ${EVAL_NODE} ==="

pids=()
gpu_base=0
running=0
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  model="${PATHS[$i]}"
  eval_tag="${EVAL_TAG_PREFIX}_${tag}"
  stamp="${TRAIN_OUT}/.eval_submitted_${eval_tag}"
  if [ -f "${stamp}" ]; then
    echo "skip ${eval_tag}"
    continue
  fi
  if (( running >= MAX_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
  g0=${gpu_base}
  g1=$((gpu_base + 1))
  gpu_base=$((gpu_base + 2))
  if (( gpu_base > 8 )); then
    gpu_base=0
  fi
  log="${TRAIN_OUT}/eval_rollouts/launch_${eval_tag}.log"
  echo "start ${eval_tag} gpus=${g0},${g1} -> ${log}"
  (
    export EVAL_TAG="${eval_tag}" MODEL_PATH="${model}" TRAIN_OUT="${TRAIN_OUT}"
    export EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX}"
    export SFT1_RUNS_ROOT="${SFT1_RUNS_ROOT}" ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR}"
    export SLURM_JOB_ID="$((HOLD_JOB * 10 + i))"
    export SLURM_CPUS_PER_TASK=14
    export CUDA_VISIBLE_DEVICES="${g0},${g1}"
    export RAY_TMPDIR="/tmp/sft2_eval_${eval_tag}_$$"
    export RAY_PORT=$((6400 + i * 17 + HOLD_JOB % 40))
    bash "${EVAL_SCRIPT}"
  ) >"${log}" 2>&1 &
  pids+=($!)
  running=$((running + 1))
  touch "${stamp}"
  sleep 8
done

echo "waiting for ${#pids[@]} eval jobs..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
python3 "${REPO}/experiments/training/sft2/summarize_ckpt_evals.py" \
  --train-out "${TRAIN_OUT}" --tag-prefix "${EVAL_TAG_PREFIX}" \
  --out "${TRAIN_OUT}/eval_rollouts/summary_all.json"

if [ "${UPLOAD_WANDB:-1}" = "1" ] && [ -n "${WANDB_API_KEY:-}" ]; then
  python3 "${REPO}/experiments/training/sft2/upload_sft2_eval_wandb.py" \
    --train-out "${TRAIN_OUT}" --tag-prefix "${EVAL_TAG_PREFIX}" \
    --summary-all "${TRAIN_OUT}/eval_rollouts/summary_all.json" || true
fi

exit "${fail}"
