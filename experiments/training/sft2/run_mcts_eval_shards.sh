#!/bin/bash

set -euo pipefail

: "${EVAL_WORKTREE:?EVAL_WORKTREE is required}"
: "${SFT2_CHECKPOINT:?SFT2_CHECKPOINT is required}"
: "${EVAL_OUTPUT:?EVAL_OUTPUT is required}"
: "${EVAL_SET:?EVAL_SET is required}"
: "${ENV_URL:?ENV_URL is required}"
: "${PY:?PY is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

SHARDS_PER_SET=${SHARDS_PER_SET:-2}
EPISODES_PER_SET=${EPISODES_PER_SET:-60}
SEED_OFFSET=${SEED_OFFSET:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.42}

if (( SHARDS_PER_SET < 1 || EPISODES_PER_SET % SHARDS_PER_SET != 0 )); then
  echo "invalid shard layout: episodes=${EPISODES_PER_SET} shards=${SHARDS_PER_SET}" >&2
  exit 2
fi

EPISODES_PER_SHARD=$((EPISODES_PER_SET / SHARDS_PER_SET))
CHILD_PIDS=()
CHILD_SHARDS=()
CHILD_ATTEMPTS=()

cleanup() {
  set +e
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

for (( shard_index=0; shard_index<SHARDS_PER_SET; shard_index++ )); do
  shard_name=$(printf 'shard_%02d' "${shard_index}")
  final_dir=${EVAL_OUTPUT}/eval_sets/${EVAL_SET}/${shard_name}
  if test -s "${final_dir}/rollout_summary.json"; then
    echo "resume_skip=${EVAL_SET}/${shard_name}"
    continue
  fi
  if test -e "${final_dir}"; then
    echo "refusing incomplete final shard directory: ${final_dir}" >&2
    exit 3
  fi

  attempt_dir=${EVAL_OUTPUT}/.attempts/${EVAL_SET}/${shard_name}
  log_path=${EVAL_OUTPUT}/${EVAL_SET}_${shard_name}_${SLURM_JOB_ID}.log
  shard_seed=$((SEED_OFFSET + shard_index * EPISODES_PER_SHARD))
  mkdir -p "${attempt_dir}"
  (
    export TRITON_CACHE_DIR=/tmp/triton_mcts_${SLURM_JOB_ID}_${EVAL_SET}_${shard_name}
    export XDG_CACHE_HOME=/tmp/xdg_mcts_${SLURM_JOB_ID}_${EVAL_SET}_${shard_name}
    export OMP_NUM_THREADS=8
    mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"
    cd "${EVAL_WORKTREE}"
    exec "${PY}" experiments/training/sft2/eval_mcts_rollout.py \
      --sft2-checkpoint "${SFT2_CHECKPOINT}" \
      --env-url "${ENV_URL}" \
      --output-dir "${attempt_dir}" --resume \
      --eval-sets "${EVAL_SET}" --split test \
      --episodes-per-eval-set "${EPISODES_PER_SHARD}" \
      --seed-offset "${shard_seed}" --max-steps 20 \
      --temperature 0.7 --top-p 0.95 --max-response-tokens 512 \
      --num-simulations 100 --exploration-constant 1.0 \
      --tensor-parallel-size 1 --planner-device cuda:0 \
      --max-model-len 32768 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --vllm-mm-processor-cache-gb 0
  ) > "${log_path}" 2>&1 &
  CHILD_PIDS+=("$!")
  CHILD_SHARDS+=("${shard_name}")
  CHILD_ATTEMPTS+=("${attempt_dir}")
  # 避免两个engine同时读取checkpoint和建立CUDA context产生瞬时峰值。
  if (( shard_index + 1 < SHARDS_PER_SET )); then
    sleep 8
  fi
done

failed=0
set +e
for index in "${!CHILD_PIDS[@]}"; do
  wait "${CHILD_PIDS[$index]}"
  code=$?
  shard_name=${CHILD_SHARDS[$index]}
  attempt_dir=${CHILD_ATTEMPTS[$index]}
  final_dir=${EVAL_OUTPUT}/eval_sets/${EVAL_SET}/${shard_name}
  if test "${code}" -eq 0 && test -s "${attempt_dir}/rollout_summary.json"; then
    mkdir -p "$(dirname "${final_dir}")"
    mv "${attempt_dir}" "${final_dir}"
    echo "shard_done=${EVAL_SET}/${shard_name}"
  else
    echo "shard_failed=${EVAL_SET}/${shard_name} exit=${code}" >&2
    failed=1
  fi
done
set -e

exit "${failed}"
