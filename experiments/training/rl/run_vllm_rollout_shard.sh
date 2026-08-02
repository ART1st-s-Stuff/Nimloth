#!/usr/bin/env bash
# Collect one independent TP rollout shard on an explicitly isolated GPU group.
set -euo pipefail

REPO=${REPO:?set REPO to the committed server worktree}
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
MODEL=${MODEL:?set MODEL to the immutable rollout policy}
WM_CKPT=${WM_CKPT:-${MODEL}}
RL_CONFIG=${RL_CONFIG:?set RL_CONFIG}
SHARD_INDEX=${SHARD_INDEX:?set SHARD_INDEX}
SHARD_SEED=${SHARD_SEED:?set SHARD_SEED}
SHARD_EVAL_SET=${SHARD_EVAL_SET:?set SHARD_EVAL_SET}
SHARD_GPU_VISIBLE=${SHARD_GPU_VISIBLE:?set SHARD_GPU_VISIBLE}
SHARD_OUT=${SHARD_OUT:?set SHARD_OUT}
ENV_PORT=${ENV_PORT:?set ENV_PORT}
ENV_PREWARM_TIMEOUT=${ENV_PREWARM_TIMEOUT:-300}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-false}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-0}

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing RL config: ${RL_CONFIG}" >&2; exit 1; }
[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "invalid SHARD_INDEX" >&2; exit 1; }
[[ "${SHARD_SEED}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SHARD_SEED" >&2; exit 1; }
[[ "${SHARD_EVAL_SET}" == *_train ]] || {
  echo "rollout shard requires a training dataset" >&2
  exit 1
}
[[ "${ENV_PREWARM_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ENV_PREWARM_TIMEOUT must be positive" >&2
  exit 1
}
(( ENV_PREWARM_TIMEOUT <= 300 )) || {
  echo "ENV_PREWARM_TIMEOUT must not exceed 300 seconds" >&2
  exit 1
}
[[ "${VLLM_ENABLE_PREFIX_CACHING}" == false || "${VLLM_ENABLE_PREFIX_CACHING}" == true ]] || {
  echo "VLLM_ENABLE_PREFIX_CACHING must be true or false" >&2
  exit 1
}

read -r TP_SIZE MAX_STEPS TEMPERATURE TOP_P CREDIT MAX_RESPONSE_TOKENS ACTOR_ENABLED PLANNING_ENABLED PLANNING_HORIZON PLANNING_SEARCH_MODE PLANNING_BEAM_WIDTH PLANNER_DEVICE < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1]))
print(
    config.distributed.rollout_tensor_parallel_size,
    config.rl.max_steps_per_episode,
    config.rollout.temperature,
    config.rollout.top_p,
    config.actor.credit_assignment,
    config.actor.max_response_tokens,
    str(config.actor.enabled).lower(),
    str(config.agent.planning.enabled).lower(),
    config.agent.planning.horizon,
    config.agent.planning.search_mode,
    config.agent.planning.beam_width,
    config.agent.planning.device,
)
' "${RL_CONFIG}"
)
IFS=',' read -r -a SHARD_GPUS <<< "${SHARD_GPU_VISIBLE}"
(( ${#SHARD_GPUS[@]} == TP_SIZE )) || {
  echo "shard ${SHARD_INDEX} has ${#SHARD_GPUS[@]} GPUs, expected TP=${TP_SIZE}" >&2
  exit 1
}
[[ "${PLANNING_ENABLED}" == true ]] || {
  echo "parallel rollout runner requires planner rollout" >&2
  exit 1
}
[[ "${ACTOR_ENABLED}" == false && "${CREDIT}" == action ]] || {
  echo "parallel planner rollout requires direct actor off and action credit" >&2
  exit 1
}
for path in "${WM_CKPT}/state_proj.pt" "${WM_CKPT}/wm_predictor/predictor.pt" "${WM_CKPT}/value_head/value_head.pt"; do
  [[ -s "${path}" ]] || { echo "missing planner checkpoint: ${path}" >&2; exit 1; }
done
[[ -f "${ENV_REPO}/external/VAGEN/vagen/env/navigation/datasets/${SHARD_EVAL_SET}.json" ]] || {
  echo "missing rollout dataset: ${SHARD_EVAL_SET}" >&2
  exit 1
}
if [[ -e "${SHARD_OUT}" ]] && find "${SHARD_OUT}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to reuse non-empty rollout shard: ${SHARD_OUT}" >&2
  exit 1
fi
mkdir -p "${SHARD_OUT}"

export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-/project/peilab/atst/nimloth/.cache/flashinfer}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# Planner construction initializes CUDA before vLLM creates its TP workers.
# The vLLM default is fork, which cannot reinitialize CUDA in those children.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "${FLASHINFER_WORKSPACE_DIR}"

export CUDA_VISIBLE_DEVICES=${SHARD_GPU_VISIBLE}
HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${HEAD_IP}" ]] || HEAD_IP=$(hostname -I | awk '{print $1}')
ENV_URL=http://${HEAD_IP}:${ENV_PORT}
ENV_LOG=${SHARD_OUT}/env_server.log
ROLLOUT_LOG=${SHARD_OUT}/rollout.log
ENV_PID=""

cleanup_env() {
  if [[ -n "${ENV_PID}" ]]; then
    kill "${ENV_PID}" 2>/dev/null || true
    wait "${ENV_PID}" 2>/dev/null || true
    ENV_PID=""
  fi
}
trap cleanup_env EXIT

(
  # Keep the renderer inside this shard's isolated TP4 group, but avoid the
  # leading device: dgx-32 repeatedly hung AI2-THOR initialization there while
  # the second device in the same group remained healthy.
  export CUDA_VISIBLE_DEVICES=${SHARD_GPUS[1]}
  export PYTHONPATH=${ENV_REPO}/external/VAGEN
  source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
  cd "${ENV_REPO}/external/VAGEN"
  exec "${PYTHON}" -m vagen.server.server \
    server.host=0.0.0.0 server.port=${ENV_PORT} use_state_reward=False \
    navigation.devices=[0] navigation.max_workers=1
) >"${ENV_LOG}" 2>&1 &
ENV_PID=$!

env_ready=false
for _ in $(seq 1 300); do
  if curl -fsS "${ENV_URL}/health" >/dev/null 2>&1; then
    env_ready=true
    break
  fi
  if ! kill -0 "${ENV_PID}" 2>/dev/null; then
    tail -100 "${ENV_LOG}" >&2
    exit 1
  fi
  sleep 1
done
[[ "${env_ready}" == true ]] || { echo "shard env did not become ready" >&2; exit 1; }
curl -fsS "${ENV_URL}/health"

PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN timeout \
  --signal=TERM --kill-after=10s "${ENV_PREWARM_TIMEOUT}s" \
  "${PYTHON}" -m nimloth.environment.navigation.prewarm \
    --env-url "${ENV_URL}" \
    --eval-set "${SHARD_EVAL_SET}" \
    --seed "${SHARD_SEED}" \
    --timeout-seconds "${ENV_PREWARM_TIMEOUT}" \
    --env-id "nimloth-navigation-shard-${SHARD_INDEX}-${SHARD_SEED}"

export CUDA_VISIBLE_DEVICES=${SHARD_GPU_VISIBLE}
export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
VLLM_ARGS=(
  --vllm-distributed-executor-backend mp
  --vllm-mm-processor-cache-gb "${VLLM_MM_PROCESSOR_CACHE_GB}"
)
if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == true ]]; then
  VLLM_ARGS+=(--vllm-enable-prefix-caching)
fi
PLANNER_ARGS=(
  --planner-enabled
  --planning-horizon "${PLANNING_HORIZON}"
  --planning-search-mode "${PLANNING_SEARCH_MODE}"
  --planner-device "${PLANNER_DEVICE}"
  --wm-checkpoint "${WM_CKPT}/wm_predictor"
  --state-proj-checkpoint "${WM_CKPT}/state_proj.pt"
  --value-head-checkpoint "${WM_CKPT}/value_head"
)
if [[ "${PLANNING_SEARCH_MODE}" == beam ]]; then
  [[ "${PLANNING_BEAM_WIDTH}" != None ]] || {
    echo "beam planner requires a beam width" >&2
    exit 1
  }
  PLANNER_ARGS+=(--planning-beam-width "${PLANNING_BEAM_WIDTH}")
fi

"${PYTHON}" "${REPO}/experiments/training/rl/rollout_env.py" \
  --backend vllm \
  --tensor-parallel-size "${TP_SIZE}" \
  "${VLLM_ARGS[@]}" \
  "${PLANNER_ARGS[@]}" \
  --model "${MODEL}" \
  --env-url "${ENV_URL}" \
  --output-dir "${SHARD_OUT}" \
  --fresh-manifest "${SHARD_OUT}/fresh_policy_manifest.json" \
  --num-episodes 1 \
  --max-steps "${MAX_STEPS}" \
  --eval-set "${SHARD_EVAL_SET}" --split train --seed-offset "${SHARD_SEED}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" \
  --credit-assignment "${CREDIT}" \
  --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
  --vllm-enforce-eager \
  2>&1 | tee "${ROLLOUT_LOG}"

[[ -s "${SHARD_OUT}/fresh_policy_manifest.json" ]] || {
  echo "rollout shard manifest is missing" >&2
  exit 1
}
[[ -s "${SHARD_OUT}/rollout_summary.json" ]] || {
  echo "rollout shard summary is missing" >&2
  exit 1
}
cleanup_env
echo "ROLLOUT_SHARD_OK index=${SHARD_INDEX} seed=${SHARD_SEED} dataset=${SHARD_EVAL_SET} gpus=${SHARD_GPU_VISIBLE}"
