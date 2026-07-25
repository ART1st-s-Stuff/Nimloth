#!/usr/bin/env bash
# Run one fresh-policy vLLM rollout followed by one config-sized PPO update.
set -euo pipefail

REPO=${REPO:?set REPO to the committed server worktree}
source "${REPO}/experiments/training/rl/slurm_allocation.sh"
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
MODEL=${MODEL:?set MODEL to a complete positive-k inject HF checkpoint}
REFERENCE_MODEL=${REFERENCE_MODEL:-${MODEL}}
WM_CKPT=${WM_CKPT:-${MODEL}}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/planner_exhaustive_h2_smoke.yaml}
RUN_OUT=${RUN_OUT:?set a new exclusive output directory}
WANDB_PROJECT_REQUESTED=${WANDB_PROJECT:-nimloth-rl}
WANDB_RUN_NAME_REQUESTED=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
WANDB_MODE_REQUESTED=${WANDB_MODE_OVERRIDE:-online}
ENV_PORT=${ENV_PORT:-8500}
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29671}
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-false}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-0}
PIPELINE_PHASE=${PIPELINE_PHASE:-all}

case "${PIPELINE_PHASE}" in
  all) RUN_ROLLOUT=true; RUN_REFERENCE=true; RUN_TRAIN=true ;;
  rollout) RUN_ROLLOUT=true; RUN_REFERENCE=false; RUN_TRAIN=false ;;
  reference) RUN_ROLLOUT=false; RUN_REFERENCE=true; RUN_TRAIN=false ;;
  train) RUN_ROLLOUT=false; RUN_REFERENCE=false; RUN_TRAIN=true ;;
  *) echo "PIPELINE_PHASE must be all, rollout, reference, or train" >&2; exit 1 ;;
esac

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing RL config: ${RL_CONFIG}" >&2; exit 1; }
read -r CONFIG_NODES CONFIG_WORLD_SIZE CONFIG_GPUS_PER_RANK CONFIG_TOTAL_GPUS CONFIG_TP_SIZE CREDIT_ASSIGNMENT MAX_RESPONSE_TOKENS REFERENCE_KL_WEIGHT CONFIG_NUM_EPISODES CONFIG_MAX_STEPS ROLLOUT_TEMPERATURE ROLLOUT_TOP_P PLANNING_ENABLED PLANNING_HORIZON PLANNING_SEARCH_MODE PLANNING_BEAM_WIDTH PLANNER_DEVICE < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1]))
print(
    config.distributed.nodes,
    config.distributed.world_size,
    config.distributed.gpus_per_rank,
    config.distributed.total_gpus,
    config.distributed.rollout_tensor_parallel_size,
    config.actor.credit_assignment,
    config.actor.max_response_tokens,
    config.actor.reference_kl_loss_weight,
    config.rl.envs_per_iteration,
    config.rl.max_steps_per_episode,
    config.rollout.temperature,
    config.rollout.top_p,
    str(config.agent.planning.enabled).lower(),
    config.agent.planning.horizon,
    config.agent.planning.search_mode,
    config.agent.planning.beam_width,
    config.agent.planning.device,
)
' "${RL_CONFIG}"
)
NUM_EPISODES=${NUM_EPISODES:-${CONFIG_NUM_EPISODES}}
MAX_STEPS=${MAX_STEPS:-${CONFIG_MAX_STEPS}}
[[ "${NUM_EPISODES}" == "${CONFIG_NUM_EPISODES}" ]] || {
  echo "NUM_EPISODES disagrees with rl.envs_per_iteration" >&2
  exit 1
}
[[ "${MAX_STEPS}" == "${CONFIG_MAX_STEPS}" ]] || {
  echo "MAX_STEPS disagrees with rl.max_steps_per_episode" >&2
  exit 1
}
[[ "${VLLM_ENABLE_PREFIX_CACHING}" == false || "${VLLM_ENABLE_PREFIX_CACHING}" == true ]] || {
  echo "VLLM_ENABLE_PREFIX_CACHING must be true or false" >&2
  exit 1
}
if [[ "${REFERENCE_KL_WEIGHT}" != 0.0 ]]; then
  [[ -f "${REFERENCE_MODEL}/config.json" ]] || {
    echo "missing reference model: ${REFERENCE_MODEL}" >&2
    exit 1
  }
fi
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${CONFIG_TP_SIZE}}
TRAIN_NNODES=${TRAIN_NNODES:-${CONFIG_NODES}}
TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE:-${CONFIG_WORLD_SIZE}}
TRAIN_GPUS_PER_RANK=${TRAIN_GPUS_PER_RANK:-${CONFIG_GPUS_PER_RANK}}
TRAIN_TOTAL_GPUS=${TRAIN_TOTAL_GPUS:-${CONFIG_TOTAL_GPUS}}
[[ "${TENSOR_PARALLEL_SIZE}" == "${CONFIG_TP_SIZE}" ]] || {
  echo "TENSOR_PARALLEL_SIZE disagrees with distributed.rollout_tensor_parallel_size" >&2
  exit 1
}
[[ "${TRAIN_NNODES}" == "${CONFIG_NODES}" ]] || {
  echo "TRAIN_NNODES disagrees with distributed.nodes" >&2
  exit 1
}
[[ "${TRAIN_WORLD_SIZE}" == "${CONFIG_WORLD_SIZE}" ]] || {
  echo "TRAIN_WORLD_SIZE disagrees with distributed.world_size" >&2
  exit 1
}
[[ "${TRAIN_GPUS_PER_RANK}" == "${CONFIG_GPUS_PER_RANK}" ]] || {
  echo "TRAIN_GPUS_PER_RANK disagrees with distributed.gpus_per_rank" >&2
  exit 1
}
[[ "${TRAIN_TOTAL_GPUS}" == "${CONFIG_TOTAL_GPUS}" ]] || {
  echo "TRAIN_TOTAL_GPUS disagrees with distributed.total_gpus" >&2
  exit 1
}
for path in "${WM_CKPT}/state_proj.pt" "${WM_CKPT}/wm_predictor/predictor.pt" "${WM_CKPT}/value_head/value_head.pt"; do
  [[ -f "${path}" ]] || { echo "missing checkpoint file: ${path}" >&2; exit 1; }
done
[[ -f "${ENV_REPO}/external/VAGEN/vagen/env/navigation/datasets/base_train.json" ]] || {
  echo "ENV_REPO does not contain base_train" >&2
  exit 1
}
if [[ "${RUN_ROLLOUT}" == true && -e "${RUN_OUT}" ]] && \
    find "${RUN_OUT}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to reuse non-empty output directory: ${RUN_OUT}" >&2
  exit 1
fi

ROLLOUT_OUT=${RUN_OUT}/rollout
TRAIN_OUT=${RUN_OUT}/train
MANIFEST=${ROLLOUT_OUT}/fresh_policy_manifest.json
LOG=${RUN_OUT}/pipeline.log
mkdir -p "${RUN_OUT}" "${ROLLOUT_OUT}" "${TRAIN_OUT}"

COMMIT=$(git -C "${REPO}" rev-parse HEAD)
ENV_COMMIT=$(git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD)
if [[ "${RUN_ROLLOUT}" == true ]]; then
  cat > "${RUN_OUT}/README.md" <<EOF
# vLLM online PPO smoke (${TRAIN_TOTAL_GPUS} GPUs)

- status: running
- Nimloth commit: ${COMMIT}
- VAGEN commit: ${ENV_COMMIT}
- model/WM initialization: ${MODEL}
- data: base_train seeds 1..${NUM_EPISODES}
- rollout: vLLM TP=${TENSOR_PARALLEL_SIZE}, backend=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-local}, ${NUM_EPISODES} episodes, max ${MAX_STEPS} steps
- config: ${RL_CONFIG}
- planning: enabled=${PLANNING_ENABLED}, horizon=${PLANNING_HORIZON}, search=${PLANNING_SEARCH_MODE}, beam_width=${PLANNING_BEAM_WIDTH}
- response credit: ${CREDIT_ASSIGNMENT}, max full response tokens=${MAX_RESPONSE_TOKENS}
- reference KL actor loss: weight=${REFERENCE_KL_WEIGHT}; no reward KL
- reference model: ${REFERENCE_MODEL}
- freshness: policy/planner/trajectory content fingerprints; consumption commits only after a post-update checkpoint
- update: ${TRAIN_NNODES} nodes, ${TRAIN_WORLD_SIZE} ranks × ${TRAIN_GPUS_PER_RANK} GPUs/rank, one grid-WM/value/SIGReg/PPO optimizer step; no DINO loss
- frozen: vision tower, GridStateProjector, EMA target encoder and DINO decoder
- trainable: Qwen language body, WM predictor and ValueHead
- W&B: ${WANDB_PROJECT_REQUESTED}/${WANDB_RUN_NAME_REQUESTED}
- output: ${RUN_OUT}
EOF
else
  [[ -s "${MANIFEST}" ]] || { echo "missing rollout manifest: ${MANIFEST}" >&2; exit 1; }
  [[ -s "${ROLLOUT_OUT}/trajectories.jsonl" ]] || {
    echo "missing rollout trajectories: ${ROLLOUT_OUT}/trajectories.jsonl" >&2
    exit 1
  }
fi

export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-/project/peilab/atst/nimloth/.cache/flashinfer}
mkdir -p "${FLASHINFER_WORKSPACE_DIR}"
# The cluster image cannot JIT FlashInfer's sampler extension reliably. vLLM's
# native sampler preserves the same requested temperature/top-p distribution.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Ray gives each GPU actor its allocated device as local cuda:0.  PyTorch's
# symmetric-memory rendezvous compares those local ordinals across ranks and
# can therefore report false overlap on multi-node tensor parallel runs.  Use
# the regular NCCL/custom-all-reduce path, which preserves the same collectives.
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
fi
export WANDB_PROJECT=${WANDB_PROJECT_REQUESTED}
export WANDB_RUN_NAME=${WANDB_RUN_NAME_REQUESTED}
export WANDB_MODE=${WANDB_MODE_REQUESTED}
export WANDB_DIR=${WANDB_DIR:-${REPO}/.cache/wandb}

VISIBLE=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TRAIN_TOTAL_GPUS - 1)))}
IFS=',' read -r -a GPUS <<< "${VISIBLE}"
if [[ "${RUN_ROLLOUT}" == true && -z "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}" ]] \
    && (( ${#GPUS[@]} != TENSOR_PARALLEL_SIZE )); then
  echo "expected ${TENSOR_PARALLEL_SIZE} visible GPUs, got ${VISIBLE}" >&2
  exit 1
fi
HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${HEAD_IP}" ]] || HEAD_IP=$(hostname -I | awk '{print $1}')
ENV_URL=http://${HEAD_IP}:${ENV_PORT}
ENV_LOG=${RUN_OUT}/env_server.log
ENV_PID=""

cleanup_env() {
  if [[ -n "${ENV_PID}" ]]; then
    kill "${ENV_PID}" 2>/dev/null || true
    wait "${ENV_PID}" 2>/dev/null || true
    ENV_PID=""
  fi
}
trap cleanup_env EXIT

if [[ "${RUN_ROLLOUT}" == true ]]; then
  {
    echo "=== ${TRAIN_TOTAL_GPUS}-GPU vLLM online PPO start $(date -Iseconds) ==="
    echo "node=$(hostname) gpus=${VISIBLE} env_url=${ENV_URL}"
    echo "nimloth=${COMMIT} vagen=${ENV_COMMIT}"
  } | tee "${LOG}"
fi

if [[ "${RUN_ROLLOUT}" == true ]]; then
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[0]}
    export PYTHONPATH=${ENV_REPO}/external/VAGEN
    source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
    cd "${ENV_REPO}/external/VAGEN"
    exec "${PYTHON}" -m vagen.server.server \
      server.host=0.0.0.0 server.port=${ENV_PORT} use_state_reward=False \
      navigation.devices=[0] navigation.max_workers=1
  ) >"${ENV_LOG}" 2>&1 &
  ENV_PID=$!
  for i in $(seq 1 300); do
    if curl -fsS "${ENV_URL}/health" >/dev/null 2>&1; then
      echo "env ready after ${i}s" | tee -a "${LOG}"
      break
    fi
    if ! kill -0 "${ENV_PID}" 2>/dev/null; then
      tail -100 "${ENV_LOG}" | tee -a "${LOG}"
      exit 1
    fi
    sleep 1
  done
  curl -fsS "${ENV_URL}/health" | tee -a "${LOG}"

  export CUDA_VISIBLE_DEVICES=${VISIBLE}
  export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
  VLLM_BACKEND_ARGS=()
  if [[ -n "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}" ]]; then
    VLLM_BACKEND_ARGS=(
      --vllm-distributed-executor-backend "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}"
    )
  fi
  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == true ]]; then
    VLLM_BACKEND_ARGS+=(--vllm-enable-prefix-caching)
  fi
  VLLM_BACKEND_ARGS+=(
    --vllm-mm-processor-cache-gb "${VLLM_MM_PROCESSOR_CACHE_GB}"
  )
  PLANNER_ARGS=()
  if [[ "${PLANNING_ENABLED}" == true ]]; then
    [[ "${CREDIT_ASSIGNMENT}" == token ]] || {
      echo "planner rollout requires token credit" >&2
      exit 1
    }
    [[ "${PLANNING_SEARCH_MODE}" != None ]] || {
      echo "missing agent.planning.search_mode" >&2
      exit 1
    }
    [[ "${PLANNER_DEVICE}" != None ]] || {
      echo "missing agent.planning.device" >&2
      exit 1
    }
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
        echo "beam planner requires agent.planning.beam_width" >&2
        exit 1
      }
      PLANNER_ARGS+=(--planning-beam-width "${PLANNING_BEAM_WIDTH}")
    elif [[ "${PLANNING_BEAM_WIDTH}" != None ]]; then
      echo "agent.planning.beam_width is only valid for beam search" >&2
      exit 1
    fi
  fi
  "${PYTHON}" "${REPO}/experiments/training/rl/rollout_env.py" \
    --backend vllm \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    "${VLLM_BACKEND_ARGS[@]}" \
    "${PLANNER_ARGS[@]}" \
    --model "${MODEL}" \
    --env-url "${ENV_URL}" \
    --output-dir "${ROLLOUT_OUT}" \
    --fresh-manifest "${MANIFEST}" \
    --num-episodes "${NUM_EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --eval-set base_train --split train --seed-offset 1 \
    --temperature "${ROLLOUT_TEMPERATURE}" --top-p "${ROLLOUT_TOP_P}" --max-pixels 3136 \
    --credit-assignment "${CREDIT_ASSIGNMENT}" \
    --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
    --vllm-enforce-eager \
    2>&1 | tee -a "${LOG}"
  cleanup_env
fi
if [[ "${RUN_REFERENCE}" == true && "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}" == ray ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Ray cleanup requires SLURM_JOB_ID" >&2; exit 1; }
  srun --jobid="${SLURM_JOB_ID}" --overlap --nodes="${TRAIN_NNODES}" \
    --ntasks="${TRAIN_NNODES}" --ntasks-per-node=1 --gpus=0 \
    timeout 20s "${PYTHON}" -m ray.scripts.scripts stop --force \
    2>&1 | tee -a "${LOG}"
fi

if [[ "${RUN_REFERENCE}" == true ]]; then
  [[ "${REFERENCE_KL_WEIGHT}" != 0.0 ]] || {
    echo "reference phase requires positive actor.reference_kl_loss_weight" >&2
    exit 1
  }
  export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
  "${PYTHON}" -m nimloth.training.rl.reference_replay \
    --manifest "${MANIFEST}" \
    --reference-model "${REFERENCE_MODEL}" \
    --output-dir "${RUN_OUT}/reference" \
    --model-parallel-size "${TRAIN_GPUS_PER_RANK}" \
    --attn-implementation sdpa --max-pixels 3136 \
    2>&1 | tee -a "${LOG}"
fi

if [[ "${RUN_TRAIN}" == true ]]; then
  export PYTHONPATH=${REPO}/src:${ENV_REPO}/external/VAGEN:${ENV_REPO}/external/VAGEN/verl:${REPO}/external/le-wm
  export NIMLOTH_DDP_GPU_STRIDE=${TRAIN_GPUS_PER_RANK}
TRAIN_ARGS=(
  -m nimloth.training.rl.cli
  --config "${RL_CONFIG}" \
  --model "${MODEL}" \
  --llm-tune full --vision-tune freeze --no-vision-ema \
  --gradient-checkpointing \
  --wm-checkpoint "${WM_CKPT}/wm_predictor" \
  --state-proj-checkpoint "${WM_CKPT}/state_proj.pt" \
  --value-head-checkpoint "${WM_CKPT}/value_head" \
  --use-jsonl-rollout \
  --fresh-rollout-manifest "${MANIFEST}" \
  --attn-implementation sdpa --max-pixels 3136 \
  --experiment-name "${WANDB_RUN_NAME_REQUESTED}" \
  --output-dir "${TRAIN_OUT}"
)
if [[ "${REFERENCE_KL_WEIGHT}" != 0.0 ]]; then
  TRAIN_ARGS+=(--reference-model "${REFERENCE_MODEL}")
fi
if (( TRAIN_NNODES == 1 )); then
  "${PYTHON}" -m torch.distributed.run --nproc_per_node="${TRAIN_WORLD_SIZE}" -- \
    "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG}"
else
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "multi-node training requires SLURM_JOB_ID" >&2; exit 1; }
  mapfile -t TRAIN_NODES_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  HEAD_NODE=${TRAIN_NODES_LIST[0]}
  RDZV_IP=$(srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    --gpus=0 hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
  [[ -n "${RDZV_IP}" ]] || { echo "missing multi-node rendezvous IP" >&2; exit 1; }
  export NIMLOTH_TRAIN_ARGS=$(printf '%q ' "${TRAIN_ARGS[@]}")
  JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}")
  declare -A TRAIN_GPU_COUNTS
  nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" TRAIN_GPU_COUNTS
  TRAIN_STEP_PIDS=()
  rank_offset=0
  for node in "${TRAIN_NODES_LIST[@]}"; do
    node_gpus=${TRAIN_GPU_COUNTS[${node}]:-}
    [[ -n "${node_gpus}" ]] || { echo "missing allocated GPU count for ${node}" >&2; exit 1; }
    (( node_gpus % TRAIN_GPUS_PER_RANK == 0 )) || {
      echo "node ${node} has ${node_gpus} GPUs, not divisible by gpus_per_rank=${TRAIN_GPUS_PER_RANK}" >&2
      exit 1
    }
    node_ranks=$((node_gpus / TRAIN_GPUS_PER_RANK))
    srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
      --gres="gpu:${node_gpus}" \
      env NIMLOTH_NODE_RANKS="${node_ranks}" NIMLOTH_RANK_OFFSET="${rank_offset}" \
        NIMLOTH_DDP_GPU_STRIDE="${TRAIN_GPUS_PER_RANK}" \
      bash -lc '
      set -euo pipefail
      export PYTHONPATH="'"${PYTHONPATH}"'"
      export MASTER_ADDR="'"${RDZV_IP}"'"
      export MASTER_PORT="'"${TRAIN_MASTER_PORT}"'"
      pids=()
      for ((local_rank=0; local_rank<NIMLOTH_NODE_RANKS; local_rank++)); do
        export RANK=$((NIMLOTH_RANK_OFFSET + local_rank))
        export WORLD_SIZE="'"${TRAIN_WORLD_SIZE}"'"
        export LOCAL_RANK="${local_rank}"
        "'"${PYTHON}"'" ${NIMLOTH_TRAIN_ARGS} &
        pids+=("$!")
      done
      status=0
      for pid in "${pids[@]}"; do
        wait "${pid}" || status=$?
      done
      exit "${status}"
    ' 2>&1 | tee -a "${LOG}" &
    TRAIN_STEP_PIDS+=("$!")
    rank_offset=$((rank_offset + node_ranks))
  done
  (( rank_offset == TRAIN_WORLD_SIZE )) || {
    echo "node GPU counts sum to ${rank_offset}, expected ${TRAIN_WORLD_SIZE}" >&2
    exit 1
  }
  train_status=0
  for pid in "${TRAIN_STEP_PIDS[@]}"; do
    wait "${pid}" || train_status=$?
  done
  if (( train_status != 0 )); then
    exit "${train_status}"
  fi
fi

"${PYTHON}" - <<PY | tee -a "${LOG}"
import csv, json, math
from pathlib import Path
root = Path("${TRAIN_OUT}")
rows = list(csv.DictReader((root / "train_step_log.csv").open()))
if len(rows) != 1 or int(rows[0]["global_step"]) != 1:
    raise SystemExit(f"expected one optimizer step: {rows}")
keys = (
    "wm_mse",
    "value_loss",
    "sigreg_loss",
    "actor_loss",
    "entropy",
    "clip_fraction",
    "mean_advantage",
    "token_value_loss",
    "action_distillation_loss",
    "action_distillation_kl",
    "reference_kl_loss",
    "mean_ratio",
    "policy_tokens",
    "total_loss",
)
bad = {key: rows[0].get(key) for key in keys if not math.isfinite(float(rows[0][key]))}
if bad:
    raise SystemExit(f"non-finite metrics: {bad}")
final = root / "final"
required = [
    final / "rl_state.pt",
    final / "model.safetensors.index.json",
    final / "state_proj.pt",
    final / "wm_predictor" / "config.json",
    final / "wm_predictor" / "predictor.pt",
    final / "value_head" / "value_head.pt",
    final / "token_value_head" / "config.json",
    final / "token_value_head" / "token_value_head.pt",
]
if ${TRAIN_GPUS_PER_RANK} == 1 and ${TRAIN_WORLD_SIZE} > 1:
    required += [
        final / f"optimizer_rank_{rank:05d}.pt"
        for rank in range(${TRAIN_WORLD_SIZE})
    ]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
state = __import__("torch").load(final / "rl_state.pt", map_location="cpu", weights_only=False)
expected_layout = "rank_sharded_fsdp" if ${TRAIN_GPUS_PER_RANK} == 1 and ${TRAIN_WORLD_SIZE} > 1 else "replicated"
if state.get("optimizer_state_layout") != expected_layout:
    raise SystemExit(f"optimizer layout mismatch: {state.get('optimizer_state_layout')}")
if expected_layout == "replicated" and state.get("optimizer") is None:
    raise SystemExit("replicated optimizer state is missing")
consumption = json.loads(Path("${MANIFEST}.consumption.json").read_text())
if consumption.get("state") != "committed" or consumption.get("committed_global_step") != 1:
    raise SystemExit(f"fresh consumption was not committed: {consumption}")
print(json.dumps({"status": "ALL_OK", "global_step": 1, "finite_metrics": keys}))
PY

sed -i 's/- status: running/- status: completed/' "${RUN_OUT}/README.md"
echo "=== ${TRAIN_TOTAL_GPUS}-GPU vLLM online PPO ALL_OK $(date -Iseconds) ===" | tee -a "${LOG}"
fi
