#!/usr/bin/env bash
# Run one fresh-policy vLLM rollout followed by one config-sized FSDP PPO update.
set -euo pipefail

REPO=${REPO:?set REPO to the committed server worktree}
ENV_REPO=${ENV_REPO:?set ENV_REPO to the verified VAGEN worktree}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
MODEL=${MODEL:?set MODEL to a complete k=1 inject HF checkpoint}
WM_CKPT=${WM_CKPT:-${MODEL}}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/e2e_smoke_h4.yaml}
RUN_OUT=${RUN_OUT:?set a new exclusive output directory}
WANDB_PROJECT_REQUESTED=${WANDB_PROJECT:-nimloth-rl}
WANDB_RUN_NAME_REQUESTED=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
WANDB_MODE_REQUESTED=${WANDB_MODE_OVERRIDE:-online}
ENV_PORT=${ENV_PORT:-8500}
NUM_EPISODES=${NUM_EPISODES:-4}
MAX_STEPS=${MAX_STEPS:-5}
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-}

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 1; }
[[ -f "${RL_CONFIG}" ]] || { echo "missing RL config: ${RL_CONFIG}" >&2; exit 1; }
read -r CONFIG_NODES CONFIG_WORLD_SIZE CONFIG_TP_SIZE < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1])).distributed
print(config.nodes, config.world_size, config.rollout_tensor_parallel_size)
' "${RL_CONFIG}"
)
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${CONFIG_TP_SIZE}}
TRAIN_NNODES=${TRAIN_NNODES:-${CONFIG_NODES}}
TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE:-${CONFIG_WORLD_SIZE}}
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
for path in "${WM_CKPT}/state_proj.pt" "${WM_CKPT}/wm_predictor/predictor.pt" "${WM_CKPT}/value_head/value_head.pt"; do
  [[ -f "${path}" ]] || { echo "missing checkpoint file: ${path}" >&2; exit 1; }
done
[[ -f "${ENV_REPO}/external/VAGEN/vagen/env/navigation/datasets/base_train.json" ]] || {
  echo "ENV_REPO does not contain base_train" >&2
  exit 1
}
if [[ -e "${RUN_OUT}" ]] && find "${RUN_OUT}" -mindepth 1 -print -quit | grep -q .; then
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
cat > "${RUN_OUT}/README.md" <<EOF
# vLLM online PPO smoke (${TRAIN_WORLD_SIZE} GPUs)

- status: running
- Nimloth commit: ${COMMIT}
- VAGEN commit: ${ENV_COMMIT}
- model/WM initialization: ${MODEL}
- data: base_train seeds 1..${NUM_EPISODES}
- rollout: vLLM TP=${TENSOR_PARALLEL_SIZE}, backend=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-local}, ${NUM_EPISODES} episodes, max ${MAX_STEPS} steps
- freshness: content fingerprint manifest, exactly one PPO consumption
- update: ${TRAIN_NNODES} nodes, ${TRAIN_WORLD_SIZE} total ranks, one WM/value/SIGReg/PPO optimizer step
- frozen: vision tower and StateProjector
- trainable: Qwen language body, WM predictor and ValueHead
- W&B: ${WANDB_PROJECT_REQUESTED}/${WANDB_RUN_NAME_REQUESTED}
- output: ${RUN_OUT}
EOF

export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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

VISIBLE=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TRAIN_WORLD_SIZE - 1)))}
IFS=',' read -r -a GPUS <<< "${VISIBLE}"
if [[ -z "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}" ]] && (( ${#GPUS[@]} != TENSOR_PARALLEL_SIZE )); then
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

{
  echo "=== ${TRAIN_WORLD_SIZE}-GPU vLLM online PPO start $(date -Iseconds) ==="
  echo "node=$(hostname) gpus=${VISIBLE} env_url=${ENV_URL}"
  echo "nimloth=${COMMIT} vagen=${ENV_COMMIT}"
} | tee "${LOG}"

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
"${PYTHON}" "${REPO}/experiments/training/rl/rollout_env.py" \
  --backend vllm \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  "${VLLM_BACKEND_ARGS[@]}" \
  --model "${MODEL}" \
  --env-url "${ENV_URL}" \
  --output-dir "${ROLLOUT_OUT}" \
  --fresh-manifest "${MANIFEST}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --eval-set base_train --split train --seed-offset 1 \
  --temperature 0.7 --top-p 0.95 --max-pixels 3136 \
  2>&1 | tee -a "${LOG}"

cleanup_env
if [[ "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}" == ray ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Ray cleanup requires SLURM_JOB_ID" >&2; exit 1; }
  srun --jobid="${SLURM_JOB_ID}" --overlap --nodes="${TRAIN_NNODES}" \
    --ntasks="${TRAIN_NNODES}" --ntasks-per-node=1 --gpus=0 \
    timeout 20s "${PYTHON}" -m ray.scripts.scripts stop --force \
    2>&1 | tee -a "${LOG}"
fi

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
  srun --jobid="${SLURM_JOB_ID}" --overlap --nodes="${TRAIN_NNODES}" \
    --ntasks="${TRAIN_WORLD_SIZE}" --gpus-per-task=1 --gpu-bind=single:1 \
    bash -lc '
      set -euo pipefail
      export PYTHONPATH="'"${PYTHONPATH}"'"
      export RANK="${SLURM_PROCID}"
      export WORLD_SIZE="${SLURM_NTASKS}"
      export LOCAL_RANK=0
      export MASTER_ADDR="'"${RDZV_IP}"'"
      export MASTER_PORT=29671
      "'"${PYTHON}"'" ${NIMLOTH_TRAIN_ARGS}
    ' 2>&1 | tee -a "${LOG}"
fi

"${PYTHON}" - <<PY | tee -a "${LOG}"
import csv, json, math
from pathlib import Path
root = Path("${TRAIN_OUT}")
rows = list(csv.DictReader((root / "train_step_log.csv").open()))
if len(rows) != 1 or int(rows[0]["global_step"]) != 1:
    raise SystemExit(f"expected one optimizer step: {rows}")
keys = ("wm_mse", "value_loss", "sigreg_loss", "actor_loss", "entropy", "total_loss")
bad = {key: rows[0].get(key) for key in keys if not math.isfinite(float(rows[0][key]))}
if bad:
    raise SystemExit(f"non-finite metrics: {bad}")
final = root / "final"
required = [final / "rl_state.pt", final / "model.safetensors.index.json"] + [
    final / f"optimizer_rank_{rank:05d}.pt" for rank in range(${TRAIN_WORLD_SIZE})
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
print(json.dumps({"status": "ALL_OK", "global_step": 1, "finite_metrics": keys}))
PY

sed -i 's/- status: running/- status: completed/' "${RUN_OUT}/README.md"
echo "=== ${TRAIN_WORLD_SIZE}-GPU vLLM online PPO ALL_OK $(date -Iseconds) ===" | tee -a "${LOG}"
