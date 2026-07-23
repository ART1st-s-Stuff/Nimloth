#!/usr/bin/env bash
# Drive vLLM rollout and FSDP PPO on an arbitrary heterogeneous Slurm allocation.
set -euo pipefail

HOLD_JOB=${HOLD_JOB:?set HOLD_JOB to one running allocation}
REPO=${REPO:?set REPO to the committed server worktree}
RL_CONFIG=${RL_CONFIG:-${REPO}/configs/training/rl/e2e_smoke_h4.yaml}
RUN_OUT=${RUN_OUT:?set RUN_OUT to a new output directory}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
RAY_PORT=${RAY_PORT:-6381}
RAY_LOG_DIR=${RUN_OUT}.ray
RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray_nimloth_${HOLD_JOB}_${BASHPID}}
RAY_OBJECT_STORE_BYTES=${RAY_OBJECT_STORE_BYTES:-10000000000}

read -r CONFIG_NODES CONFIG_WORLD_SIZE CONFIG_TP_SIZE < <(
  PYTHONPATH="${REPO}/src" "${PYTHON}" -c '
import sys
from pathlib import Path
from nimloth.config.rl import load_rl_config
config = load_rl_config(Path(sys.argv[1])).distributed
print(config.nodes, config.world_size, config.rollout_tensor_parallel_size)
' "${RL_CONFIG}"
)
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o %N)")
(( ${#NODES[@]} == CONFIG_NODES )) || {
  echo "config requests ${CONFIG_NODES} nodes, allocation has: ${NODES[*]}" >&2
  exit 1
}
HEAD_NODE=${NODES[0]}
JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")
declare -A GPU_COUNTS
total_gpus=0
for node in "${NODES[@]}"; do
  count=$(sed -n "s/.*Nodes=${node} .*GRES=gpu:\([0-9][0-9]*\).*/\1/p" <<< "${JOB_DETAILS}")
  [[ -n "${count}" ]] || { echo "missing allocated GPU count for ${node}" >&2; exit 1; }
  GPU_COUNTS[${node}]=${count}
  total_gpus=$((total_gpus + count))
done
(( total_gpus == CONFIG_WORLD_SIZE )) || {
  echo "allocation has ${total_gpus} GPUs, config requests ${CONFIG_WORLD_SIZE}" >&2
  exit 1
}
HEAD_IP=$(srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --gpus=0 hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${HEAD_IP}" ]] || { echo "could not resolve Ray head IP" >&2; exit 1; }
mkdir -p "${RAY_LOG_DIR}"

stop_ray() {
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes="${CONFIG_NODES}" \
    --ntasks="${CONFIG_NODES}" --ntasks-per-node=1 --gpus=0 \
    timeout 20s "${PYTHON}" -m ray.scripts.scripts stop --force \
    >/dev/null 2>&1 || true
}
trap stop_ray EXIT
stop_ray

declare -a RAY_STEP_PIDS=()
head_gpus=${GPU_COUNTS[${HEAD_NODE}]}
head_cpus=$((head_gpus > 4 ? head_gpus : 4))
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --gpus="${head_gpus}" \
  "${PYTHON}" -m ray.scripts.scripts start --head \
    --port="${RAY_PORT}" --node-ip-address="${HEAD_IP}" \
    --num-cpus="${head_cpus}" --num-gpus="${head_gpus}" \
    --object-store-memory="${RAY_OBJECT_STORE_BYTES}" \
    --temp-dir="${RAY_TMP_DIR}" --disable-usage-stats \
    --include-dashboard=false --block \
  >"${RAY_LOG_DIR}/${HEAD_NODE}.log" 2>&1 &
RAY_STEP_PIDS+=("$!")
head_ready=false
for _ in $(seq 1 90); do
  if grep -q "Ray runtime started" "${RAY_LOG_DIR}/${HEAD_NODE}.log"; then
    head_ready=true
    break
  fi
  kill -0 "${RAY_STEP_PIDS[0]}" 2>/dev/null || {
    tail -n 200 "${RAY_LOG_DIR}/${HEAD_NODE}.log" >&2
    exit 1
  }
  sleep 2
done
[[ "${head_ready}" == true ]] || { echo "Ray head port did not become ready" >&2; exit 1; }
for node in "${NODES[@]:1}"; do
  node_gpus=${GPU_COUNTS[${node}]}
  node_cpus=$((node_gpus > 4 ? node_gpus : 4))
  srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
    --gpus="${node_gpus}" \
    "${PYTHON}" -m ray.scripts.scripts start \
      --address="${HEAD_IP}:${RAY_PORT}" --num-cpus="${node_cpus}" \
      --num-gpus="${node_gpus}" --object-store-memory="${RAY_OBJECT_STORE_BYTES}" \
      --temp-dir="${RAY_TMP_DIR}" --block \
    >"${RAY_LOG_DIR}/${node}.log" 2>&1 &
  RAY_STEP_PIDS+=("$!")
done

resources=""
for _ in $(seq 1 90); do
  resources=$(timeout 20s srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    --gpus=0 env RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" "${PYTHON}" -c \
    'import ray; ray.init(address="auto"); print(int(ray.cluster_resources().get("GPU", 0)))' \
    2>/dev/null | tail -1 || true)
  [[ "${resources}" == "${CONFIG_WORLD_SIZE}" ]] && break
  for pid in "${RAY_STEP_PIDS[@]}"; do
    kill -0 "${pid}" 2>/dev/null || {
      tail -n 200 "${RAY_LOG_DIR}"/*.log >&2
      exit 1
    }
  done
  sleep 2
done
[[ "${resources}" == "${CONFIG_WORLD_SIZE}" ]] || {
  echo "Ray exposes ${resources:-0} GPUs, config requests world_size=${CONFIG_WORLD_SIZE}" >&2
  exit 1
}

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --gpus="${head_gpus}" \
  env RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
    REPO="${REPO}" RUN_OUT="${RUN_OUT}" RL_CONFIG="${RL_CONFIG}" \
    ENV_REPO="${ENV_REPO:?set ENV_REPO}" MODEL="${MODEL:?set MODEL}" \
    WM_CKPT="${WM_CKPT:-${MODEL}}" \
    WANDB_PROJECT="${WANDB_PROJECT:-nimloth-rl}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:?set WANDB_RUN_NAME}" \
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}" \
    VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray \
    bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_smoke.sh"

for pid in "${RAY_STEP_PIDS[@]}"; do
  wait "${pid}" 2>/dev/null || true
done
