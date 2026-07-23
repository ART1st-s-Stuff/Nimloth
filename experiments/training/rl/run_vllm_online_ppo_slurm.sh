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
HEAD_IP=$(srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${HEAD_IP}" ]] || { echo "could not resolve Ray head IP" >&2; exit 1; }
mkdir -p "${RAY_LOG_DIR}"

stop_ray() {
  srun --jobid="${HOLD_JOB}" --overlap --nodes="${CONFIG_NODES}" \
    --ntasks="${CONFIG_NODES}" --ntasks-per-node=1 \
    "${PYTHON}" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
}
trap stop_ray EXIT
stop_ray

export NIMLOTH_RAY_HEAD_IP=${HEAD_IP}
export NIMLOTH_RAY_PORT=${RAY_PORT}
export NIMLOTH_PYTHON=${PYTHON}
srun --jobid="${HOLD_JOB}" --overlap --nodes="${CONFIG_NODES}" \
  --ntasks="${CONFIG_NODES}" --ntasks-per-node=1 \
  bash -lc '
    set -euo pipefail
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
      echo "node $(hostname) has no visible allocated GPUs" >&2
      exit 1
    }
    IFS=, read -r -a local_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    count=${#local_gpus[@]}
    if (( SLURM_PROCID == 0 )); then
      exec "${NIMLOTH_PYTHON}" -m ray.scripts.scripts start --head \
        --port="${NIMLOTH_RAY_PORT}" --node-ip-address="${NIMLOTH_RAY_HEAD_IP}" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-24}" --num-gpus="${count}" \
        --include-dashboard=false --block
    fi
    sleep 8
    exec "${NIMLOTH_PYTHON}" -m ray.scripts.scripts start \
      --address="${NIMLOTH_RAY_HEAD_IP}:${NIMLOTH_RAY_PORT}" \
      --num-cpus="${SLURM_CPUS_PER_TASK:-24}" --num-gpus="${count}" --block
  ' >"${RAY_LOG_DIR}/cluster.log" 2>&1 &
RAY_STEP_PID=$!

resources=""
for _ in $(seq 1 90); do
  resources=$(srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    env RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" "${PYTHON}" -c \
    'import ray; ray.init(address="auto"); print(int(ray.cluster_resources().get("GPU", 0)))' \
    2>/dev/null | tail -1 || true)
  [[ "${resources}" == "${CONFIG_WORLD_SIZE}" ]] && break
  kill -0 "${RAY_STEP_PID}" 2>/dev/null || {
    tail -200 "${RAY_LOG_DIR}/cluster.log" >&2
    exit 1
  }
  sleep 2
done
[[ "${resources}" == "${CONFIG_WORLD_SIZE}" ]] || {
  echo "Ray exposes ${resources:-0} GPUs, config requests world_size=${CONFIG_WORLD_SIZE}" >&2
  exit 1
}

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  env RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
    REPO="${REPO}" RUN_OUT="${RUN_OUT}" RL_CONFIG="${RL_CONFIG}" \
    ENV_REPO="${ENV_REPO:?set ENV_REPO}" MODEL="${MODEL:?set MODEL}" \
    WM_CKPT="${WM_CKPT:-${MODEL}}" \
    WANDB_PROJECT="${WANDB_PROJECT:-nimloth-rl}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:?set WANDB_RUN_NAME}" \
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}" \
    VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray \
    bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_smoke.sh"

wait "${RAY_STEP_PID}" 2>/dev/null || true
