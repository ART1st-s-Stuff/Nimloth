#!/usr/bin/env bash
# Drive the vLLM/PPO smoke inside one held 2-node, 4-GPU-per-node allocation.
set -euo pipefail

HOLD_JOB=${HOLD_JOB:?set HOLD_JOB to a running two-node allocation}
REPO=${REPO:?set REPO to the committed server worktree}
RUN_OUT=${RUN_OUT:?set RUN_OUT to a new output directory}
PYTHON=${PYTHON:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
RAY_PORT=${RAY_PORT:-6381}
RAY_LOG_DIR=${RUN_OUT}.ray

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o %N)")
(( ${#NODES[@]} == 2 )) || { echo "expected two held nodes, got: ${NODES[*]}" >&2; exit 1; }
HEAD_NODE=${NODES[0]}
WORKER_NODE=${NODES[1]}
HEAD_IP=$(srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[[ -n "${HEAD_IP}" ]] || { echo "could not resolve Ray head IP" >&2; exit 1; }
mkdir -p "${RAY_LOG_DIR}"

stop_ray() {
  for node in "${NODES[@]}"; do
    srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
      "${PYTHON}" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  done
}
trap stop_ray EXIT
stop_ray

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON}" -m ray.scripts.scripts start --head \
    --port="${RAY_PORT}" --num-cpus=80 --num-gpus=4 \
    --node-ip-address="${HEAD_IP}" --include-dashboard=false --block \
  >"${RAY_LOG_DIR}/head.log" 2>&1 &
sleep 10
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${WORKER_NODE}" \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON}" -m ray.scripts.scripts start \
    --address="${HEAD_IP}:${RAY_PORT}" --num-cpus=80 --num-gpus=4 --block \
  >"${RAY_LOG_DIR}/worker.log" 2>&1 &

for _ in $(seq 1 60); do
  resources=$(srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    env RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" "${PYTHON}" -c \
    'import ray; ray.init(address="auto"); print(int(ray.cluster_resources().get("GPU", 0)))' \
    2>/dev/null | tail -1 || true)
  [[ "${resources}" == 8 ]] && break
  sleep 2
done
[[ "${resources:-}" == 8 ]] || { echo "Ray cluster did not expose exactly 8 GPUs" >&2; exit 1; }

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
    REPO="${REPO}" RUN_OUT="${RUN_OUT}" \
    ENV_REPO="${ENV_REPO:?set ENV_REPO}" MODEL="${MODEL:?set MODEL}" \
    WM_CKPT="${WM_CKPT:-${MODEL}}" RL_CONFIG="${RL_CONFIG:-${REPO}/configs/training/rl/e2e_smoke_h4.yaml}" \
    WANDB_PROJECT="${WANDB_PROJECT:-nimloth-rl}" WANDB_RUN_NAME="${WANDB_RUN_NAME:?set WANDB_RUN_NAME}" \
    WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}" \
    TENSOR_PARALLEL_SIZE=8 VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray \
    TRAIN_NNODES=2 TRAIN_GPUS_PER_NODE=4 \
    bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_smoke.sh"

