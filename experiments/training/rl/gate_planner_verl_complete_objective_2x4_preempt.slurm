#!/bin/bash
#SBATCH --job-name=planner-complete-fsdp
#SBATCH --partition=preempt
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/planner-complete-fsdp_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/planner-complete-fsdp_%j.err

set -euo pipefail

: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${MODEL:?MODEL is required}"
: "${WM_CHECKPOINT:?WM_CHECKPOINT is required}"
: "${STATE_PROJ_CHECKPOINT:?STATE_PROJ_CHECKPOINT is required}"
: "${VALUE_HEAD_CHECKPOINT:?VALUE_HEAD_CHECKPOINT is required}"
: "${PLANNER_POLICY_HEAD_CHECKPOINT:?PLANNER_POLICY_HEAD_CHECKPOINT is required}"
: "${TRAJECTORY_JSONL:?TRAJECTORY_JSONL is required}"
: "${FRESH_ROLLOUT_MANIFEST:?FRESH_ROLLOUT_MANIFEST is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"
GATE_MINIMUM_STATE_TOKENS=${GATE_MINIMUM_STATE_TOKENS:-6000}
GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX=${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX:-false}
GATE_EXPECTED_MANIFEST_SHA256=${GATE_EXPECTED_MANIFEST_SHA256:-}
GATE_EXPECTED_TRAJECTORY_COUNT=${GATE_EXPECTED_TRAJECTORY_COUNT:-}
GATE_EXPECTED_SELECTED_RECORD_ID=${GATE_EXPECTED_SELECTED_RECORD_ID:-}
GATE_EXPECTED_SELECTED_STATE_TOKENS=${GATE_EXPECTED_SELECTED_STATE_TOKENS:-}
GATE_EXPECTED_CURRENT_POLICY_SHA256=${GATE_EXPECTED_CURRENT_POLICY_SHA256:-}
GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256=${GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256:-}
GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256=${GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256:-}
GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256=${GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256:-}
GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256=${GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256:-}
if [[ "${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX}" == true ]]; then
  (( GATE_MINIMUM_STATE_TOKENS >= 14000 )) || {
    echo "nonbehavior diagnostic requires >=14000 real state tokens" >&2
    exit 2
  }
  : "${GATE_EXPECTED_MANIFEST_SHA256:?diagnostic manifest SHA256 is required}"
  : "${GATE_EXPECTED_TRAJECTORY_COUNT:?diagnostic trajectory count is required}"
  : "${GATE_EXPECTED_SELECTED_RECORD_ID:?diagnostic selected record is required}"
  : "${GATE_EXPECTED_SELECTED_STATE_TOKENS:?diagnostic selected tokens are required}"
  : "${GATE_EXPECTED_CURRENT_POLICY_SHA256:?current policy SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256:?current StateProjector SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256:?current WM SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256:?current ValueHead SHA256 is required}"
  : "${GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256:?current PlannerPolicyHead SHA256 is required}"
elif [[ "${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX}" != false ]]; then
  echo "GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX must be true or false" >&2
  exit 2
fi

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
RAY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/ray
CONFIG=${REPO}/configs/training/rl/planner_policy_h1_complete_objective_fsdp_gate.yaml
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm

REQUESTED_WANDB_PROJECT="${WANDB_PROJECT}"
REQUESTED_WANDB_RUN_NAME="${WANDB_RUN_NAME}"
REQUESTED_WANDB_RUN_ID="${WANDB_RUN_ID}"
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
export WANDB_PROJECT="${REQUESTED_WANDB_PROJECT}"
export WANDB_RUN_NAME="${REQUESTED_WANDB_RUN_NAME}"
export WANDB_RUN_ID="${REQUESTED_WANDB_RUN_ID}"

cd "${REPO}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git status --short)" ]]
[[ "$(git -C external/VAGEN rev-parse HEAD)" == "192c35a91f3941b72d5e1272af6603ef7a7d93e0" ]]
[[ "$(git -C external/VAGEN/verl rev-parse HEAD)" == "65316156d1011d71d62e0542e4b954f9499e872e" ]]
[[ "$(git -C external/le-wm rev-parse HEAD)" == "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac" ]]
[[ ! -e "${OUTPUT_DIR}" ]]
[[ -x "${PYTHON}" ]]
[[ -x "${RAY}" ]]
[[ -f "${CONFIG}" ]]
[[ -d "${MODEL}" ]]
[[ -d "${WM_CHECKPOINT}" ]]
[[ -f "${STATE_PROJ_CHECKPOINT}" ]]
[[ -d "${VALUE_HEAD_CHECKPOINT}" ]]
[[ -d "${PLANNER_POLICY_HEAD_CHECKPOINT}" ]]
[[ -f "${TRAJECTORY_JSONL}" ]]
[[ -f "${FRESH_ROLLOUT_MANIFEST}" ]]
[[ ! -e "${FRESH_ROLLOUT_MANIFEST}.consumption.json" ]]

export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#RAY_NODES[@]}" -eq 2 ]]
HEAD_NODE=${RAY_NODES[0]}
WORKER_NODE=${RAY_NODES[1]}
[[ "$(hostname -s)" == "${HEAD_NODE}" ]]
RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))
RAY_TEMP_DIR=/tmp/nimloth-ray-${USER}-${SLURM_JOB_ID}-${RAY_PORT}

mapfile -t FABRIC_ROWS < <(
  srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus=0 \
    bash -lc '
      row=$(ip -o -4 addr show | awk '\''$4 ~ /^10\.23\./ {split($4, a, "/"); print a[1], $2; exit}'\'')
      [[ -n "${row}" ]]
      printf "%s %s\n" "$(hostname -s)" "${row}"
    '
)
[[ "${#FABRIC_ROWS[@]}" -eq 2 ]]
declare -A NODE_IP NODE_IFACE
for row in "${FABRIC_ROWS[@]}"; do
  read -r node ip_address interface <<<"${row}"
  [[ -z "${NODE_IP[${node}]:-}" ]]
  NODE_IP[${node}]=${ip_address}
  NODE_IFACE[${node}]=${interface}
done
[[ -n "${NODE_IP[${HEAD_NODE}]:-}" && -n "${NODE_IP[${WORKER_NODE}]:-}" ]]
[[ "${NODE_IFACE[${HEAD_NODE}]}" == "${NODE_IFACE[${WORKER_NODE}]}" ]]
HEAD_IP=${NODE_IP[${HEAD_NODE}]}
WORKER_IP=${NODE_IP[${WORKER_NODE}]}
FABRIC_IFACE=${NODE_IFACE[${HEAD_NODE}]}
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_EXPECTED_NODE_IPS="${HEAD_IP},${WORKER_IP}"
export NCCL_SOCKET_IFNAME="${FABRIC_IFACE}"
export GLOO_SOCKET_IFNAME="${FABRIC_IFACE}"

RAY_STEP_PIDS=()
cleanup_ray() {
  set +e
  for pid in "${RAY_STEP_PIDS[@]}"; do
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done
  for _ in $(seq 1 20); do
    alive=false
    for pid in "${RAY_STEP_PIDS[@]}"; do
      kill -0 "${pid}" >/dev/null 2>&1 && alive=true
    done
    [[ "${alive}" == false ]] && break
    sleep 1
  done
  for pid in "${RAY_STEP_PIDS[@]}"; do
    kill -KILL "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  done
  timeout 15s srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus=0 env RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
    timeout 10s bash -lc \
    'pkill -TERM -f "${RAY_TEMP_DIR}" >/dev/null 2>&1 || true; rm -rf "${RAY_TEMP_DIR}"' \
    >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus=0 \
  env RAY_TEMP_DIR="${RAY_TEMP_DIR}" bash -lc '[[ ! -e "${RAY_TEMP_DIR}" ]]'
srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" --gpus=0 \
  env HEAD_IP="${HEAD_IP}" RAY_PORT="${RAY_PORT}" "${PYTHON}" - <<'PY'
import os
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind((os.environ["HEAD_IP"], int(os.environ["RAY_PORT"])))
PY

echo "PLANNER_COMPLETE_FSDP_START commit=${EXPECTED_COMMIT} job=${SLURM_JOB_ID} nodes=${RAY_NODES[*]} ips=${HEAD_IP},${WORKER_IP} iface=${FABRIC_IFACE} ray_port=${RAY_PORT} diagnostic_nonbehavior=${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX} minimum_tokens=${GATE_MINIMUM_STATE_TOKENS}"
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader

srun --overlap --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
  --gpus-per-task=4 --cpus-per-task=32 \
  "${RAY}" start --block --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --temp-dir="${RAY_TEMP_DIR}" \
    --num-cpus=32 --num-gpus=4 --include-dashboard=false &
RAY_STEP_PIDS+=("$!")
sleep 5
srun --overlap --nodes=1 --ntasks=1 --nodelist="${WORKER_NODE}" \
  --gpus-per-task=4 --cpus-per-task=32 \
  "${RAY}" start --block --address="${RAY_ADDRESS}" \
    --node-ip-address="${WORKER_IP}" \
    --temp-dir="${RAY_TEMP_DIR}" \
    --num-cpus=32 --num-gpus=4 &
RAY_STEP_PIDS+=("$!")
for _ in $(seq 1 30); do
  if "${PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"])
alive = [node for node in ray.nodes() if node["Alive"]]
counts = sorted(float(node["Resources"].get("GPU", 0)) for node in alive)
addresses = sorted(str(node["NodeManagerAddress"]) for node in alive)
ray.shutdown()
expected_addresses = sorted(os.environ["RAY_EXPECTED_NODE_IPS"].split(","))
if counts != [4.0, 4.0] or addresses != expected_addresses:
    raise RuntimeError((counts, addresses, expected_addresses))
PY
  then
    break
  fi
  sleep 2
done
"${PYTHON}" - <<'PY'
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"])
alive = [node for node in ray.nodes() if node["Alive"]]
counts = sorted(float(node["Resources"].get("GPU", 0)) for node in alive)
addresses = sorted(str(node["NodeManagerAddress"]) for node in alive)
print({
    "ray_address": os.environ["RAY_ADDRESS"],
    "node_addresses": addresses,
    "gpu_counts": counts,
    "nccl_socket_ifname": os.environ["NCCL_SOCKET_IFNAME"],
})
ray.shutdown()
expected_addresses = sorted(os.environ["RAY_EXPECTED_NODE_IPS"].split(","))
if counts != [4.0, 4.0] or addresses != expected_addresses:
    raise RuntimeError((counts, addresses, expected_addresses))
PY

GATE_EXTRA_ARGS=()
if [[ "${GATE_DIAGNOSTIC_NONBEHAVIOR_PREFIX}" == true ]]; then
  GATE_EXTRA_ARGS+=(
    --diagnostic-nonbehavior-prefix
    --expected-manifest-sha256 "${GATE_EXPECTED_MANIFEST_SHA256}"
    --expected-trajectory-count "${GATE_EXPECTED_TRAJECTORY_COUNT}"
    --expected-selected-record-id "${GATE_EXPECTED_SELECTED_RECORD_ID}"
    --expected-selected-state-tokens "${GATE_EXPECTED_SELECTED_STATE_TOKENS}"
    --expected-current-policy-sha256 "${GATE_EXPECTED_CURRENT_POLICY_SHA256}"
    --expected-current-state-projector-sha256 "${GATE_EXPECTED_CURRENT_STATE_PROJECTOR_SHA256}"
    --expected-current-wm-predictor-sha256 "${GATE_EXPECTED_CURRENT_WM_PREDICTOR_SHA256}"
    --expected-current-value-head-sha256 "${GATE_EXPECTED_CURRENT_VALUE_HEAD_SHA256}"
    --expected-current-planner-policy-head-sha256 "${GATE_EXPECTED_CURRENT_PLANNER_POLICY_HEAD_SHA256}"
  )
fi

"${PYTHON}" experiments/training/rl/gate_planner_verl_complete_objective.py \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --state-proj-checkpoint "${STATE_PROJ_CHECKPOINT}" \
  --value-head-checkpoint "${VALUE_HEAD_CHECKPOINT}" \
  --planner-policy-head-checkpoint "${PLANNER_POLICY_HEAD_CHECKPOINT}" \
  --trajectory-jsonl "${TRAJECTORY_JSONL}" \
  --fresh-rollout-manifest "${FRESH_ROLLOUT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --world-size 8 \
  --workers-per-node 4 \
  --expected-node-count 2 \
  --minimum-state-tokens "${GATE_MINIMUM_STATE_TOKENS}" \
  "${GATE_EXTRA_ARGS[@]}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"

echo "PLANNER_COMPLETE_FSDP_ALL_OK output=${OUTPUT_DIR}"
