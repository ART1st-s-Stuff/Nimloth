#!/usr/bin/env bash
# Wait for one existing 1x8 hold, run exact-node preflights, then launch the
# batch-owned RL controller as a detached Slurm step.
set -euo pipefail

: "${HOLD_JOB:?}"
: "${REPO:?}"
: "${ENV_REPO:?}"
: "${PYTHON:?}"
: "${EXPECTED_COMMIT:?}"
: "${RL_CONFIG:?}"
: "${RUN_OUT:?}"
: "${FORMAL_OUTPUT_ROOT:?}"
: "${INITIAL_MODEL:?}"
: "${INITIAL_WM_CKPT:?}"
: "${INITIAL_RESUME_CHECKPOINT:?}"
: "${REFERENCE_MODEL:?}"
: "${INITIAL_GLOBAL_STEP:?}"
: "${FIRST_ITERATION_SEED_OFFSET:?}"
: "${TOTAL_ITERATIONS:?}"
: "${WANDB_ENTITY:?}"
: "${WANDB_PROJECT:?}"
: "${WANDB_RUN_NAME:?}"
: "${WANDB_ENV_FILE:?}"
: "${PREFLIGHT_OUT:?}"
: "${WAIT_LOG:?}"
: "${CLIENT_LOG:?}"
: "${CLIENT_PID_FILE:?}"

SLURM_BIN_DIR=${SLURM_BIN_DIR:-/cm/shared/apps/slurm/current/bin}
SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
FORBIDDEN_NODES=${FORBIDDEN_NODES:-dgx-32,dgx-37,dgx-51}
POLL_SECONDS=${POLL_SECONDS:-30}
CONTROLLER_TIMEOUT_SECONDS=${CONTROLLER_TIMEOUT_SECONDS:-8400}
RAY_PORT_BASE=${RAY_PORT_BASE:-7420}
ENV_PORT_BASE=${ENV_PORT_BASE:-9520}
TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-32320}
export SLURM_CONF
export PATH="${SLURM_BIN_DIR}:${PATH}"

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 1
}
[[ "${CONTROLLER_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "CONTROLLER_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
}
[[ "${INITIAL_GLOBAL_STEP}" =~ ^[0-9]+$ ]] || {
  echo "INITIAL_GLOBAL_STEP must be a non-negative integer" >&2
  exit 1
}
[[ "${TOTAL_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "TOTAL_ITERATIONS must be a positive integer" >&2
  exit 1
}
(( INITIAL_GLOBAL_STEP < TOTAL_ITERATIONS )) || {
  echo "initial checkpoint must precede TOTAL_ITERATIONS" >&2
  exit 1
}

mkdir -p "$(dirname "${WAIT_LOG}")" "$(dirname "${CLIENT_LOG}")" \
  "$(dirname "${CLIENT_PID_FILE}")"
exec >>"${WAIT_LOG}" 2>&1
echo "WAIT_START time=$(date -Iseconds) hold=${HOLD_JOB} commit=${EXPECTED_COMMIT}"

while true; do
  state=$(squeue -h -j "${HOLD_JOB}" -o '%T')
  reason=$(squeue -h -j "${HOLD_JOB}" -o '%R')
  echo "WAIT_STATUS time=$(date -Iseconds) state=${state:-absent} reason=${reason:-none}"
  case "${state}" in
    RUNNING) break ;;
    PENDING) sleep "${POLL_SECONDS}" ;;
    "") echo "hold disappeared before launch" >&2; exit 1 ;;
    *) echo "hold reached unexpected state before launch: ${state}" >&2; exit 1 ;;
  esac
done

mapfile -t nodes < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o '%N')")
(( ${#nodes[@]} == 1 )) || {
  echo "hold does not contain exactly one node: ${nodes[*]}" >&2
  exit 1
}
node=${nodes[0]}
IFS=',' read -r -a forbidden_nodes <<< "${FORBIDDEN_NODES}"
for forbidden in "${forbidden_nodes[@]}"; do
  [[ "${node}" != "${forbidden}" ]] || {
    echo "allocated node is forbidden: ${node}" >&2
    exit 1
  }
done

source "${REPO}/experiments/training/rl/slurm_allocation.sh"
job_details=$(scontrol show job -dd "${HOLD_JOB}")
declare -A gpu_counts
nimloth_load_slurm_gpu_counts "${job_details}" gpu_counts
[[ "${gpu_counts[${node}]:-}" == 8 ]] || {
  echo "hold does not allocate exactly eight GPUs on ${node}" >&2
  exit 1
}

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "server worktree commit changed before launch" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=untracked)" ]] || {
  echo "server worktree has tracked changes" >&2
  exit 1
}
[[ -f "${RL_CONFIG}" ]] || { echo "missing RL config" >&2; exit 1; }
[[ -s "${INITIAL_RESUME_CHECKPOINT}/rl_state.pt" ]] || {
  echo "missing optimizer resume checkpoint" >&2
  exit 1
}
[[ ! -e "${RUN_OUT}" && ! -e "${RUN_OUT}.iteration_progress.log" ]] || {
  echo "formal output already exists before first launch" >&2
  exit 1
}
[[ ! -e "${PREFLIGHT_OUT}" ]] || {
  echo "renderer preflight output already exists" >&2
  exit 1
}

set -a
source "${WANDB_ENV_FILE}"
set +a
PYTHONPATH="${REPO}/src" \
  NIMLOTH_WANDB_ENTITY="${WANDB_ENTITY}" \
  NIMLOTH_WANDB_PROJECT="${WANDB_PROJECT}" \
  NIMLOTH_WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  "${PYTHON}" -c '
import os
import wandb

api = wandb.Api()
path = f"{os.environ["NIMLOTH_WANDB_ENTITY"]}/{os.environ["NIMLOTH_WANDB_PROJECT"]}"
for name in (
    os.environ["NIMLOTH_WANDB_RUN_NAME"],
    os.environ["NIMLOTH_WANDB_RUN_NAME"] + "-eval",
):
    matches = list(api.runs(path, filters={"display_name": name}))
    if matches:
        raise SystemExit(f"W&B name already exists: {name}")
print("WANDB_IDENTITIES_OK", flush=True)
'
INITIAL_RESUME_CHECKPOINT="${INITIAL_RESUME_CHECKPOINT}" \
  INITIAL_GLOBAL_STEP="${INITIAL_GLOBAL_STEP}" \
  "${PYTHON}" -c '
import os
import torch

path = os.path.join(os.environ["INITIAL_RESUME_CHECKPOINT"], "rl_state.pt")
state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
expected = int(os.environ["INITIAL_GLOBAL_STEP"])
actual_step = state.get("global_step")
if int(actual_step if actual_step is not None else -1) != expected:
    raise SystemExit(f"resume global step mismatch: {actual_step}")
actual_objective = state.get("objective")
if actual_objective != "receding_horizon_decision_state_ppo_value_v1":
    raise SystemExit(f"resume objective mismatch: {actual_objective}")
print("RESUME_CHECKPOINT_OK", flush=True)
'

mkdir -p "${PREFLIGHT_OUT}"
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
  --gres=gpu:8 \
  env REPO="${REPO}" PYTHON="${PYTHON}" PREFLIGHT_OUT="${PREFLIGHT_OUT}" \
  bash -lc '
set -euo pipefail
IFS="," read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#allocated_gpus[@]} == 8 )) || {
  echo "renderer preflight sees ${#allocated_gpus[@]} GPUs, expected 8" >&2
  exit 1
}
pids=()
for slot in 0 4; do
  (
    export CUDA_VISIBLE_DEVICES="${allocated_gpus[${slot}]}"
    export AI2THOR_HOME_ROOT="${PREFLIGHT_OUT}/home_gpu${slot}"
    source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
    export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm"
    timeout --signal=TERM --kill-after=10s 150s \
      "${PYTHON}" -m nimloth.environment.navigation.direct_render_probe \
        --gpu-device 0
  ) >"${PREFLIGHT_OUT}/gpu${slot}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
for slot in 0 4; do
  cat "${PREFLIGHT_OUT}/gpu${slot}.log"
  grep -Fq "\"status\": \"AI2THOR_RENDER_OK\"" \
    "${PREFLIGHT_OUT}/gpu${slot}.log" || status=1
done
exit "${status}"
'

port_list=()
for ((iteration=INITIAL_GLOBAL_STEP + 1; iteration<=TOTAL_ITERATIONS; iteration++)); do
  port_list+=(
    "$((RAY_PORT_BASE + iteration))"
    "$((ENV_PORT_BASE + iteration))"
    "$((ENV_PORT_BASE + iteration + 1))"
    "$((TRAIN_MASTER_PORT_BASE + iteration))"
  )
done
port_list+=(
  "$((ENV_PORT_BASE + TOTAL_ITERATIONS + 100))"
  "$((ENV_PORT_BASE + TOTAL_ITERATIONS + 101))"
)
ports_csv=$(IFS=,; echo "${port_list[*]}")
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
  --gres=gpu:8 env NIMLOTH_PORTS="${ports_csv}" bash -lc '
set -euo pipefail
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed "/^[[:space:]]*$/d")
[[ -z "${gpu_processes}" ]] || {
  echo "GPU processes remain after renderer preflight: ${gpu_processes}" >&2
  exit 1
}
IFS="," read -r -a ports <<< "${NIMLOTH_PORTS}"
listeners=$(ss -ltnH)
for port in "${ports[@]}"; do
  if awk -v suffix=":${port}" '$4 ~ suffix "$" {found=1} END {exit !found}' <<< "${listeners}"; then
    echo "required port is already listening: ${port}" >&2
    exit 1
  fi
done
echo "NODE_CLEAN_OK"
'

echo "LAUNCH_PREFLIGHT_OK time=$(date -Iseconds) node=${node}"
nohup timeout --signal=TERM --kill-after=30s "${CONTROLLER_TIMEOUT_SECONDS}s" \
  srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "${node}" \
    --gres=gpu:8 \
    env REPO="${REPO}" ENV_REPO="${ENV_REPO}" PYTHON="${PYTHON}" \
      EXPECTED_COMMIT="${EXPECTED_COMMIT}" RL_CONFIG="${RL_CONFIG}" \
      ITERATION_RUNNER="${REPO}/experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh" \
      EVALUATION_RUNNER="${REPO}/experiments/training/rl/run_vllm_online_ppo_parallel_slurm.sh" \
      RUN_OUT="${RUN_OUT}" FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT}" \
      INITIAL_MODEL="${INITIAL_MODEL}" INITIAL_WM_CKPT="${INITIAL_WM_CKPT}" \
      INITIAL_RESUME_CHECKPOINT="${INITIAL_RESUME_CHECKPOINT}" \
      REFERENCE_MODEL="${REFERENCE_MODEL}" \
      INITIAL_GLOBAL_STEP="${INITIAL_GLOBAL_STEP}" \
      FIRST_ITERATION_SEED_OFFSET="${FIRST_ITERATION_SEED_OFFSET}" \
      TOTAL_ITERATIONS="${TOTAL_ITERATIONS}" \
      WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
      WANDB_MODE_OVERRIDE=online RAY_PORT_BASE="${RAY_PORT_BASE}" \
      ENV_PORT_BASE="${ENV_PORT_BASE}" \
      TRAIN_MASTER_PORT_BASE="${TRAIN_MASTER_PORT_BASE}" \
      bash "${REPO}/experiments/training/rl/train_8gpu_1x8.slurm" \
  </dev/null >"${CLIENT_LOG}" 2>&1 &
client_pid=$!
printf '%s\n' "${client_pid}" > "${CLIENT_PID_FILE}"
echo "DETACHED_SRUN_LAUNCHED time=$(date -Iseconds) pid=${client_pid} node=${node}"

set +e
wait "${client_pid}"
client_status=$?
set -e
echo "DETACHED_SRUN_EXIT time=$(date -Iseconds) status=${client_status}"
exit "${client_status}"
