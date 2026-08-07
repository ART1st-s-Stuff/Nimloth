#!/usr/bin/env bash
# Resources are supplied on the sbatch command line: two nodes with four GPUs
# each. This batch-owned controller keeps the full launch alive across SSH exits.
#SBATCH --job-name=nimloth-rl8-44
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err

set -euo pipefail

: "${REPO:?}"
: "${ENV_REPO:?}"
: "${PYTHON:?}"
: "${EXPECTED_COMMIT:?}"
: "${RL_CONFIG:?}"
: "${ITERATION_RUNNER:?}"
: "${RUN_OUT:?}"
: "${FORMAL_OUTPUT_ROOT:?}"
: "${INITIAL_MODEL:?}"
: "${INITIAL_WM_CKPT:?}"
: "${INITIAL_RESUME_CHECKPOINT?}"
: "${REFERENCE_MODEL:?}"
: "${INITIAL_GLOBAL_STEP:?}"
: "${TOTAL_ITERATIONS:?}"
: "${WANDB_PROJECT:?}"
: "${WANDB_RUN_NAME:?}"

RENDER_PREFLIGHT_TIMEOUT=${RENDER_PREFLIGHT_TIMEOUT:-150}
[[ "${RENDER_PREFLIGHT_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "RENDER_PREFLIGHT_TIMEOUT must be a positive integer" >&2
  exit 1
}

export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export PATH=/cm/shared/apps/slurm/current/bin:${PATH}

[[ "${SLURM_HET_SIZE:-1}" == 1 ]] || {
  echo "8-GPU 4+4 entrypoint requires one homogeneous Slurm component" >&2
  exit 1
}
mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#ALLOCATED_NODES[@]}" == 2 ]] || {
  echo "8-GPU 4+4 entrypoint requires exactly two nodes" >&2
  exit 1
}

source "${REPO}/experiments/training/rl/slurm_allocation.sh"
JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}")
declare -A GPU_COUNTS
nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
for node in "${ALLOCATED_NODES[@]}"; do
  [[ "${GPU_COUNTS[${node}]:-}" == 4 ]] || {
    echo "node ${node} does not have exactly four allocated GPUs" >&2
    exit 1
  }
done

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "commit gate failed: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=untracked)" ]] || {
  echo "tracked worktree is dirty" >&2
  exit 1
}

if [[ -n "${WANDB_ENTITY:-}" ]]; then
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
    if list(api.runs(path, filters={"display_name": name})):
        raise SystemExit(f"W&B identity already exists: {name}")
print("WANDB_IDENTITIES_OK", flush=True)
'
fi

if [[ -n "${RENDER_PREFLIGHT_OUT:-}" ]]; then
  [[ ! -e "${RENDER_PREFLIGHT_OUT}" ]] || {
    echo "renderer preflight output already exists: ${RENDER_PREFLIGHT_OUT}" >&2
    exit 1
  }
  mkdir -p "${RENDER_PREFLIGHT_OUT}"
  RENDER_STEP_PIDS=()
  for node in "${ALLOCATED_NODES[@]}"; do
    node_out=${RENDER_PREFLIGHT_OUT}/${node}
    mkdir -p "${node_out}"
    srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=1 --ntasks=1 \
      -w "${node}" --gres=gpu:4 \
      env REPO="${REPO}" PYTHON="${PYTHON}" NODE_OUT="${node_out}" \
        RENDER_PREFLIGHT_TIMEOUT="${RENDER_PREFLIGHT_TIMEOUT}" \
      bash -lc '
set -euo pipefail
IFS="," read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#allocated_gpus[@]} == 4 )) || {
  echo "renderer preflight sees ${#allocated_gpus[@]} GPUs, expected 4" >&2
  exit 1
}
export CUDA_VISIBLE_DEVICES="${allocated_gpus[0]}"
export AI2THOR_HOME_ROOT="${NODE_OUT}/home_gpu0"
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm"
timeout --signal=TERM --kill-after=10s "${RENDER_PREFLIGHT_TIMEOUT}s" \
  "${PYTHON}" -m nimloth.environment.navigation.direct_render_probe \
    --gpu-device 0
' >"${node_out}/gpu0.log" 2>&1 &
    RENDER_STEP_PIDS+=("$!")
  done
  render_status=0
  for pid in "${RENDER_STEP_PIDS[@]}"; do
    wait "${pid}" || render_status=$?
  done
  for node in "${ALLOCATED_NODES[@]}"; do
    node_log=${RENDER_PREFLIGHT_OUT}/${node}/gpu0.log
    cat "${node_log}"
    grep -Fq '"status": "AI2THOR_RENDER_OK"' "${node_log}" || render_status=1
  done
  (( render_status == 0 )) || {
    echo "one or more allocated nodes failed the renderer preflight" >&2
    exit "${render_status}"
  }
  echo "RENDER_PREFLIGHT_ALL_OK nodes=${ALLOCATED_NODES[*]}"
fi

export ROLLOUT_WORKERS=2
export WANDB_MODE_OVERRIDE=${WANDB_MODE_OVERRIDE:-online}
export RAY_PORT_BASE=${RAY_PORT_BASE:-7400}
export ENV_PORT_BASE=${ENV_PORT_BASE:-9500}
export TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-32300}

exec bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_full.sh"
