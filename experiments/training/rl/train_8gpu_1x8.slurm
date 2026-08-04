#!/usr/bin/env bash
# Resources are supplied on the sbatch command line: one node with eight GPUs.
# This batch-owned controller keeps the full launch alive across SSH exits.
#SBATCH --job-name=nimloth-rl8-1x8
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

export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export PATH=/cm/shared/apps/slurm/current/bin:${PATH}

[[ "${SLURM_HET_SIZE:-1}" == 1 ]] || {
  echo "8-GPU 1x8 entrypoint requires one homogeneous Slurm component" >&2
  exit 1
}
mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#ALLOCATED_NODES[@]}" == 1 ]] || {
  echo "8-GPU 1x8 entrypoint requires exactly one node" >&2
  exit 1
}

source "${REPO}/experiments/training/rl/slurm_allocation.sh"
JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}")
declare -A GPU_COUNTS
nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
node=${ALLOCATED_NODES[0]}
[[ "${GPU_COUNTS[${node}]:-}" == 8 ]] || {
  echo "node ${node} does not have exactly eight allocated GPUs" >&2
  exit 1
}

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "commit gate failed: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=untracked)" ]] || {
  echo "tracked worktree is dirty" >&2
  exit 1
}

export ROLLOUT_WORKERS=2
export WANDB_MODE_OVERRIDE=${WANDB_MODE_OVERRIDE:-online}
export RAY_PORT_BASE=${RAY_PORT_BASE:-7420}
export ENV_PORT_BASE=${ENV_PORT_BASE:-9520}
export TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-32320}

exec bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_full.sh"
