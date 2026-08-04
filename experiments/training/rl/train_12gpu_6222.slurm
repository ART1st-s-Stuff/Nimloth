#!/usr/bin/env bash
# Resource components are supplied on the sbatch command line:
#   component 0: one physical node with six GPUs
#   component 1: three physical nodes with two GPUs each
# Rollout uses one TP4 worker; training uses six two-GPU ranks.
#SBATCH --job-name=nimloth-rl12-6222
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
: "${INITIAL_RESUME_CHECKPOINT:?}"
: "${REFERENCE_MODEL:?}"
: "${INITIAL_GLOBAL_STEP:?}"
: "${TOTAL_ITERATIONS:?}"
: "${WANDB_PROJECT:?}"
: "${WANDB_RUN_NAME:?}"

export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export PATH=/cm/shared/apps/slurm/current/bin:${PATH}

[[ "${SLURM_HET_SIZE}" == 2 ]] || {
  echo "12-GPU 6+2+2+2 entrypoint requires two Slurm components" >&2
  exit 1
}
mapfile -t HET_NODES_6 < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}")
mapfile -t HET_NODES_2 < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}")
[[ "${#HET_NODES_6[@]}" == 1 ]] || {
  echo "component 0 must contain one six-GPU node" >&2
  exit 1
}
[[ "${#HET_NODES_2[@]}" == 3 ]] || {
  echo "component 1 must contain three two-GPU nodes" >&2
  exit 1
}
[[ "$(printf '%s\n' "${HET_NODES_6[@]}" "${HET_NODES_2[@]}" | sort -u | wc -l)" == 4 ]] || {
  echo "6+2+2+2 allocation must contain four unique nodes" >&2
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

export NIMLOTH_HET_GPUS_PER_NODE=6,2
export ROLLOUT_WORKERS=1
export WANDB_MODE_OVERRIDE=${WANDB_MODE_OVERRIDE:-online}
export RAY_PORT_BASE=${RAY_PORT_BASE:-8000}
export ENV_PORT_BASE=${ENV_PORT_BASE:-10400}
export TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-33000}

exec bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_full.sh"
