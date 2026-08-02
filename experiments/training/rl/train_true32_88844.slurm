#!/usr/bin/env bash
# Resource components are supplied on the sbatch command line:
#   component 0: 3 physical nodes, 8 GPUs/node
#   component 1: 2 physical nodes, 4 GPUs/node
# Together they provide eight TP4 rollout workers and sixteen 2-GPU ranks.
#SBATCH --job-name=nimloth-rl32-88844
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

test "${SLURM_HET_SIZE}" -eq 2
mapfile -t HET_NODES_8 < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}")
mapfile -t HET_NODES_4 < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}")
test "${#HET_NODES_8[@]}" -eq 3
test "${#HET_NODES_4[@]}" -eq 2
test "$(printf '%s\n' "${HET_NODES_8[@]}" "${HET_NODES_4[@]}" | sort -u | wc -l)" -eq 5

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "commit gate failed: ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree is dirty" >&2
  exit 1
}

export NIMLOTH_HET_GPUS_PER_NODE=8,4
export WANDB_MODE_OVERRIDE=${WANDB_MODE_OVERRIDE:-online}
export RAY_PORT_BASE=${RAY_PORT_BASE:-7200}
export ENV_PORT_BASE=${ENV_PORT_BASE:-9300}
export TRAIN_MASTER_PORT_BASE=${TRAIN_MASTER_PORT_BASE:-32100}

exec bash "${REPO}/experiments/training/rl/run_vllm_online_ppo_full.sh"
