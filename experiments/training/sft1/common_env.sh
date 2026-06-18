#!/usr/bin/env bash
# Shared paths and environment for SFT1 jobs.
# Source from Slurm scripts: source "${SCRIPTDIR}/common_env.sh"

REPO="${REPO:-/project/peilab/atst/nimloth}"
SCRIPTDIR="${REPO}/experiments/training/sft1"
BASELINE_SCRIPTDIR="${REPO}/experiments/training/baseline"

# Legacy run artifacts remain under navigation_baseline/runs until migrated.
SFT1_RUNS_ROOT="${SFT1_RUNS_ROOT:-${REPO}/experiments/navigation_baseline/runs}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2}"
INIT_HF_STEP="${INIT_HF_STEP:-79}"
INIT_HF="${INIT_HF:-${SFT1_RUNS_ROOT}/${BASELINE_RUN_NAME}/checkpoints/global_step_${INIT_HF_STEP}/actor/huggingface}"
RECORDS_ROOT="${RECORDS_ROOT:-${SFT1_RUNS_ROOT}/sft1_sft_records_vagen79_nimloth_format}"
ROLLOUT_RUN_NAME="${ROLLOUT_RUN_NAME:-sft1_rollouts_vagen79_greedy_parallel}"
ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR:-${SFT1_RUNS_ROOT}/${ROLLOUT_RUN_NAME}}"

export UV_CACHE_DIR="${REPO}/.cache/uv"
export HOME="${REPO}/.home"
export WANDB_DIR="${REPO}/.cache/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${REPO}/.venv/bin:${REPO}/.local/bin:$PATH"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export PYTHONPATH="${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${PYTHONPATH:-}"
mkdir -p "$HOME" "$WANDB_DIR"

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/.env
  set +a
fi

# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
