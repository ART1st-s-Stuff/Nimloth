#!/usr/bin/env bash
# Shared paths for SFT2 jobs.
# Source from Slurm scripts: source "${SCRIPTDIR}/common_env.sh"

REPO="${REPO:-/project/peilab/atst/nimloth}"
SCRIPTDIR="${REPO}/experiments/training/sft2"
SFT1="${REPO}/experiments/training/sft1"
SFT1_RUNS="${SFT1_RUNS:-${REPO}/experiments/navigation_baseline/runs}"
SFT2_OUTPUTS_ROOT="${REPO}/outputs/experiments/training/sft2"

RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sft2_latentwm_default}"
TRAIN_OUT="${TRAIN_OUT:-${SFT2_OUTPUTS_ROOT}/${RUN_DATE}/${EXPERIMENT_NAME}}"

RECORDS_ROOT="${RECORDS_ROOT:-${SFT1_RUNS}/sft1_sft_records_vagen79_nimloth_format}"
SFT1_RUN="${SFT1_RUN:-${SFT1_RUNS}/sft1_train_vagen79_qwen25vl_alltrain_8gpu_lora}"
BASE_HF="${BASE_HF:-${SFT1_RUNS}/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface}"
