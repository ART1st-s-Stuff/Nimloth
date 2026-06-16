#!/bin/bash
# Submit greedy val/test eval after both SFT vagen79 train jobs finish.
# Models: success-only best, all-train best, and original global_step_79 baseline.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

TRAIN_JOB_SUCCESS="${TRAIN_JOB_SUCCESS:-454527}"
TRAIN_JOB_ALL="${TRAIN_JOB_ALL:-454528}"
TRAIN_DEP="afterok:${TRAIN_JOB_SUCCESS}:${TRAIN_JOB_ALL}"

BASE_RUN=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
STEP79_HF=${ROOT}/runs/${BASE_RUN}/checkpoints/global_step_79/actor/huggingface
SUCCESS_BEST=${ROOT}/runs/sft1_train_vagen79_qwen25vl/best
ALLTRAIN_BEST=${ROOT}/runs/sft1_train_vagen79_qwen25vl_alltrain/best
ENV_READY=${ROOT}/runs/sft1_rollouts_vagen79_greedy_parallel/external_env_4gpu/ready

submit_eval() {
  local tag=$1
  local model_path=$2
  local dep=$3
  $SLURM --dependency="${dep}" --job-name="sft1-eval-v79-${tag}" \
    --export=ALL,EVAL_TAG="${tag}",MODEL_PATH="${model_path}" \
    "${ROOT}/sft1_eval_vagen79_greedy_valtest.slurm"
}

echo "=== Submit SFT1 vagen79 post-train eval at $(date) ==="
echo "Train deps: ${TRAIN_DEP}"
echo "Val params: greedy (do_sample=False, temperature=0, n=1) — same as VAGEN79 rollout/train val"

if [ -f "${ENV_READY}" ]; then
  echo "Reusing existing env ready file"
  ENV_DEP="${TRAIN_DEP}"
else
  J_ENV=$($SLURM --dependency="${TRAIN_DEP}" "${ROOT}/sft1_env_vagen79_4gpu.slurm" | awk '{print $NF}')
  echo "env job: ${J_ENV}"
  ENV_DEP="afterok:${J_ENV}"
fi

J_SUCCESS=$(submit_eval success_best "${SUCCESS_BEST}" "${ENV_DEP}" | awk '{print $NF}')
echo "eval success_best: ${J_SUCCESS} model=${SUCCESS_BEST}"

J_ALL=$(submit_eval alltrain_best "${ALLTRAIN_BEST}" "${ENV_DEP}" | awk '{print $NF}')
echo "eval alltrain_best: ${J_ALL} model=${ALLTRAIN_BEST}"

J_STEP79=$(submit_eval step79_baseline "${STEP79_HF}" "${ENV_DEP}" | awk '{print $NF}')
echo "eval step79_baseline: ${J_STEP79} model=${STEP79_HF}"

echo
echo "Outputs:"
echo "  ${ROOT}/runs/sft1_eval_vagen79_success_best/"
echo "  ${ROOT}/runs/sft1_eval_vagen79_alltrain_best/"
echo "  ${ROOT}/runs/sft1_eval_vagen79_step79_baseline/"
echo
echo "Compare summaries:"
echo "  cat ${ROOT}/runs/sft1_eval_vagen79_*/summary_0.json"
