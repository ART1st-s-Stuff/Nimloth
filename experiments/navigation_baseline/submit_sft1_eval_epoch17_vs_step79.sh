#!/bin/bash
# Greedy val/test eval: SFT success epoch_017 vs VAGEN global_step_79 baseline.
# Decode matches training-time val: do_sample=False, temperature=0, n=1.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

BASE_RUN=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
STEP79_HF=${ROOT}/runs/${BASE_RUN}/checkpoints/global_step_79/actor/huggingface
EPOCH17=${ROOT}/runs/sft1_train_vagen79_qwen25vl/epoch_017
ENV_READY=${ROOT}/runs/sft1_rollouts_vagen79_greedy_parallel/external_env_4gpu/ready

submit_eval() {
  local tag=$1
  local model_path=$2
  local dep=$3
  local dep_args=()
  if [ -n "${dep}" ] && [ "${dep}" != "none" ]; then
    dep_args=(--dependency="${dep}")
  fi
  $SLURM "${dep_args[@]}" --job-name="sft1-eval-v79-${tag}" \
    --export=ALL,EVAL_TAG="${tag}",MODEL_PATH="${model_path}" \
    "${ROOT}/sft1_eval_vagen79_greedy_valtest.slurm"
}

echo "=== Submit epoch17 vs step79 eval at $(date) ==="
echo "SFT model: ${EPOCH17} (val_loss=7.295 at epoch 17)"
echo "Baseline:  ${STEP79_HF}"
echo "Val: greedy do_sample=False temperature=0 n=1 (VAGEN79 train val)"

# Drop stale pending eval jobs only (keep running env).
$SCANCEL 454605 454606 454607 454608 2>/dev/null || true

if [ ! -f "${EPOCH17}/model-00001-of-00002.safetensors" ]; then
  echo "ERROR missing epoch_017 checkpoint"
  exit 2
fi

if [ -f "${ENV_READY}" ]; then
  echo "Reusing env ready: ${ENV_READY} (env job may still be RUNNING on dgx-12)"
  ENV_DEP="none"
elif [ -n "${ENV_JOB:-}" ] && squeue -j "${ENV_JOB}" -h 2>/dev/null | grep -q .; then
  echo "Waiting for env job ${ENV_JOB} ready file"
  ENV_DEP="after:${ENV_JOB}"
else
  J_ENV=$($SLURM "${ROOT}/sft1_env_vagen79_4gpu.slurm" | awk '{print $NF}')
  echo "env job: ${J_ENV}"
  ENV_DEP="after:${J_ENV}"
fi

J_E17=$(submit_eval success_epoch17 "${EPOCH17}" "${ENV_DEP}" | awk '{print $NF}')
echo "eval epoch17: ${J_E17}"

J_S79=$(submit_eval step79_baseline "${STEP79_HF}" "${ENV_DEP}" | awk '{print $NF}')
echo "eval step79:  ${J_S79}"

echo
echo "Results:"
echo "  ${ROOT}/runs/sft1_eval_vagen79_success_epoch17/summary_0.json"
echo "  ${ROOT}/runs/sft1_eval_vagen79_step79_baseline/summary_0.json"
echo
echo "Compare after jobs finish:"
echo "  python3 ${ROOT}/compare_sft1_eval_summaries.py \\"
echo "    ${ROOT}/runs/sft1_eval_vagen79_success_epoch17/summary_0.json \\"
echo "    ${ROOT}/runs/sft1_eval_vagen79_step79_baseline/summary_0.json"
