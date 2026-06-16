#!/bin/bash
# Greedy val/test eval: SFT epoch_017 with Nimloth prompt_format (matches SFT training).
# Compare result to pre-SFT VAGEN greedy rollout (global_step_79, XML prompt) success rate.
#
# Set RESTART_ENV=1 to start a fresh env server (required after VAGEN prompt/parse changes).
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

EPOCH17=${ROOT}/runs/sft1_train_vagen79_qwen25vl/epoch_017
ROLLOUT_BASELINE=${ROOT}/runs/sft1_rollouts_vagen79_greedy_parallel
BASELINE_SUMMARY=${ROLLOUT_BASELINE}/summary_baseline_step79_rollout.json
BASELINE_JSONL_STEP=0
ENV_CONTROL=${ROLLOUT_BASELINE}/external_env_4gpu
ENV_READY=${ENV_CONTROL}/ready
EVAL_TAG=${EVAL_TAG:-success_epoch17_nimloth}
RESTART_ENV=${RESTART_ENV:-0}
ENV_NODE=${ENV_NODE:-dgx-46}
PORT_BASE=${PORT_BASE:-8400}

submit_env() {
  local node=$1
  local port=$2
  $SLURM --nodelist="${node}" --export=ALL,PORT_BASE="${port}" \
    "${ROOT}/sft1_env_vagen79_4gpu.slurm"
}

echo "ENV_NODE=${ENV_NODE} PORT_BASE=${PORT_BASE}"

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

echo "=== Submit SFT epoch17 Nimloth eval at $(date) ==="
echo "SFT model: ${EPOCH17}"
echo "EVAL_TAG=${EVAL_TAG}"
echo "RESTART_ENV=${RESTART_ENV}"
echo "Prompt: prompt_format=nimloth (greedy decode unchanged)"
echo "Baseline: pre-SFT VAGEN rollout global_step_79 (${ROLLOUT_BASELINE})"

if [ ! -f "${BASELINE_SUMMARY}" ]; then
  python3 "${ROOT}/summarize_sft1_eval_rollouts.py" \
    --root "${ROLLOUT_BASELINE}" \
    --step "${BASELINE_JSONL_STEP}" > "${BASELINE_SUMMARY}"
  echo "Wrote baseline summary: ${BASELINE_SUMMARY}"
fi

if [ ! -f "${EPOCH17}/model-00001-of-00002.safetensors" ]; then
  echo "ERROR missing epoch_017 checkpoint"
  exit 2
fi

if [ "${RESTART_ENV}" = "1" ]; then
  echo "Restarting external env (drop stale ready, cancel prior env jobs)"
  rm -f "${ENV_READY}" "${ENV_CONTROL}/failed"
  # Best-effort cancel of recent env jobs for this experiment.
  for j in $($SLURM/squeue -u "${USER:-csejzhang}" -h -o "%i %j" 2>/dev/null | awk '/sft1-env-v79|sft1-eval-v79-success_epoch17_nimloth_r/ {print $1}'); do
    echo "scancel job ${j}"
    $SCANCEL "${j}" 2>/dev/null || true
  done
  J_ENV=$(submit_env "${ENV_NODE}" "${PORT_BASE}" | awk '{print $NF}')
  echo "new env job: ${J_ENV} (node=${ENV_NODE} PORT_BASE=${PORT_BASE})"
  ENV_DEP="after:${J_ENV}"
elif [ -f "${ENV_READY}" ]; then
  echo "Reusing env ready: ${ENV_READY}"
  ENV_DEP="none"
elif [ -n "${ENV_JOB:-}" ]; then
  echo "Waiting for env job ${ENV_JOB}"
  ENV_DEP="after:${ENV_JOB}"
else
  J_ENV=$(submit_env "${ENV_NODE}" "${PORT_BASE}" | awk '{print $NF}')
  echo "env job: ${J_ENV} (node=${ENV_NODE})"
  ENV_DEP="after:${J_ENV}"
fi

J_E17=$(submit_eval "${EVAL_TAG}" "${EPOCH17}" "${ENV_DEP}" | awk '{print $NF}')
echo "eval job: ${J_E17}"

echo
echo "Results:"
echo "  ${ROOT}/runs/sft1_eval_vagen79_${EVAL_TAG}/summary_0.json"
echo
echo "Compare vs pre-SFT VAGEN rollout (step 79):"
echo "  python3 ${ROOT}/compare_sft1_eval_summaries.py \\"
echo "    ${ROOT}/runs/sft1_eval_vagen79_${EVAL_TAG}/summary_0.json \\"
echo "    ${BASELINE_SUMMARY}"
