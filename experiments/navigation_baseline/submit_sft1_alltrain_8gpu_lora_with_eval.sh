#!/bin/bash
# LoRA SFT on train_all (3240) | 8GPU train | env/eval on separate nodes.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

TRAIN_OUT=${ROOT}/runs/sft1_train_vagen79_qwen25vl_alltrain_8gpu_lora
EVAL_MODEL=${TRAIN_OUT}/best/hf_merged
ROLLOUT_ENV_RUN=${ROOT}/runs/sft1_rollouts_vagen79_greedy_parallel
ENV_CONTROL=${ROLLOUT_ENV_RUN}/external_env_4gpu
ENV_READY=${ENV_CONTROL}/ready
EVAL_TAG=${EVAL_TAG:-alltrain_8gpu_lora_best}
TRAIN_NODE=${TRAIN_NODE:-dgx-52}
ENV_NODE=${ENV_NODE:-dgx-13}
EVAL_NODE=${EVAL_NODE:-dgx-09}
PORT_BASE=${PORT_BASE:-8500}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft1-alltrain-8gpu-lora}

submit_env() {
  $SLURM --nodelist="${ENV_NODE}" --export=ALL,PORT_BASE="${PORT_BASE}" \
    "${ROOT}/sft1_env_vagen79_4gpu.slurm"
}

submit_eval() {
  local dep=$1
  $SLURM --dependency="afterok:${dep}" --nodelist="${EVAL_NODE}" \
    --job-name="sft1-eval-${EVAL_TAG}" \
    --export=ALL,EVAL_TAG="${EVAL_TAG}",MODEL_PATH="${EVAL_MODEL}" \
    "${ROOT}/sft1_eval_vagen79_greedy_valtest.slurm"
}

echo "=== Submit alltrain 8GPU LoRA SFT + eval at $(date) ==="
echo "Train: ${TRAIN_NODE} | LoRA | train_all.jsonl | Qwen2.5-VL-3B step79"
echo "Env:   ${ENV_NODE} | Eval: ${EVAL_NODE} (model ${EVAL_MODEL})"

rm -f "${ENV_CONTROL}/failed"
# Keep ENV_READY if a healthy env is already serving evals.
if [ -f "${ENV_READY}" ] && [ -s "${ENV_CONTROL}/env_urls.txt" ]; then
  echo "keeping existing env ready marker"
else
  rm -f "${ENV_READY}"
fi
for j in $($SLURM/squeue -u "${USER:-csejzhang}" -h -o "%i %j" 2>/dev/null | awk '/sft1-env-v79/ {print $1}'); do
  : # do not cancel running env on train submit
done

J_TRAIN=$($SLURM --nodelist="${TRAIN_NODE}" \
  --export=ALL,WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  "${ROOT}/train_sft1_vagen79_1node8gpu_alltrain_lora.slurm" | awk '{print $NF}')
echo "train job: ${J_TRAIN}"

J_ENV=$(submit_env | awk '{print $NF}')
echo "env job:   ${J_ENV}"

echo
echo "Per-epoch eval: run submit_sft1_lora_ckpt_eval_watcher.sh (not end-of-train eval)"
