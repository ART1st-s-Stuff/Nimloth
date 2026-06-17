#!/bin/bash
# 8-GPU alltrain SFT (higher embedding lr, wandb dataset upload) + env/eval on separate nodes.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

TRAIN_OUT=${ROOT}/runs/sft1_train_vagen79_qwen25vl_alltrain_8gpu_embedlr
ROLLOUT_ENV_RUN=${ROOT}/runs/sft1_rollouts_vagen79_greedy_parallel
ENV_CONTROL=${ROLLOUT_ENV_RUN}/external_env_4gpu
ENV_READY=${ENV_CONTROL}/ready
EVAL_TAG=${EVAL_TAG:-alltrain_8gpu_embedlr_best}
TRAIN_NODE=${TRAIN_NODE:-dgx-52}
ENV_NODE=${ENV_NODE:-dgx-13}
EVAL_NODE=${EVAL_NODE:-dgx-09}
PORT_BASE=${PORT_BASE:-8500}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft1-alltrain-8gpu-embedlr}

submit_env() {
  $SLURM --nodelist="${ENV_NODE}" --export=ALL,PORT_BASE="${PORT_BASE}" \
    "${ROOT}/sft1_env_vagen79_4gpu.slurm"
}

submit_eval() {
  local dep=$1
  $SLURM --dependency="afterok:${dep}" --nodelist="${EVAL_NODE}" \
    --job-name="sft1-eval-${EVAL_TAG}" \
    --export=ALL,EVAL_TAG="${EVAL_TAG}",MODEL_PATH="${TRAIN_OUT}/best" \
    "${ROOT}/sft1_eval_vagen79_greedy_valtest.slurm"
}

echo "=== Submit alltrain 8GPU embed-lr SFT + cross-node eval at $(date) ==="
echo "Train node: ${TRAIN_NODE} (8 GPU)"
echo "Env node:   ${ENV_NODE} (4 GPU)"
echo "Eval node:  ${EVAL_NODE} (2 GPU, after train)"
echo "Train out:  ${TRAIN_OUT}"
echo "Eval tag:   ${EVAL_TAG}"
echo "WANDB run:  ${WANDB_RUN_NAME}"

echo "Restart external env (PORT_BASE=${PORT_BASE})"
rm -f "${ENV_READY}" "${ENV_CONTROL}/failed"
for j in $($SLURM/squeue -u "${USER:-csejzhang}" -h -o "%i %j" 2>/dev/null | awk '/sft1-env-v79/ {print $1}'); do
  echo "scancel env job ${j}"
  $SCANCEL "${j}" 2>/dev/null || true
done

J_TRAIN=$($SLURM --nodelist="${TRAIN_NODE}" \
  --export=ALL,WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  "${ROOT}/train_sft1_vagen79_1node8gpu_alltrain_embedlr.slurm" | awk '{print $NF}')
echo "train job: ${J_TRAIN}"

J_ENV=$(submit_env | awk '{print $NF}')
echo "env job:   ${J_ENV} (node=${ENV_NODE} PORT_BASE=${PORT_BASE})"

# Eval waits for train; env can warm up in parallel (eval slurm waits on ready file).
J_EVAL=$(submit_eval "${J_TRAIN}" | awk '{print $NF}')
echo "eval job:  ${J_EVAL} (node=${EVAL_NODE}, model=${TRAIN_OUT}/best)"

echo
echo "Logs:"
echo "  ${TRAIN_OUT}/sft1_train_alltrain_8gpu_embedlr.log"
echo "  ${ROOT}/runs/sft1_eval_vagen79_${EVAL_TAG}/"
echo "WandB: dataset artifact + metrics under run ${WANDB_RUN_NAME}"
