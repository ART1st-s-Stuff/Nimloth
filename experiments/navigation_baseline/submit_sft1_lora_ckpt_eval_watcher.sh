#!/bin/bash
# Per-epoch eval: watcher polls latest LoRA ckpt and sbatch eval on EVAL_NODE.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

# Cancel eval jobs that wait for full training to finish.
for j in $($SLURM/squeue -u "${USER:-csejzhang}" -h -o "%i %j" 2>/dev/null | awk '/sft1-eval-alltrain/ {print $1}'); do
  echo "scancel stale eval ${j}"
  $SCANCEL "${j}" 2>/dev/null || true
done

TRAIN_OUT=${TRAIN_OUT:-${ROOT}/runs/sft1_train_vagen79_qwen25vl_alltrain_8gpu_lora}
BASE_MODEL=${BASE_MODEL:-${ROOT}/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface}
EVAL_NODE=${EVAL_NODE:-dgx-12}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-alltrain_8gpu_lora}
TRAIN_JOB_ID=${TRAIN_JOB_ID:-}
WATCH_NODE=${WATCH_NODE:-}
WATCH_PARTITION=${WATCH_PARTITION:-cpu}

NODELIST_ARGS=()
if [[ -n "${WATCH_NODE}" ]]; then
  NODELIST_ARGS=(--nodelist="${WATCH_NODE}")
fi

J=$($SLURM --account=peilab --partition="${WATCH_PARTITION}" "${NODELIST_ARGS[@]}" \
  --export=ALL,TRAIN_OUT="${TRAIN_OUT}",BASE_MODEL="${BASE_MODEL}",EVAL_NODE="${EVAL_NODE}",EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX}",TRAIN_JOB_ID="${TRAIN_JOB_ID}" \
  "${ROOT}/sft1_lora_ckpt_eval_watcher.slurm" | awk '{print $NF}')

echo "ckpt-eval-watcher job: ${J}"
echo "  watches: ${TRAIN_OUT}/epoch_*"
echo "  eval node: ${EVAL_NODE}"
echo "  train job: ${TRAIN_JOB_ID:-auto-detect}"
