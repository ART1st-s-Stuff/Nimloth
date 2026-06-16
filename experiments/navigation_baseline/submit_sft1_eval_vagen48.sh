#!/bin/bash
# Submit HF->ws2 conversion + val/test eval for SFT epoch12 and final checkpoints.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
TRAIN_ROOT=${ROOT}/runs/sft1_train_vagen48_qwen25vl

submit_convert() {
  local tag=$1
  local hf_path=$2
  local run_name=$3
  $SLURM --job-name="sft1-ws2-${tag}" \
    --export=ALL,SOURCE_HF_PATH="${hf_path}",TARGET_RUN_NAME="${run_name}",CONVERT_STEP=1 \
    "${ROOT}/convert_sft1_hf_to_world_size2.slurm"
}

submit_eval() {
  local tag=$1
  local ckpt_dir=$2
  local dep=$3
  $SLURM --dependency=afterok:${dep} --job-name="sft1-eval-${tag}" \
    --export=ALL,EVAL_TAG="${tag}",CHECKPOINT_DIR="${ckpt_dir}",CHECKPOINT_STEP=1 \
    "${ROOT}/sft1_eval_one_model_valtest.slurm"
}

echo "=== Submit SFT eval pipeline at $(date) ==="

J_EVAL_E12=$(submit_eval epoch12 "${ROOT}/runs/sft1_eval_ws2_epoch012/checkpoints" "454050" | awk '{print $NF}')
echo "eval epoch12 job: ${J_EVAL_E12}"

J_CONV_FINAL=$(submit_convert final "${TRAIN_ROOT}/final" sft1_eval_ws2_final | awk '{print $NF}')
echo "convert final job: ${J_CONV_FINAL}"

J_EVAL_FINAL=$(submit_eval final "${ROOT}/runs/sft1_eval_ws2_final/checkpoints" "${J_CONV_FINAL}" | awk '{print $NF}')
echo "eval final job: ${J_EVAL_FINAL}"

echo "baseline val/test already in runs/sft1_rollouts_vagen48_ws2_2node_externalenv (step 48)"
