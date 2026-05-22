#!/usr/bin/env bash
#SBATCH --job-name=qwen_plan_hist_sft
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=dgx-16
#SBATCH --requeue
#SBATCH --output=/project/peilab/atst/flower/running/slurm_logs/%x-%j.out
#SBATCH --error=/project/peilab/atst/flower/running/slurm_logs/%x-%j.err

set -euo pipefail

cd /project/peilab/atst/flower

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a
  source /project/peilab/atst/.env
  set +a
fi

. .venv/bin/activate
export PYTHONUNBUFFERED=1
export HOME=/project/peilab/atst
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch

RUN_ROOT=${RUN_ROOT:-outputs/superpod/qwen_planner_success_history_continue_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN_ROOT"

python src/train/train_qwen_planner_lora.py \
  --sft-jsonl outputs/superpod/qwen_planner_success_history_sft_20260522/qwen_planner_success_history_train.jsonl \
  --validation-jsonl outputs/superpod/qwen_planner_success_history_sft_20260522/qwen_planner_success_history_test.jsonl \
  --init-adapter-path models/qwen_planner_lora_preaction_leakfree_8192_step00008000 \
  --output-dir "$RUN_ROOT/adapter" \
  --checkpoint-dir "$RUN_ROOT/checkpoints" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --lr "${LR:-1e-5}" \
  --limit "${LIMIT:-2048}" \
  --save-every-steps "${SAVE_EVERY_STEPS:-256}" \
  --validate-every-steps "${VALIDATE_EVERY_STEPS:-256}" \
  --validation-limit "${VALIDATION_LIMIT:-64}" \
  --validation-max-new-tokens "${VALIDATION_MAX_NEW_TOKENS:-256}" \
  --validation-extract-latent \
  --no-progress
