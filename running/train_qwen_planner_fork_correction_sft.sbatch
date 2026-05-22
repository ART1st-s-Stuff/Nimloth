#!/usr/bin/env bash
#SBATCH --job-name=qwen_fork_sft
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

: "${FORK_JSONL:?Set FORK_JSONL to fork_samples.jsonl paths or globs.}"
: "${INIT_ADAPTER:=models/qwen_planner_lora_preaction_leakfree_8192_step00008000}"
: "${RUN_ROOT:=outputs/superpod/qwen_planner_fork_correction_$(date +%Y%m%d_%H%M%S)}"
: "${MIN_EFFECTIVE_LR_SCALE:=0.05}"
: "${MIN_OUTCOME_GAP:=0.02}"
: "${EPOCHS:=1}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=8}"
: "${LR:=5e-6}"
: "${LIMIT:=0}"

mkdir -p "$RUN_ROOT"

python dev/build_eb_nav_fork_qwen_correction_sft.py \
  --fork-jsonl ${FORK_JSONL} \
  --output-dir "$RUN_ROOT/sft" \
  --min-effective-lr-scale "$MIN_EFFECTIVE_LR_SCALE" \
  --min-outcome-gap "$MIN_OUTCOME_GAP"

python src/train/train_qwen_planner_lora.py \
  --sft-jsonl "$RUN_ROOT/sft/qwen_planner_fork_correction_train.jsonl" \
  --validation-jsonl "$RUN_ROOT/sft/qwen_planner_fork_correction_test.jsonl" \
  --init-adapter-path "$INIT_ADAPTER" \
  --output-dir "$RUN_ROOT/adapter" \
  --checkpoint-dir "$RUN_ROOT/checkpoints" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --lr "$LR" \
  --limit "$LIMIT" \
  --min-sample-weight "$MIN_EFFECTIVE_LR_SCALE" \
  --save-every-steps "${SAVE_EVERY_STEPS:-256}" \
  --validate-every-steps "${VALIDATE_EVERY_STEPS:-256}" \
  --validation-limit "${VALIDATION_LIMIT:-64}" \
  --validation-max-new-tokens "${VALIDATION_MAX_NEW_TOKENS:-256}" \
  --validation-extract-latent \
  --no-progress
