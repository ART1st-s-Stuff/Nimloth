#!/bin/bash
#SBATCH --job-name=sft1-train-8g
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --requeue

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft1
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"

# SFT1_TUNE_MODE: lora | embedlr
SFT1_TUNE_MODE=${SFT1_TUNE_MODE:-lora}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft1_train_${SFT1_TUNE_MODE}}
TRAIN_OUT=${TRAIN_OUT:-${REPO}/outputs/experiments/training/sft1/${RUN_DATE}/${EXPERIMENT_NAME}}
LOG=${TRAIN_OUT}/sft1_train_${SFT1_TUNE_MODE}.log
NGPUS=${NGPUS:-8}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft1-${SFT1_TUNE_MODE}-8gpu}
EPOCHS=${EPOCHS:-20}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-20000}

mkdir -p "${TRAIN_OUT}" "${REPO}/outputs/experiments/training/sft1/slurm"

{
  echo "=== SFT1 8GPU train (${SFT1_TUNE_MODE}) starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Init HF: ${INIT_HF}"
  echo "Train: ${RECORDS_ROOT}/train_all.jsonl"
  echo "Output: ${TRAIN_OUT}"
} | tee -a "${LOG}"

if [ ! -s "${RECORDS_ROOT}/train_all.jsonl" ]; then
  echo "ERROR missing train_all.jsonl" | tee -a "${LOG}"
  exit 1
fi

TRAIN_ARGS=(
  --model "${INIT_HF}"
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl"
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl"
  --output-dir "${TRAIN_OUT}"
  --epochs "${EPOCHS}"
  --batch-size 1
  --grad-accum "${GRAD_ACCUM}"
  --max-length "${MAX_LENGTH}"
  --max-pixels 602112
  --format-eval-samples 32
  --num-workers 8
  --preprocess-workers 16
  --prefetch-factor 2
  --resume
  --wandb-run-name "${WANDB_RUN_NAME}"
  --seed 42
)

case "${SFT1_TUNE_MODE}" in
  lora)
    TRAIN_ARGS+=(
      --lora --lora-r 64 --lora-alpha 128
      --lr 2e-4 --embedding-lr 5e-4
      --gradient-checkpointing
    )
    ;;
  embedlr)
    TRAIN_ARGS+=(--lr 1e-6 --embedding-lr 5e-6)
    ;;
  *)
    echo "ERROR unknown SFT1_TUNE_MODE=${SFT1_TUNE_MODE}" | tee -a "${LOG}"
    exit 2
    ;;
esac

PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
  "${SCRIPTDIR}/train.py" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"

echo "TRAIN_DONE" > "${TRAIN_OUT}/train_done.flag"
echo "=== SFT1 8GPU train finished at $(date) ===" | tee -a "${LOG}"
