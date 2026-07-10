#!/bin/bash
#SBATCH --job-name=sft1-cache
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
SCRIPTDIR=${REPO}/experiments/training/sft1
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"

RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft1_train_lora}
TRAIN_OUT=${TRAIN_OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft1/${RUN_DATE}/${EXPERIMENT_NAME}}
TRAIN_JSONL=${TRAIN_JSONL:-${RECORDS_ROOT}/train_success.jsonl}
VAL_JSONL=${VAL_JSONL:-${RECORDS_ROOT}/val_all.jsonl}
CACHE_LOG=${TRAIN_OUT}/sft1_cache_build.log
LATENT_TOKEN_COUNT=${LATENT_TOKEN_COUNT:-8}
MASK_LATENT_QUERY_LABELS=${MASK_LATENT_QUERY_LABELS:-1}

mkdir -p "${TRAIN_OUT}" "${REPO}/outputs/experiments/training/sft1/slurm"
{
  echo "=== SFT1 preprocess cache build starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Git commit: $(git -C "${REPO}" rev-parse HEAD)"
  echo "Init HF: ${INIT_HF}"
  echo "Train: ${TRAIN_JSONL}"
  echo "Val: ${VAL_JSONL}"
  echo "Output/cache root: ${TRAIN_OUT}"
  echo "Pixel dtype: ${CACHE_PIXEL_DTYPE:-bfloat16}"
} | tee -a "${CACHE_LOG}"

CACHE_ARGS=(
  --model "${INIT_HF}"
  --train-jsonl "${TRAIN_JSONL}"
  --val-jsonl "${VAL_JSONL}"
  --output-dir "${TRAIN_OUT}"
  --cache-only
  --no-wandb
  --max-length "${MAX_LENGTH:-20000}"
  --max-pixels 602112
  --latent-token-count "${LATENT_TOKEN_COUNT}"
  --cache-pixel-dtype "${CACHE_PIXEL_DTYPE:-bfloat16}"
  --preprocess-workers "${PREPROCESS_WORKERS:-16}"
)
if [ "${MASK_LATENT_QUERY_LABELS}" = "0" ]; then
  CACHE_ARGS+=(--no-mask-latent-query-labels)
fi

"${PYTHON_ENV}/bin/python3" "${SCRIPTDIR}/train.py" "${CACHE_ARGS[@]}" 2>&1 | tee -a "${CACHE_LOG}"
touch "${TRAIN_OUT}/preprocess_cache_done.flag"
echo "=== SFT1 preprocess cache build finished at $(date) ===" | tee -a "${CACHE_LOG}"
