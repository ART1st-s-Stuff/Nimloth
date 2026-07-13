#!/bin/bash
#SBATCH --job-name=sft2-cache-compact
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU partition QOS currently permits at most 8 CPUs per node.
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue

set -euo pipefail

: "${PREPROCESS_CACHE_DIR:?PREPROCESS_CACHE_DIR required}"
: "${SFT1_RUN:?SFT1_RUN required}"
: "${BASE_HF:?BASE_HF required}"
: "${RECORDS_ROOT:?RECORDS_ROOT required}"

REPO=${REPO:-/project/peilab/atst/nimloth}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
SCRIPTDIR=${REPO}/experiments/training/sft2
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/latent_wm_value_k8.yaml}
LATENT_TOKEN_COUNT=${LATENT_TOKEN_COUNT:-8}
LATENT_QUERY_MODE=${LATENT_QUERY_MODE:-}
PREPROCESS_WORKERS=${PREPROCESS_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
CACHE_LOG=${PREPROCESS_CACHE_DIR}/build_compact_cache.log
CACHE_BUILD_OUTPUT_DIR=${CACHE_BUILD_OUTPUT_DIR:-${PREPROCESS_CACHE_DIR}/build_metadata}

mkdir -p "${PREPROCESS_CACHE_DIR}" "${CACHE_BUILD_OUTPUT_DIR}" "${REPO}/outputs/experiments/training/sft2/slurm"

export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:$PATH
export VIRTUAL_ENV=${PYTHON_ENV}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=false
hash -r
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi
# The credential file may contain stale runtime variables.
export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:$PATH
export VIRTUAL_ENV=${PYTHON_ENV}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
hash -r

{
  echo "=== compact SFT2 cache build starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Git commit: $(git -C "${REPO}" rev-parse HEAD)"
  echo "Config: ${CONFIG}"
  echo "Records: ${RECORDS_ROOT}"
  echo "Cache: ${PREPROCESS_CACHE_DIR}"
  echo "Format: compact image_dtype=${PREPROCESS_CACHE_IMAGE_DTYPE:-bfloat16}"
  echo "Workers: ${PREPROCESS_WORKERS}"
} | tee -a "${CACHE_LOG}"

export SFT1_RUN BASE_HF
bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh" | tee -a "${CACHE_LOG}"
# shellcheck disable=SC1090
source "${SFT1_RUN}/.sft2_init_env"

LATENT_ARGS=(--latent-token-count "${LATENT_TOKEN_COUNT}")
if [ -n "${LATENT_QUERY_MODE}" ]; then
  LATENT_ARGS+=(--latent-query-mode "${LATENT_QUERY_MODE}")
fi

python "${SCRIPTDIR}/build_preprocess_cache.py" \
  --config "${CONFIG}" \
  --model "${SFT1_BEST}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${CACHE_BUILD_OUTPUT_DIR}" \
  --preprocess-cache-dir "${PREPROCESS_CACHE_DIR}" \
  --preprocess-cache-format compact \
  --preprocess-cache-image-dtype "${PREPROCESS_CACHE_IMAGE_DTYPE:-bfloat16}" \
  --preprocess-cache-image-shard-size "${PREPROCESS_CACHE_IMAGE_SHARD_SIZE:-128}" \
  --preprocess-cache-transition-shard-size "${PREPROCESS_CACHE_TRANSITION_SHARD_SIZE:-256}" \
  --preprocess-workers "${PREPROCESS_WORKERS}" \
  "${LATENT_ARGS[@]}" \
  2>&1 | tee -a "${CACHE_LOG}"

touch "${PREPROCESS_CACHE_DIR}/cache_done.flag"
echo "=== compact SFT2 cache build finished at $(date) ===" | tee -a "${CACHE_LOG}"
