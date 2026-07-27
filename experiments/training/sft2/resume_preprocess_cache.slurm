#!/bin/bash
#SBATCH --job-name=sft2-cache-resume
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:?set REPO to the clean server worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
CONFIG=${CONFIG:?set CONFIG}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
TRAIN_JSONL=${TRAIN_JSONL:?set TRAIN_JSONL}
VAL_JSONL=${VAL_JSONL:?set VAL_JSONL}
PREPROCESS_CACHE_DIR=${PREPROCESS_CACHE_DIR:?set PREPROCESS_CACHE_DIR}
CACHE_METADATA_DIR=${CACHE_METADATA_DIR:?set CACHE_METADATA_DIR}
CACHE_LOG=${CACHE_LOG:?set CACHE_LOG beside the owning experiment output}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
CACHE_MODE=${CACHE_MODE:-resume}
MAX_TRAIN_RECORDS=${MAX_TRAIN_RECORDS:--1}
MAX_VAL_RECORDS=${MAX_VAL_RECORDS:--1}

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ERROR commit mismatch: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR remote worktree has tracked changes: ${REPO}" >&2
  exit 1
fi
for required in "${CONFIG}" "${TRAIN_JSONL}" "${VAL_JSONL}" "${MODEL_PATH}/config.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "ERROR required input missing: ${required}" >&2
    exit 1
  fi
done
case "${CACHE_MODE}" in
  resume)
    if [[ ! -s "${PREPROCESS_CACHE_DIR}/train/build_state.json" ]]; then
      echo "ERROR resumable train cache build_state missing" >&2
      exit 1
    fi
    ;;
  fresh)
    if [[ -e "${PREPROCESS_CACHE_DIR}" ]]; then
      echo "ERROR fresh cache target already exists: ${PREPROCESS_CACHE_DIR}" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR CACHE_MODE must be fresh or resume, got ${CACHE_MODE}" >&2
    exit 1
    ;;
esac

RECORD_ARGS=()
if (( MAX_TRAIN_RECORDS > 0 )); then
  RECORD_ARGS+=(--max-train-records "${MAX_TRAIN_RECORDS}")
fi
if (( MAX_VAL_RECORDS > 0 )); then
  RECORD_ARGS+=(--max-val-records "${MAX_VAL_RECORDS}")
fi

mkdir -p "${CACHE_METADATA_DIR}" "$(dirname "${CACHE_LOG}")"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache/xdg
export VIRTUAL_ENV=${PYTHON_ENV}
export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:${PATH}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
export TOKENIZERS_PARALLELISM=false
mkdir -p "${XDG_CACHE_HOME}"

{
  echo "cache_build_start=$(date --iso-8601=seconds)"
  echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
  echo "commit=${actual_commit}"
  echo "config=${CONFIG}"
  echo "train_jsonl=${TRAIN_JSONL}"
  echo "val_jsonl=${VAL_JSONL}"
  echo "cache=${PREPROCESS_CACHE_DIR}"
  echo "cache_mode=${CACHE_MODE}"
  echo "max_train_records=${MAX_TRAIN_RECORDS} max_val_records=${MAX_VAL_RECORDS}"
  if [[ "${CACHE_MODE}" == "resume" ]]; then
    echo "initial_train_image_shards=$(find "${PREPROCESS_CACHE_DIR}/train/images" -maxdepth 1 -name 'shard_*.pt' -type f | wc -l)"
  fi
} | tee -a "${CACHE_LOG}"

"${PYTHON_ENV}/bin/python3" \
  "${REPO}/experiments/training/sft2/build_preprocess_cache.py" \
  --config "${CONFIG}" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${CACHE_METADATA_DIR}" \
  --preprocess-cache-dir "${PREPROCESS_CACHE_DIR}" \
  --preprocess-cache-processor-source "${MODEL_PATH}" \
  --preprocess-workers "${SLURM_CPUS_PER_TASK}" \
  "${RECORD_ARGS[@]}" \
  2>&1 | tee -a "${CACHE_LOG}"

for manifest in \
  "${PREPROCESS_CACHE_DIR}/train/manifest.json" \
  "${PREPROCESS_CACHE_DIR}/val/manifest.json"; do
  if [[ ! -s "${manifest}" ]]; then
    echo "ERROR completed cache is missing ${manifest}" >&2
    exit 1
  fi
done
touch "${PREPROCESS_CACHE_DIR}/cache_done.flag"
echo "cache_build_done=$(date --iso-8601=seconds)" | tee -a "${CACHE_LOG}"
