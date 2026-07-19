#!/bin/bash
#SBATCH --job-name=dino-cls-cache
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err

set -euo pipefail
REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-cache-perf}
SOURCE_CACHE=${SOURCE_CACHE:?SOURCE_CACHE is required}
DINO_CACHE_ROOT=${DINO_CACHE_ROOT:?DINO_CACHE_ROOT is required}
DINO_MODEL=${DINO_MODEL:-facebook/dinov2-large}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${DINO_CACHE_ROOT}" /project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm
for split in train val; do
  mkdir -p "${DINO_CACHE_ROOT}/${split}"
  ln -sfn "${SOURCE_CACHE}/${split}/manifest.json" "${DINO_CACHE_ROOT}/${split}/manifest.json"
  ln -sfn "${SOURCE_CACHE}/${split}/image_index.json" "${DINO_CACHE_ROOT}/${split}/image_index.json"
done
cd "${REPO}"
printf 'job=%s node=%s commit=%s source=%s output=%s model=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname -s)" "$(git rev-parse HEAD)" \
  "${SOURCE_CACHE}" "${DINO_CACHE_ROOT}" "${DINO_MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PY}" experiments/training/sft2/build_dino_feature_cache.py \
  --preprocess-cache-dir "${DINO_CACHE_ROOT}" \
  --dino-model "${DINO_MODEL}" --batch-size 32 --shard-size 1024 --verify-samples 8
"${PY}" experiments/training/sft2/benchmark_dino_feature_cache.py \
  --dino-cache-dir "${DINO_CACHE_ROOT}" \
  --dino-model "${DINO_MODEL}" --samples 256 --batch-sizes 1,2,4,8,16 --repeats 3
printf '%s\n' "$(date -Is)" > "${DINO_CACHE_ROOT}/done.flag"
