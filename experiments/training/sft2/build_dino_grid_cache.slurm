#!/usr/bin/env bash
#SBATCH --job-name=sft2-dino-grid-cache
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail
: "${REPO:?}" "${CACHE_ROOT:?}"
test -f "${CACHE_ROOT}/qwen_cache_done.flag"
ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
cd "${REPO}"
srun "${PY}" experiments/training/sft2/build_dino_grid_cache.py \
  --cache-root "${CACHE_ROOT}" \
  --dino-model facebook/dinov2-large \
  --grid-size 4 --batch-size 32 --shard-size 256 \
  | tee "${CACHE_ROOT}/build_dino_grid_cache.log"
printf '\n- DINO grid cache status: COMPLETE; online/cache bitwise gate passed\n' >>"${CACHE_ROOT}/README.md"
touch "${CACHE_ROOT}/dino_grid_cache_done.flag"
