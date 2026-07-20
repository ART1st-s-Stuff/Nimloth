#!/usr/bin/env bash
#SBATCH --job-name=sft2-grid-qwen-cache
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail
: "${REPO:?}" "${SFT1_CHECKPOINT:?}" "${TRAIN_JSONL:?}" "${VAL_JSONL:?}" "${CACHE_ROOT:?}"
ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
mkdir -p "${CACHE_ROOT}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
cat >"${CACHE_ROOT}/README.md" <<EOF
# k16 compact Qwen cache for decoded-DINO grid SFT2

- Status: BUILDING_QWEN_CACHE
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- SFT1 checkpoint: ${SFT1_CHECKPOINT}
- Train: ${TRAIN_JSONL}; all 3217 trajectories
- Val: ${VAL_JSONL}; all 355 trajectories
- Format: compact deduplicated BF16 image tensors, k16/inject, max_pixels=100352
- Terminal next RGB paths are included for the DINO 4x4 sidecar.
EOF
cd "${REPO}"
"${PY}" experiments/training/sft2/build_grid_preprocess_cache.py \
  --sft1-checkpoint "${SFT1_CHECKPOINT}" \
  --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --cache-root "${CACHE_ROOT}" --workers 8 --max-length 12000 --max-pixels 100352
printf '\n- Qwen cache status: COMPLETE\n' >>"${CACHE_ROOT}/README.md"
touch "${CACHE_ROOT}/qwen_cache_done.flag"
