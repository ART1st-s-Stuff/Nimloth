#!/bin/bash
#SBATCH --job-name=rcdm-gt-compress
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/exp-latent-repr-ablation}
PY_VENV=${PY_VENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
PYTHON_BIN=${PYTHON_BIN:-${PY_VENV}/bin/python}
DATE_TAG=${DATE_TAG:-$(date +%F)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/${DATE_TAG}}
RUN_NAME=${RUN_NAME:-rcdm_gt_compressed_rollout4_b128_gt_current_1k}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${SLURM_JOB_ID:-local}}
QWEN_CHECKPOINT=${QWEN_CHECKPOINT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_lejepa_align_fn1_dgx56/ckpt_step1000_preserved}
RECORDS_ROOT=${RECORDS_ROOT:-/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format}
TRAIN_JSONL=${TRAIN_JSONL:-${RECORDS_ROOT}/train_all.jsonl}
VAL_JSONL=${VAL_JSONL:-${RECORDS_ROOT}/val_all.jsonl}
VISION_TOKEN_CACHE_DIR=${VISION_TOKEN_CACHE_DIR:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/cache/qwen_vision_sft2step1000_vagen79_trainall_valall_k81d2048}
COMPRESSOR_CHECKPOINT=${COMPRESSOR_CHECKPOINT:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/2026-07-06/compressed_vision_rollout4_full_cache_b128_519cbb4_466404/compressor}
STATE_CACHE_DIR=${STATE_CACHE_DIR:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/cache/rcdm_gt_compressed_rollout4_b128_current}
MAX_TRAIN_RECORDS=${MAX_TRAIN_RECORDS:--1}
MAX_VAL_RECORDS=${MAX_VAL_RECORDS:-360}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
SAMPLE_NUM_ITEMS=${SAMPLE_NUM_ITEMS:-8}
SAMPLE_TIMESTEP_RESPACING=${SAMPLE_TIMESTEP_RESPACING:-100}

cd "${REPO}"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p "${OUT_DIR}" /project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm
LOG=${OUT_DIR}/train.log
export HOME=/project/peilab/atst/nimloth/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${REPO}/src:${REPO}/external/le-wm
export PATH=${PY_VENV}/bin:${PATH}

cat > "${OUT_DIR}/README.md" <<README
# RCDM GT compressed-token reconstruction

- purpose: train post-hoc RCDM only on GT compressed Qwen vision-token conditions, then sample reconstruction strips.
- git_commit: ${COMMIT}
- condition: flatten(AttentionTokenCompressor(Qwen vision tokens of current image)), K=16, D=512, cond_dim=8192.
- target: current image reconstruction from GT current compressed tokens.
- compressor_checkpoint: ${COMPRESSOR_CHECKPOINT}
- vision_token_cache_dir: ${VISION_TOKEN_CACHE_DIR}
- state_cache_dir: ${STATE_CACHE_DIR}
- train_jsonl: ${TRAIN_JSONL}
- val_jsonl: ${VAL_JSONL}
- split_semantics: train_all is training split; val_all is held-out validation split per records README/manifest.
- trainable: RCDM UNet only.
- frozen: Qwen vision tokens are cached; compressor is frozen for cache build.
- max_steps: ${MAX_STEPS}
- output_dir: ${OUT_DIR}
- wandb: disabled
README

{
  echo "=== RCDM GT compressed train $(date) ==="
  echo "job=${SLURM_JOB_ID:-local} node=${SLURM_JOB_NODELIST:-local} commit=${COMMIT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "out=${OUT_DIR}"
  echo "state_cache=${STATE_CACHE_DIR}"
  echo "compressor=${COMPRESSOR_CHECKPOINT}"
  echo "max_steps=${MAX_STEPS} train_batch=${TRAIN_BATCH_SIZE}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" -m nimloth.training.representation_ablation.build_rcdm_gt_cache \
  --jsonl "${TRAIN_JSONL}" \
  --vision-token-cache-dir "${VISION_TOKEN_CACHE_DIR}/train" \
  --compressor-checkpoint "${COMPRESSOR_CHECKPOINT}" \
  --cache-dir "${STATE_CACHE_DIR}/train" \
  --split-name train \
  --max-records "${MAX_TRAIN_RECORDS}" \
  --batch-size "${CACHE_BATCH_SIZE}" 2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" -m nimloth.training.representation_ablation.build_rcdm_gt_cache \
  --jsonl "${VAL_JSONL}" \
  --vision-token-cache-dir "${VISION_TOKEN_CACHE_DIR}/val" \
  --compressor-checkpoint "${COMPRESSOR_CHECKPOINT}" \
  --cache-dir "${STATE_CACHE_DIR}/val" \
  --split-name val \
  --max-records "${MAX_VAL_RECORDS}" \
  --batch-size "${CACHE_BATCH_SIZE}" 2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" -m nimloth.training.reconstruction.rcdm_sft2 \
  --model "${QWEN_CHECKPOINT}" \
  --state-proj-checkpoint "${QWEN_CHECKPOINT}/state_proj.pt" \
  --wm-checkpoint "${QWEN_CHECKPOINT}/wm_predictor" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUT_DIR}" \
  --state-cache-dir "${STATE_CACHE_DIR}" \
  --epochs 1 \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --lr 1.0e-4 \
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --save-interval "${SAVE_INTERVAL}" \
  --max-val-batches 8 \
  --no-wandb 2>&1 | tee -a "${LOG}"

CKPT=$(ls "${OUT_DIR}"/model_*.pt | sort | tail -1)
"${PYTHON_BIN}" -m nimloth.eval.rcdm_cache_reconstruction \
  --state-cache-dir "${STATE_CACHE_DIR}/val" \
  --rcdm-checkpoint "${CKPT}" \
  --metadata "${OUT_DIR}/metadata.json" \
  --output-dir "${OUT_DIR}/samples_gt_current" \
  --batch-size 4 \
  --num-items "${SAMPLE_NUM_ITEMS}" \
  --timestep-respacing "${SAMPLE_TIMESTEP_RESPACING}" 2>&1 | tee -a "${LOG}"

echo "=== done $(date) output=${OUT_DIR} checkpoint=${CKPT} ===" | tee -a "${LOG}"
