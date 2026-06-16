#!/bin/bash
#SBATCH --job-name=sft2-e2e-v79
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen79_nimloth_format
SFT1_BEST=${SCRIPTDIR}/runs/sft1_train_vagen79_qwen25vl/best
TRAIN_OUT=${SCRIPTDIR}/runs/sft2_e2e_vagen79_qwen25vl
LOG=${TRAIN_OUT}/sft2_e2e_vagen79_${SLURM_JOB_ID:-local}.log
NGPUS=4

mkdir -p "${TRAIN_OUT}" "${ROOT}/outputs/experiments/navigation_baseline/slurm"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1; export PYTHONPATH=/project/peilab/atst/nimloth/src'

{
  echo "=== SFT2 end-to-end vagen79 starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Parallel: 1 node x ${NGPUS} GPU DDP"
  echo "Init model (SFT1 best): ${SFT1_BEST}"
  echo "Train jsonl: ${RECORDS_ROOT}/train_success.jsonl"
  echo "Val jsonl: ${RECORDS_ROOT}/val_all.jsonl"
  echo "Output: ${TRAIN_OUT}"
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ ! -f "${SFT1_BEST}/config.json" ]; then
  echo "ERROR: missing SFT1 best checkpoint at ${SFT1_BEST}" | tee -a "${LOG}"
  exit 1
fi
if [ ! -s "${RECORDS_ROOT}/train_success.jsonl" ]; then
  echo "ERROR: missing train_success.jsonl" | tee -a "${LOG}"
  exit 1
fi

echo "=== torchrun ${NGPUS}-GPU DDP end-to-end SFT2 ===" | tee -a "${LOG}"
PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
  "${SCRIPTDIR}/train_sft2_qwen25vl.py" \
  --end-to-end \
  --model "${SFT1_BEST}" \
  --train-jsonl "${RECORDS_ROOT}/train_success.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr-qwen-start 1e-8 \
  --lr-qwen-peak 5e-7 \
  --qwen-lr-warmup-ratio 0.15 \
  --state-proj-lr 1e-4 \
  --lewm-lr 3e-4 \
  --lambda-ce 1.0 \
  --lambda-lewm 1.0 \
  --lambda-wm-start 0.1 \
  --lambda-wm-end 1.0 \
  --max-length 20000 \
  --img-size 96 \
  --emb-dim 128 \
  --gradient-checkpointing \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "=== SFT2 end-to-end finished at $(date) ===" | tee -a "${LOG}"
