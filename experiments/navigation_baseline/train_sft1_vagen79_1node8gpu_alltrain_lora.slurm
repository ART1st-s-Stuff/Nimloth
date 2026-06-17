#!/bin/bash
#SBATCH --job-name=sft1-8g-lora
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
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

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
INIT_HF=${BASE_RUN_DIR}/checkpoints/global_step_79/actor/huggingface
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen79_nimloth_format
TRAIN_OUT=${SCRIPTDIR}/runs/sft1_train_vagen79_qwen25vl_alltrain_8gpu_lora
LOG=${TRAIN_OUT}/sft1_train_alltrain_8gpu_lora.log
NGPUS=8
WANDB_RUN_NAME=${WANDB_RUN_NAME:-sft1-alltrain-8gpu-lora}

mkdir -p "${TRAIN_OUT}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export WANDB_DIR=/project/peilab/atst/nimloth/.cache/wandb; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; mkdir -p "$HOME" "$WANDB_DIR"; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

{
  echo "=== SFT1 alltrain 8GPU LoRA starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Model: Qwen2.5-VL-3B step79 | LoRA (not full fine-tune)"
  echo "Train: ${RECORDS_ROOT}/train_all.jsonl"
  echo "Output: ${TRAIN_OUT}"
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ ! -s "${RECORDS_ROOT}/train_all.jsonl" ]; then
  echo "ERROR missing train_all.jsonl" | tee -a "${LOG}"
  exit 1
fi

PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
  "${SCRIPTDIR}/train_sft1_qwen25vl.py" \
  --model "${INIT_HF}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --epochs 20 \
  --batch-size 1 \
  --grad-accum 8 \
  --lora \
  --lora-r 64 \
  --lora-alpha 128 \
  --lr 2e-4 \
  --embedding-lr 5e-4 \
  --max-length 20000 \
  --max-pixels 602112 \
  --format-eval-samples 32 \
  --num-workers 8 \
  --preprocess-workers 16 \
  --prefetch-factor 2 \
  --resume \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "TRAIN_DONE" > "${TRAIN_OUT}/train_done.flag"
echo "=== SFT1 alltrain 8GPU LoRA finished at $(date) ===" | tee -a "${LOG}"
