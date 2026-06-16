#!/bin/bash
#SBATCH --job-name=sft1-train-v48
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodelist=dgx-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
CHECKPOINT_STEP=48
INIT_HF=${BASE_RUN_DIR}/checkpoints/global_step_${CHECKPOINT_STEP}/actor/huggingface
ROLLOUT_ROOT=${SCRIPTDIR}/runs/sft1_rollouts_vagen48_ws2_2node_externalenv/validation
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen48_nimloth_format
TRAIN_OUT=${SCRIPTDIR}/runs/sft1_train_vagen48_qwen25vl
LOG=${TRAIN_OUT}/sft1_train_vagen48.log
NGPUS=4

mkdir -p "${TRAIN_OUT}" "$(dirname "${LOG}")"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; mkdir -p "$HOME" /project/peilab/atst/nimloth/.cache/torch/kernels; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

{
  echo "=== SFT1 step48 train/val starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Purpose: sft1_exp.md step 2+3 on VAGEN global_step_${CHECKPOINT_STEP} rollouts (train/val only)."
  echo "Rollout input: ${ROLLOUT_ROOT}"
  echo "Converted records: ${RECORDS_ROOT}"
  echo "Init HF checkpoint: ${INIT_HF}"
  echo "Train output: ${TRAIN_OUT}"
  echo "Train policy: train_success.jsonl only; val_all.jsonl each epoch."
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ ! -d "${INIT_HF}" ]; then
  echo "ERROR missing init HF checkpoint ${INIT_HF}" | tee -a "${LOG}"
  exit 3
fi

if [ ! -f "${RECORDS_ROOT}/train_success.jsonl" ] || [ ! -f "${RECORDS_ROOT}/val_all.jsonl" ]; then
  echo "=== Converting rollouts to Nimloth SFT records (train/val only) ===" | tee -a "${LOG}"
  PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/convert_sft1_rollouts_to_nimloth.py" \
    --input-root "${ROLLOUT_ROOT}" \
    --output-root "${RECORDS_ROOT}" \
    --checkpoint-hf "${INIT_HF}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --splits train val \
    --force 2>&1 | tee -a "${LOG}"
else
  echo "=== Reusing existing converted records at ${RECORDS_ROOT} ===" | tee -a "${LOG}"
fi

if [ ! -s "${RECORDS_ROOT}/train_success.jsonl" ]; then
  echo "ERROR empty train_success.jsonl after conversion" | tee -a "${LOG}"
  exit 4
fi
if [ ! -s "${RECORDS_ROOT}/val_all.jsonl" ]; then
  echo "ERROR empty val_all.jsonl after conversion" | tee -a "${LOG}"
  exit 5
fi

MODEL_PATH="${INIT_HF}"
RESUME_FLAG=()
if [ -f "${TRAIN_OUT}/best/training_state.pt" ]; then
  MODEL_PATH="${TRAIN_OUT}/best"
  RESUME_FLAG=(--resume)
  echo "=== Resuming from existing checkpoint in ${TRAIN_OUT}/best ===" | tee -a "${LOG}"
fi

echo "=== Starting Qwen2.5-VL SFT with ${NGPUS} GPUs model=${MODEL_PATH} ===" | tee -a "${LOG}"
PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
  "${SCRIPTDIR}/train_sft1_qwen25vl.py" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_success.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --epochs 20 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-6 \
  --max-length 20000 \
  "${RESUME_FLAG[@]}" 2>&1 | tee -a "${LOG}"

echo "=== SFT1 step48 training finished at $(date) ===" | tee -a "${LOG}"
