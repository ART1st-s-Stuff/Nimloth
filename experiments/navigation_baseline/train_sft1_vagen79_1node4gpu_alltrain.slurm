#!/bin/bash
#SBATCH --job-name=sft1-train-v79-all
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
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
ROLLOUT_STEP=0
INIT_HF=${BASE_RUN_DIR}/checkpoints/global_step_79/actor/huggingface
ROLLOUT_ROOT=${SCRIPTDIR}/runs/sft1_rollouts_vagen79_greedy_parallel/validation
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen79_nimloth_format
TRAIN_OUT=${SCRIPTDIR}/runs/sft1_train_vagen79_qwen25vl_alltrain
LOG=${TRAIN_OUT}/sft1_train_vagen79_alltrain.log
NGPUS=4

mkdir -p "${TRAIN_OUT}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

{
  echo "=== SFT1 vagen79 all-train starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Parallel: 1 node x ${NGPUS} GPU DDP"
  echo "Records: ${RECORDS_ROOT} (train_all + val_all)"
  echo "Init HF: ${INIT_HF}"
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ ! -s "${RECORDS_ROOT}/train_all.jsonl" ]; then
  PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/convert_sft1_rollouts_to_nimloth.py" \
    --input-root "${ROLLOUT_ROOT}" \
    --output-root "${RECORDS_ROOT}" \
    --checkpoint-hf "${INIT_HF}" \
    --checkpoint-step "${ROLLOUT_STEP}" \
    --splits train val test \
    --force 2>&1 | tee -a "${LOG}"
fi

if [ ! -s "${RECORDS_ROOT}/train_all.jsonl" ]; then
  echo "ERROR empty train_all.jsonl after conversion" | tee -a "${LOG}"
  exit 4
fi

MODEL_PATH="${INIT_HF}"
RESUME_FLAG=()
if [ -f "${TRAIN_OUT}/best/training_state.pt" ] && compgen -G "${TRAIN_OUT}/best/*.safetensors" >/dev/null; then
  MODEL_PATH="${TRAIN_OUT}/best"
  RESUME_FLAG=(--resume)
  echo "=== Resuming from ${TRAIN_OUT}/best ===" | tee -a "${LOG}"
elif [ -f "${TRAIN_OUT}/epoch_001/training_state.pt" ] && compgen -G "${TRAIN_OUT}/epoch_001/*.safetensors" >/dev/null; then
  echo "=== best incomplete; seeding best from epoch_001 for resume ===" | tee -a "${LOG}"
  mkdir -p "${TRAIN_OUT}/best"
  rsync -a "${TRAIN_OUT}/epoch_001/" "${TRAIN_OUT}/best/"
  MODEL_PATH="${TRAIN_OUT}/best"
  RESUME_FLAG=(--resume)
fi

echo "=== torchrun 4-GPU DDP model=${MODEL_PATH} train_all.jsonl ===" | tee -a "${LOG}"
PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
  "${SCRIPTDIR}/train_sft1_qwen25vl.py" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --epochs 20 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-6 \
  --max-length 20000 \
  --max-pixels 602112 \
  --seed 42 \
  "${RESUME_FLAG[@]}" 2>&1 | tee -a "${LOG}"

echo "=== SFT1 vagen79 all-train finished at $(date) ===" | tee -a "${LOG}"
