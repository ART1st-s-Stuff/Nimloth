#!/bin/bash
#SBATCH --job-name=sft2-default
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --requeue

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/sft2
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/latent_wm_value.yaml}

SFT2_LLM_TUNE=${SFT2_LLM_TUNE:-freeze}
SFT2_VISION_TUNE=${SFT2_VISION_TUNE:-full}
SFT2_MODE=${SFT2_MODE:-default}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-alltrain_8gpu_lora_cache}
if [ -n "${TRAIN_OUT_OVERRIDE:-}" ]; then
  TRAIN_OUT="${TRAIN_OUT_OVERRIDE}"
fi
LOG=${TRAIN_OUT}/sft2_train_${SLURM_JOB_ID:-local}.log
NGPUS=${NGPUS:-8}
SFT1_ADAPTER=""
SFT1_BEST=""

mkdir -p "${TRAIN_OUT}" "${REPO}/outputs/experiments/training/sft2/slurm"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export NCCL_ASYNC_ERROR_HANDLING=1; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; export PYTHONPATH=/project/peilab/atst/nimloth/src; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

ensure_sft1_merged() {
  if [ -f "${SFT1_BEST}/config.json" ]; then
    return 0
  fi
  if [ ! -f "${SFT1_ADAPTER}/adapter_config.json" ]; then
    echo "ERROR missing SFT1 LoRA adapter at ${SFT1_ADAPTER}" | tee -a "${LOG}"
    exit 1
  fi
  echo "=== Merge SFT1 LoRA ${SFT1_ADAPTER} -> hf_merged ===" | tee -a "${LOG}"
  python3 "${SFT1}/merge_lora_ckpt.py" \
    --base-model "${BASE_HF}" \
    --adapter-dir "${SFT1_ADAPTER}" \
    --out-dir "${SFT1_BEST}"
}

{
  echo "=== SFT2 train starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "Config: ${CONFIG}"
  echo "LLM tune: ${SFT2_LLM_TUNE} vision tune: ${SFT2_VISION_TUNE}"
  echo "Init HF: ${SFT1_BEST}"
  echo "Train jsonl: ${RECORDS_ROOT}/train_all.jsonl"
  echo "Output: ${TRAIN_OUT}"
  echo "Packed forward: ${PACKED_FORWARD:-0}"
  echo "Trajectory-aware batching: ${TRAJECTORY_AWARE_BATCHING:-0}"
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ "${SKIP_SFT1_DONE:-0}" != "1" ] && [ ! -f "${SFT1_RUN}/train_done.flag" ]; then
  echo "ERROR SFT1 train_done.flag missing at ${SFT1_RUN}" | tee -a "${LOG}"
  exit 1
fi
if [ ! -s "${RECORDS_ROOT}/train_all.jsonl" ]; then
  echo "ERROR missing train_all.jsonl" | tee -a "${LOG}"
  exit 1
fi

export SFT1_RUN BASE_HF EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX}" SFT1_EPOCH="${SFT1_EPOCH:-}"
bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh" | tee -a "${LOG}"
# shellcheck disable=SC1091
source "${SFT1_RUN}/.sft2_init_env"

MODEL_PATH="${SFT1_BEST}"
TUNE_ARGS=(--llm-tune "${SFT2_LLM_TUNE}" --vision-tune "${SFT2_VISION_TUNE}")
if [ "${SFT2_LLM_TUNE}" = "lora" ] || [ "${SFT2_VISION_TUNE}" = "lora" ]; then
  TUNE_ARGS+=(--lora-r "${LORA_R:-64}" --lora-alpha "${LORA_ALPHA:-128}")
fi
RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
  RESUME_ARGS=(--resume --resume-from "${RESUME_FROM}")
  echo "=== Resume SFT2 from ${RESUME_FROM} ===" | tee -a "${LOG}"
elif compgen -G "${TRAIN_OUT}/epoch_*/training_state.pt" > /dev/null \
  || [ -f "${TRAIN_OUT}/best/training_state.pt" ]; then
  RESUME_ARGS=(--resume)
  echo "=== Resume SFT2 from latest checkpoint under ${TRAIN_OUT} ===" | tee -a "${LOG}"
fi

WM_PRED_ARGS=()
if [ -n "${WM_PREDICTOR_CHECKPOINT:-}" ]; then
  WM_PRED_ARGS=(--wm-predictor-checkpoint "${WM_PREDICTOR_CHECKPOINT}")
fi

PREPROCESS_ARGS=()
if [ -n "${PREPROCESS_CACHE_DIR:-}" ]; then
  PREPROCESS_ARGS=(--preprocess-cache-dir "${PREPROCESS_CACHE_DIR}")
fi
TIMING_ARGS=()
if [ "${STEP_TIMING:-0}" = "1" ]; then
  TIMING_ARGS=(--step-timing --step-timing-interval "${STEP_TIMING_INTERVAL:-50}")
fi
PACKED_ARGS=()
if [ "${PACKED_FORWARD:-0}" = "1" ]; then
  PACKED_ARGS=(--packed-forward)
  if [ "${ALLOW_APPROX_TRAJECTORY_ONCE:-0}" = "1" ]; then
    PACKED_ARGS+=(--allow-approx-trajectory-once)
  fi
fi
TRAJECTORY_BATCH_ARGS=()
if [ "${TRAJECTORY_AWARE_BATCHING:-0}" = "1" ]; then
  TRAJECTORY_BATCH_ARGS=(--trajectory-aware-batching)
fi

if [ -n "${PREPROCESS_CACHE_DIR:-}" ]; then
  echo "=== prebuild preprocess cache ===" | tee -a "${LOG}"
  PYTHONUNBUFFERED=1 python "${SCRIPTDIR}/build_preprocess_cache.py" \
    --config "${CONFIG}" \
    "${TUNE_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${WM_PRED_ARGS[@]}" \
    "${PREPROCESS_ARGS[@]}" \
    "${PACKED_ARGS[@]}" \
    --model "${MODEL_PATH}" \
    --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
    --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
    --output-dir "${TRAIN_OUT}" \
    2>&1 | tee -a "${LOG}"
fi

export NIMLOTH_DDP_GPU_STRIDE="${NIMLOTH_DDP_GPU_STRIDE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPUS}}"
echo "=== torchrun ${NGPUS}-GPU SFT2 (nproc=${NPROC_PER_NODE}, gpu_stride=${NIMLOTH_DDP_GPU_STRIDE}) ===" | tee -a "${LOG}"
PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  "${SCRIPTDIR}/train.py" \
  --config "${CONFIG}" \
  "${TUNE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${WM_PRED_ARGS[@]}" \
  "${PREPROCESS_ARGS[@]}" \
  "${TIMING_ARGS[@]}" \
  "${PACKED_ARGS[@]}" \
  "${TRAJECTORY_BATCH_ARGS[@]}" \
  ${EXTRA_TRAIN_ARGS:-} \
  --epochs "${SFT2_EPOCHS:-10}" \
  --value-gamma "${VALUE_GAMMA:-1.0}" \
  --checkpoint-interval-steps "${CHECKPOINT_INTERVAL_STEPS:-0}" \
  --checkpoint-keep-last "${CHECKPOINT_KEEP_LAST:-0}" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --gradient-checkpointing \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "SFT2_DONE" > "${TRAIN_OUT}/sft2_done.flag"
echo "=== SFT2 finished at $(date) ===" | tee -a "${LOG}"
