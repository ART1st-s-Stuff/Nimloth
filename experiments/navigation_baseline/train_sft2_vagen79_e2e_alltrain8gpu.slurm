#!/bin/bash
#SBATCH --job-name=sft2-e2e-all8g
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
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

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen79_nimloth_format
SFT1_RUN=${SCRIPTDIR}/runs/sft1_train_vagen79_qwen25vl_alltrain_8gpu_lora
BASE_HF=${SCRIPTDIR}/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface
# Default first SFT2 test: LoRA e2e. Set SFT2_MODE=full for full-parameter Qwen.
SFT2_MODE=${SFT2_MODE:-lora}
EVAL_TAG_PREFIX=${EVAL_TAG_PREFIX:-alltrain_8gpu_lora_cache}
TRAIN_OUT=${SCRIPTDIR}/runs/sft2_e2e_vagen79_alltrain8gpu
if [ "${SFT2_MODE}" = "lora" ]; then
  TRAIN_OUT=${SCRIPTDIR}/runs/sft2_e2e_vagen79_alltrain8gpu_lora
fi
if [ -n "${TRAIN_OUT_OVERRIDE:-}" ]; then
  TRAIN_OUT="${TRAIN_OUT_OVERRIDE}"
fi
LOG=${TRAIN_OUT}/sft2_e2e_alltrain8gpu_${SLURM_JOB_ID:-local}.log
NGPUS=${NGPUS:-8}
SFT1_ADAPTER=""
SFT1_BEST=""

mkdir -p "${TRAIN_OUT}" "${ROOT}/outputs/experiments/navigation_baseline/slurm"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; export PYTHONPATH=/project/peilab/atst/nimloth/src; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

ensure_sft1_merged() {
  if [ -f "${SFT1_BEST}/config.json" ]; then
    return 0
  fi
  if [ ! -f "${SFT1_ADAPTER}/adapter_config.json" ]; then
    echo "ERROR missing SFT1 LoRA adapter at ${SFT1_ADAPTER}" | tee -a "${LOG}"
    exit 1
  fi
  echo "=== Merge SFT1 LoRA ${SFT1_ADAPTER} -> hf_merged ===" | tee -a "${LOG}"
  python3 "${SCRIPTDIR}/merge_sft1_lora_ckpt.py" \
    --base-model "${BASE_HF}" \
    --adapter-dir "${SFT1_ADAPTER}" \
    --out-dir "${SFT1_BEST}"
}

{
  echo "=== SFT2 e2e after alltrain SFT1 LoRA starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} node: ${SLURM_JOB_NODELIST}"
  echo "SFT1 LoRA adapter: ${SFT1_ADAPTER}"
  echo "Init HF (merged): ${SFT1_BEST}"
  echo "SFT2_MODE: ${SFT2_MODE}"
  echo "SFT1 pick manifest: ${SFT1_RUN}/sft2_init_pick.json"
  echo "Train jsonl: ${RECORDS_ROOT}/train_all.jsonl"
  echo "Output: ${TRAIN_OUT}"
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ "${SKIP_SFT1_DONE:-0}" != "1" ] && [ ! -f "${SFT1_RUN}/train_done.flag" ]; then
  echo "ERROR SFT1 train_done.flag missing at ${SFT1_RUN} (set SKIP_SFT1_DONE=1 to override)" | tee -a "${LOG}"
  exit 1
fi
if [ ! -s "${RECORDS_ROOT}/train_all.jsonl" ]; then
  echo "ERROR missing train_all.jsonl" | tee -a "${LOG}"
  exit 1
fi

# Pick SFT1 init (SFT1_EPOCH=N or success-rate early-stop).
export SFT1_RUN BASE_HF EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX}" SFT1_EPOCH="${SFT1_EPOCH:-}"
bash "${SCRIPTDIR}/resolve_sft1_init_for_sft2.sh" | tee -a "${LOG}"
# shellcheck disable=SC1091
source "${SFT1_RUN}/.sft2_init_env"

MODEL_PATH="${SFT1_BEST}"
RESUME_ARGS=()
TUNE_ARGS=(
  --llm-tune "${SFT2_LLM_TUNE:-lora}"
  --vision-tune "${SFT2_VISION_TUNE:-freeze}"
  --lora-r "${LORA_R:-64}"
  --lora-alpha "${LORA_ALPHA:-128}"
)
if [ "${SFT2_MODE}" = "lora" ]; then
  if [ -f "${TRAIN_OUT}/best/training_state.pt" ] && [ -f "${TRAIN_OUT}/best/adapter_config.json" ]; then
    RESUME_ARGS=(--resume)
    echo "=== Continue SFT2 LoRA from ${TRAIN_OUT}/best (base=${SFT1_BEST}) ===" | tee -a "${LOG}"
  fi
elif [ "${SFT2_MODE}" = "full" ]; then
  TUNE_ARGS=(--llm-tune full --vision-tune freeze)
  if [ -f "${TRAIN_OUT}/best/config.json" ] && [ -f "${TRAIN_OUT}/best/state_proj.pt" ]; then
    MODEL_PATH="${TRAIN_OUT}/best"
    RESUME_ARGS=(--resume)
    echo "=== Continue SFT2 full from ${MODEL_PATH} ===" | tee -a "${LOG}"
  fi
fi

echo "=== torchrun ${NGPUS}-GPU DDP SFT2 model=${MODEL_PATH} llm=${SFT2_LLM_TUNE:-lora} vision=${SFT2_VISION_TUNE:-freeze} ===" | tee -a "${LOG}"
WM_PRED_ARGS=()
if [ -n "${WM_PREDICTOR_CHECKPOINT:-}" ]; then
  WM_PRED_ARGS=(--wm-predictor-checkpoint "${WM_PREDICTOR_CHECKPOINT}")
fi
PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
  "${SCRIPTDIR}/train_sft2_qwen25vl.py" \
  "${TUNE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${WM_PRED_ARGS[@]}" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl" \
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr-qwen-start 1e-8 \
  --lr-qwen-peak 5e-7 \
  --qwen-lr-warmup-ratio 0.15 \
  --state-proj-lr 1e-4 \
  --wm-predictor-lr 3e-4 \
  --lambda-ce 1.0 \
  --lambda-wm-start 0.1 \
  --lambda-wm-end 1.0 \
  --max-length 12000 \
  --max-pixels 602112 \
  --emb-dim 128 \
  --gradient-checkpointing \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "SFT2_DONE" > "${TRAIN_OUT}/sft2_done.flag"
echo "=== SFT2 e2e alltrain8gpu finished at $(date) ===" | tee -a "${LOG}"
