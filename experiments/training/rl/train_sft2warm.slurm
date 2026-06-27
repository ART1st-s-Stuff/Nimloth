#!/bin/bash
#SBATCH --job-name=rl-sft2warm
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
## Use --nodelist=<node> to pin; omit for auto-assignment.

set -euo pipefail

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO=/project/peilab/atst/nimloth-feat-rl
SFT2_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2
SFT2_MODEL=${SFT2_OUT}/export_best_hf
SFT2_CHECKPOINT=${SFT2_OUT}/best

RUN_DATE=2026-06-27
EXPERIMENT_NAME=rl_sft2warm_60iter_env2gpu
TRAIN_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/rl/${RUN_DATE}/${EXPERIMENT_NAME}
ENV_PORT=5000
ENV_LOG=${TRAIN_OUT}/env_server.log
TRAIN_LOG=${TRAIN_OUT}/rl_train.log

mkdir -p "${TRAIN_OUT}" /project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv
export HOME=/project/peilab/atst/nimloth/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:$PATH
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm:${REPO}/external/VAGEN/verl:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Vulkan / AI2-THOR
VULKAN_ROOT=/project/peilab/atst/flower/.local-vulkan
VULKAN_LIB_DIR=${VULKAN_ROOT}/extracted/usr/lib/x86_64-linux-gnu
VULKAN_RUNTIME_DIR=${VULKAN_ROOT}/runtime/extracted/usr/lib/x86_64-linux-gnu
VULKAN_TOOLS_DIR=${VULKAN_ROOT}/tools/extracted/usr/bin
AI2THOR_HOME_ROOT=/project/peilab/atst/flower/.ai2thor-home

mkdir -p "${AI2THOR_HOME_ROOT}"
if [ -f "${VULKAN_LIB_DIR}/libvulkan.so.1" ]; then
    ln -sf libvulkan.so.1 "${VULKAN_LIB_DIR}/libvulkan.so" 2>/dev/null || true
    export LD_LIBRARY_PATH="${VULKAN_LIB_DIR}:${VULKAN_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${VULKAN_LIB_DIR}:${VULKAN_RUNTIME_DIR}:${LIBRARY_PATH:-}"
    export PATH="${VULKAN_TOOLS_DIR}:${PATH}"
    export HOME="${AI2THOR_HOME_ROOT}"
    export VK_ICD_FILENAMES="${VULKAN_ROOT}/icd.d/nvidia_icd.json"
    export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
    rm -f ${HOME}/.ai2thor/cuda-vulkan-mapping.json 2>/dev/null || true
fi

# ------------------------------------------------------------------
# Log header
# ------------------------------------------------------------------
{
    echo "=== RL SFT2 warm-start training $(date) ==="
    echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURM_JOB_NODELIST}"
    echo "Model: ${SFT2_MODEL}"
    echo "SFT2 checkpoint: ${SFT2_CHECKPOINT}"
    echo "Output: ${TRAIN_OUT}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES:-auto}"
} | tee "${TRAIN_LOG}"

# ------------------------------------------------------------------
# Start env server on GPU 0
# ------------------------------------------------------------------
echo "=== Starting env server ===" | tee -a "${TRAIN_LOG}"

cd "${REPO}/external/VAGEN"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -m vagen.server.server \
    server.host=0.0.0.0 \
    server.port=${ENV_PORT} \
    use_state_reward=False \
    navigation.devices=[0] \
    navigation.max_workers=16 \
    > "${ENV_LOG}" 2>&1 &

ENV_PID=$!
echo "Env server PID: ${ENV_PID}" | tee -a "${TRAIN_LOG}"

# Wait for env server
ENV_URL="http://127.0.0.1:${ENV_PORT}"
for i in $(seq 1 60); do
    if curl -s "${ENV_URL}/health" > /dev/null 2>&1; then
        echo "Env server ready after ${i}s" | tee -a "${TRAIN_LOG}"
        break
    fi
    sleep 1
done
if ! curl -s "${ENV_URL}/health" > /dev/null 2>&1; then
    echo "FATAL: env server failed to start" | tee -a "${TRAIN_LOG}"
    kill ${ENV_PID} 2>/dev/null || true
    exit 1
fi

# ------------------------------------------------------------------
# Verify SFT2 checkpoint
# ------------------------------------------------------------------
echo "=== Verifying SFT2 checkpoint ===" | tee -a "${TRAIN_LOG}"
for f in state_proj.pt wm_predictor/predictor.pt value_head/value_head.pt; do
    if [ ! -f "${SFT2_CHECKPOINT}/${f}" ]; then
        echo "FATAL: missing ${SFT2_CHECKPOINT}/${f}" | tee -a "${TRAIN_LOG}"
        kill ${ENV_PID} 2>/dev/null || true
        exit 1
    fi
    echo "  OK: ${f}" | tee -a "${TRAIN_LOG}"
done

# Verify model
if [ ! -f "${SFT2_MODEL}/config.json" ]; then
    echo "FATAL: missing ${SFT2_MODEL}/config.json" | tee -a "${TRAIN_LOG}"
    kill ${ENV_PID} 2>/dev/null || true
    exit 1
fi
echo "  OK: model at ${SFT2_MODEL}" | tee -a "${TRAIN_LOG}"

# ------------------------------------------------------------------
# RL Training
# ------------------------------------------------------------------
echo "=== Launching RL training ===" | tee -a "${TRAIN_LOG}"

cd "${REPO}"

# Use single GPU (GPU 1) for Qwen model + training
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -m nimloth.training.rl.cli \
    --config configs/training/rl/exp_60iter_val5_save10.yaml \
    --model "${SFT2_MODEL}" \
    --llm-tune freeze \
    --vision-tune freeze \
    --wm-checkpoint "${SFT2_CHECKPOINT}/wm_predictor" \
    --state-proj-checkpoint "${SFT2_CHECKPOINT}/state_proj.pt" \
    --value-head-checkpoint "${SFT2_CHECKPOINT}/value_head" \
    --env-url "${ENV_URL}" \
    --attn-implementation sdpa \
    --max-pixels 3136 \
    --output-dir "${TRAIN_OUT}" \
    >> "${TRAIN_LOG}" 2>&1

TRAIN_EXIT=$?

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
kill ${ENV_PID} 2>/dev/null || true
wait ${ENV_PID} 2>/dev/null || true

echo "=== RL training finished exit=${TRAIN_EXIT} at $(date) ===" | tee -a "${TRAIN_LOG}"
exit ${TRAIN_EXIT}
