#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
## --- Set ENV_URL before submitting, e.g.:
##   ENV_URL=http://10.23.x.x:5000 sbatch experiments/training/rl/train_only.slurm

set -euo pipefail
: "${ENV_URL:?must set ENV_URL to the dedicated env server}"

REPO=/project/peilab/atst/nimloth-feat-rl
SFT2_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2
SFT2_MODEL=${SFT2_OUT}/export_best_hf
SFT2_CHECKPOINT=${SFT2_OUT}/best

RUN_DATE=2026-06-27
EXPERIMENT_NAME=rl_sft2warm_60iter_2gpu
TRAIN_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/rl/${RUN_DATE}/${EXPERIMENT_NAME}
TRAIN_LOG=${TRAIN_OUT}/rl_train.log

mkdir -p "${TRAIN_OUT}"

# Wandb credentials
source /project/peilab/atst/flower/.env 2>/dev/null || true
export WANDB_DIR="${REPO}/.cache/wandb"

export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:$PATH
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm:${REPO}/external/VAGEN/verl:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export HOME=/project/peilab/atst/nimloth/.home

{
    echo "=== RL training $(date) ==="
    echo "Node: $(hostname)  ENV_URL: ${ENV_URL}"
    echo "Model: ${SFT2_MODEL}"
    echo "Output: ${TRAIN_OUT}"
} | tee "${TRAIN_LOG}"

cd "${REPO}"
exec python3 -m nimloth.training.rl.cli \
    --config configs/training/rl/exp_60iter_val5_save10.yaml \
    --model "${SFT2_MODEL}" \
    --llm-tune lora \
    --vision-tune freeze \
    --gradient-checkpointing \
    --wm-checkpoint "${SFT2_CHECKPOINT}/wm_predictor" \
    --state-proj-checkpoint "${SFT2_CHECKPOINT}/state_proj.pt" \
    --value-head-checkpoint "${SFT2_CHECKPOINT}/value_head" \
    --env-url "${ENV_URL}" \
    --attn-implementation sdpa \
    --max-pixels 784 \
    --output-dir "${TRAIN_OUT}" \
    2>&1 | tee -a "${TRAIN_LOG}"
