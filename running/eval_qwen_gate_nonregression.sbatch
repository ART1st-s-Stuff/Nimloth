#!/bin/bash
#SBATCH --job-name=qwen_gate_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=running/slurm_logs/%x_%j.out
#SBATCH --error=running/slurm_logs/%x_%j.err

set -euo pipefail

cd /project/peilab/atst/flower
mkdir -p running/slurm_logs

source running/setup_ai2thor_env.sh
source .venv/bin/activate

: "${WM_CHECKPOINT:?Set WM_CHECKPOINT to the trained WM checkpoint path.}"
: "${VALUE_CHECKPOINTS:?Set VALUE_CHECKPOINTS to one or more value checkpoint paths.}"
: "${PLANNER_LORA:=models/qwen_planner_lora_preaction_leakfree_8192_step00008000}"
: "${ROOT_OUT:=outputs/superpod/qwen_gate_nonregression_$(date +%Y%m%d_%H%M%S)}"
: "${NUM_EPISODES:=20}"
: "${MAX_STEPS:=20}"
: "${SPLIT:=test}"
: "${SEED:=20260522}"
: "${VISUAL_ENCODER:=qwen}"
: "${MODEL_NAME:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${MODEL_DTYPE:=auto}"
: "${DEVICE_MAP:=auto}"
: "${MAX_NEW_TOKENS:=256}"
: "${OVERRIDE_MARGIN:=0.05}"
: "${OVERRIDE_MAX_VALUE_STD:=0.03}"
: "${OVERRIDE_MAX_PRED_UNCERTAINTY:=0.08}"

mkdir -p "${ROOT_OUT}"

python dev/tmp_direct_qwen_ebnav_eval.py \
  --planner-lora "${PLANNER_LORA}" \
  --output-dir "${ROOT_OUT}/direct_qwen" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --split "${SPLIT}" \
  --seed "${SEED}" \
  --model-name "${MODEL_NAME}" \
  --model-dtype "${MODEL_DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-variant success_history

python dev/evaluate_eb_nav_qwen_override_rollout.py \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --value-checkpoints ${VALUE_CHECKPOINTS} \
  --planner-lora "${PLANNER_LORA}" \
  --output-dir "${ROOT_OUT}/qwen_gate" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --split "${SPLIT}" \
  --seed "${SEED}" \
  --visual-encoder "${VISUAL_ENCODER}" \
  --model-name "${MODEL_NAME}" \
  --model-dtype "${MODEL_DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --qwen-prompt-format special_token \
  --override-margin "${OVERRIDE_MARGIN}" \
  --override-max-value-std "${OVERRIDE_MAX_VALUE_STD}" \
  --override-max-pred-uncertainty "${OVERRIDE_MAX_PRED_UNCERTAINTY}"

python dev/compare_eb_nav_gate_nonregression.py \
  --baseline-summary "${ROOT_OUT}/direct_qwen/summary.json" \
  --gate-summary "${ROOT_OUT}/qwen_gate/summary.json" \
  --output-json "${ROOT_OUT}/nonregression.json"
