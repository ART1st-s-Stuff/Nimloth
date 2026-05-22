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

: "${OFFLINE_STACK_SUMMARY:=}"
if [[ -n "${OFFLINE_STACK_SUMMARY}" ]]; then
  eval "$(python - <<'PY'
import json
import os
import shlex
from pathlib import Path

summary = json.loads(Path(os.environ["OFFLINE_STACK_SUMMARY"]).read_text(encoding="utf-8"))
mapping = {
    "WM_CHECKPOINT": summary.get("wm_checkpoint", ""),
    "VALUE_CHECKPOINTS": summary.get("value_checkpoint", ""),
    "PLANNER_LORA": summary.get("qwen_adapter", ""),
    "GATE_CONFIG": summary.get("gate_config", ""),
    "FORK_JSONL_FOR_GATE": summary.get("fork_jsonl", ""),
}
for key, value in mapping.items():
    if value and not os.environ.get(key):
        print(f"export {key}={shlex.quote(str(value))}")
PY
)"
fi

: "${WM_CHECKPOINT:?Set WM_CHECKPOINT to the trained WM checkpoint path.}"
: "${VALUE_CHECKPOINTS:?Set VALUE_CHECKPOINTS to one or more value checkpoint paths.}"
: "${PLANNER_LORA:=models/qwen_planner_lora_preaction_leakfree_8192_step00008000}"
: "${ROOT_OUT:=outputs/superpod/qwen_gate_nonregression_${SLURM_JOB_ID:-manual}}"
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
: "${GATE_CONFIG:=}"
: "${FORK_JSONL_FOR_GATE:=}"
: "${MIN_SUCCESS_DELTA:=0.0}"
: "${MAX_COLLISION_DELTA:=0.0}"
: "${MAX_PLANNER_FAILURE_DELTA:=0.0}"

mkdir -p "${ROOT_OUT}"

GATE_ARGS=()
if [[ -n "${FORK_JSONL_FOR_GATE}" ]]; then
  python dev/calibrate_eb_nav_gate_from_forks.py \
    --fork-jsonl ${FORK_JSONL_FOR_GATE} \
    --output-json "${ROOT_OUT}/gate_config.json"
  GATE_CONFIG="${ROOT_OUT}/gate_config.json"
fi
if [[ -n "${GATE_CONFIG}" ]]; then
  GATE_ARGS=(--gate-config "${GATE_CONFIG}")
fi

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
  "${GATE_ARGS[@]}" \
  --override-margin "${OVERRIDE_MARGIN}" \
  --override-max-value-std "${OVERRIDE_MAX_VALUE_STD}" \
  --override-max-pred-uncertainty "${OVERRIDE_MAX_PRED_UNCERTAINTY}"

python dev/compare_eb_nav_gate_nonregression.py \
  --baseline-summary "${ROOT_OUT}/direct_qwen/summary.json" \
  --gate-summary "${ROOT_OUT}/qwen_gate/summary.json" \
  --output-json "${ROOT_OUT}/nonregression.json" \
  --min-success-delta "${MIN_SUCCESS_DELTA}" \
  --max-collision-delta "${MAX_COLLISION_DELTA}" \
  --max-planner-failure-delta "${MAX_PLANNER_FAILURE_DELTA}"
