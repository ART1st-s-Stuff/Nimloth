#!/usr/bin/env bash
#SBATCH --job-name=fork_offline_stack
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --exclude=dgx-16
#SBATCH --requeue
#SBATCH --output=/project/peilab/atst/flower/running/slurm_logs/%x-%j.out
#SBATCH --error=/project/peilab/atst/flower/running/slurm_logs/%x-%j.err

set -euo pipefail

cd /project/peilab/atst/flower

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a
  source /project/peilab/atst/.env
  set +a
fi

. .venv/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-flower}
export HOME=/project/peilab/atst
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/flower-fork-offline-${USER}-${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" running/slurm_logs
trap 'rm -rf "$TMPDIR"' EXIT

: "${FORK_JSONL:?Set FORK_JSONL to one or more fork_samples.jsonl paths or globs.}"
: "${BASE_WM_CHECKPOINT:=outputs/superpod/qwen_wm_s3_headens8_fixed_5ep/latest_for_fork_test.pt}"
: "${BASE_VALUE_CHECKPOINT:=outputs/superpod/qwen_value_rpe_headens8_rank_8192_5ep/fork_refine/best_value_ensemble.pt}"
: "${INIT_PLANNER_ADAPTER:=models/qwen_planner_lora_preaction_leakfree_8192_step00008000}"
: "${RUN_ROOT:=outputs/superpod/fork_offline_stack_${SLURM_JOB_ID:-manual}}"
: "${MODEL_NAME:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${MODEL_DTYPE:=auto}"
: "${DEVICE_MAP:=auto}"
: "${VISUAL_ENCODER:=qwen}"
: "${MIN_EFFECTIVE_LR_SCALE:=0.05}"

: "${TRAIN_WM:=1}"
: "${TRAIN_VALUE:=1}"
: "${TRAIN_QWEN:=1}"
: "${CALIBRATE_GATE:=1}"

mkdir -p "$RUN_ROOT"
export RUN_ROOT FORK_JSONL
python dev/verify_eb_nav_fork_pipeline.py --quiet

if [[ "$TRAIN_WM" == "1" ]]; then
  python dev/train_eb_nav_wm_from_forks.py \
    --fork-jsonl ${FORK_JSONL} \
    --wm-checkpoint "$BASE_WM_CHECKPOINT" \
    --output-dir "$RUN_ROOT/wm" \
    --epochs "${WM_EPOCHS:-1}" \
    --batch-size "${WM_BATCH_SIZE:-8}" \
    --lr "${WM_LR:-1e-5}" \
    --min-effective-lr-scale "$MIN_EFFECTIVE_LR_SCALE" \
    --save-every-steps "${WM_SAVE_EVERY_STEPS:-20}" \
    --cuda-device "${CUDA_DEVICE:-0}" \
    --model-name "$MODEL_NAME" \
    --model-dtype "$MODEL_DTYPE" \
    --device-map "$DEVICE_MAP" \
    --visual-encoder "$VISUAL_ENCODER" \
    --action-sensitivity-loss-weight "${WM_ACTION_SENSITIVITY_LOSS_WEIGHT:-0.1}"
fi

VALUE_WM_CHECKPOINT="$BASE_WM_CHECKPOINT"
if [[ -f "$RUN_ROOT/wm/best_wm.pt" ]]; then
  VALUE_WM_CHECKPOINT="$RUN_ROOT/wm/best_wm.pt"
fi

if [[ "$TRAIN_VALUE" == "1" ]]; then
  python dev/train_eb_nav_value_head_refine_from_forks.py \
    --fork-jsonl ${FORK_JSONL} \
    --init-checkpoint "$BASE_VALUE_CHECKPOINT" \
    --wm-checkpoint "$VALUE_WM_CHECKPOINT" \
    --output-dir "$RUN_ROOT/value" \
    --target-field "${VALUE_TARGET_FIELD:-continuation_reward}" \
    --epochs "${VALUE_EPOCHS:-3}" \
    --batch-size "${VALUE_BATCH_SIZE:-8}" \
    --lr "${VALUE_LR:-3e-4}" \
    --min-effective-lr-scale "$MIN_EFFECTIVE_LR_SCALE" \
    --rank-loss-weight "${VALUE_RANK_LOSS_WEIGHT:-0.5}" \
    --save-every-steps "${VALUE_SAVE_EVERY_STEPS:-20}" \
    --cuda-device "${CUDA_DEVICE:-0}" \
    --model-name "$MODEL_NAME" \
    --model-dtype "$MODEL_DTYPE" \
    --device-map "$DEVICE_MAP" \
    --visual-encoder "$VISUAL_ENCODER"
fi

if [[ "$TRAIN_QWEN" == "1" ]]; then
  python dev/build_eb_nav_fork_qwen_correction_sft.py \
    --fork-jsonl ${FORK_JSONL} \
    --output-dir "$RUN_ROOT/qwen_sft_data" \
    --target-field "${QWEN_TARGET_FIELD:-continuation_reward}" \
    --min-effective-lr-scale "$MIN_EFFECTIVE_LR_SCALE" \
    --min-outcome-gap "${QWEN_MIN_OUTCOME_GAP:-0.02}" \
    --max-history-actions "${QWEN_MAX_HISTORY_ACTIONS:-0}"

  python src/train/train_qwen_planner_lora.py \
    --sft-jsonl "$RUN_ROOT/qwen_sft_data/qwen_planner_fork_correction_train.jsonl" \
    --validation-jsonl "$RUN_ROOT/qwen_sft_data/qwen_planner_fork_correction_test.jsonl" \
    --init-adapter-path "$INIT_PLANNER_ADAPTER" \
    --output-dir "$RUN_ROOT/qwen_adapter" \
    --checkpoint-dir "$RUN_ROOT/qwen_checkpoints" \
    --epochs "${QWEN_EPOCHS:-1}" \
    --batch-size "${QWEN_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${QWEN_GRAD_ACCUM:-8}" \
    --lr "${QWEN_LR:-5e-6}" \
    --limit "${QWEN_LIMIT:-0}" \
    --min-sample-weight "$MIN_EFFECTIVE_LR_SCALE" \
    --save-every-steps "${QWEN_SAVE_EVERY_STEPS:-256}" \
    --validate-every-steps "${QWEN_VALIDATE_EVERY_STEPS:-256}" \
    --validation-limit "${QWEN_VALIDATION_LIMIT:-64}" \
    --validation-max-new-tokens "${QWEN_VALIDATION_MAX_NEW_TOKENS:-256}" \
    --resume-from-checkpoint latest \
    --validation-extract-latent \
    --no-progress
fi

if [[ "$CALIBRATE_GATE" == "1" ]]; then
  python dev/calibrate_eb_nav_gate_from_forks.py \
    --fork-jsonl ${FORK_JSONL} \
    --output-json "$RUN_ROOT/gate_config.json" \
    --target-field "${GATE_TARGET_FIELD:-continuation_reward}" \
    --min-effective-lr-scale "$MIN_EFFECTIVE_LR_SCALE" \
    --min-outcome-delta "${GATE_MIN_OUTCOME_DELTA:-0.02}" \
    --min-precision "${GATE_MIN_PRECISION:-0.6}" \
    --min-coverage "${GATE_MIN_COVERAGE:-0.01}"
fi

python - <<'PY'
import json
import os
import shlex
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
summary = {
    "run_root": str(root),
    "fork_jsonl": os.environ["FORK_JSONL"],
    "wm_checkpoint": str(root / "wm" / "best_wm.pt"),
    "value_checkpoint": str(root / "value" / "best_value_ensemble.pt"),
    "qwen_adapter": str(root / "qwen_adapter"),
    "gate_config": str(root / "gate_config.json"),
    "next_eval_sbatch": "running/eval_qwen_gate_nonregression.sbatch",
}
(root / "offline_stack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(root / "next_eval_env.sh").write_text(
    "\n".join(
        [
            f"export OFFLINE_STACK_SUMMARY={shlex.quote(str(root / 'offline_stack_summary.json'))}",
            f"export WM_CHECKPOINT={shlex.quote(summary['wm_checkpoint'])}",
            f"export VALUE_CHECKPOINTS={shlex.quote(summary['value_checkpoint'])}",
            f"export PLANNER_LORA={shlex.quote(summary['qwen_adapter'])}",
            f"export GATE_CONFIG={shlex.quote(summary['gate_config'])}",
            f"export FORK_JSONL_FOR_GATE={shlex.quote(summary['fork_jsonl'])}",
            'sbatch running/eval_qwen_gate_nonregression.sbatch',
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps(summary), flush=True)
PY
