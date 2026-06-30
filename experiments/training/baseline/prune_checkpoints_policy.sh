#!/usr/bin/env bash
# Apply Nimloth baseline checkpoint retention: latest + every 10th + best val.
set -euo pipefail

REPO="${REPO:-/project/peilab/atst/nimloth}"
SCRIPTDIR="${REPO}/experiments/training/baseline"

TRAIN_RUN_DIR="${1:?TRAIN_RUN_DIR required}"
KEEP_EVERY="${KEEP_EVERY:-10}"
DRY_RUN="${DRY_RUN:-0}"

CKPT_DIR="${TRAIN_RUN_DIR}/checkpoints"
VAL_DIR="${TRAIN_RUN_DIR}/validation"
VAL_CURVE="${TRAIN_RUN_DIR}/val_wandb_watcher/val_curve.jsonl"
WATCHER_VAL_DIR="${TRAIN_RUN_DIR}/val_wandb_watcher/validation"
VAL_RUN_DIR="${TRAIN_RUN_DIR}/val_wandb_watcher"

args=(--checkpoint-dir "${CKPT_DIR}" --validation-dir "${VAL_DIR}" --keep-every "${KEEP_EVERY}")
args+=(--val-run-dir "${VAL_RUN_DIR}")
[ -f "${VAL_CURVE}" ] && args+=(--val-curve "${VAL_CURVE}")
if [ -d "${WATCHER_VAL_DIR}" ] && [ -z "$(ls -A "${VAL_DIR}" 2>/dev/null)" ]; then
  args=(--checkpoint-dir "${CKPT_DIR}" --validation-dir "${WATCHER_VAL_DIR}" --keep-every "${KEEP_EVERY}")
  args+=(--val-run-dir "${VAL_RUN_DIR}")
  [ -f "${VAL_CURVE}" ] && args+=(--val-curve "${VAL_CURVE}")
fi
[ "${DRY_RUN}" = "1" ] && args+=(--dry-run)

python3 "${SCRIPTDIR}/prune_checkpoints.py" "${args[@]}"
