#!/usr/bin/env bash
set -euo pipefail

# Full post-hoc reconstruction decoder training inside an existing dgx-12 hold allocation.
# Uses all train rollouts. Checkpoints every 500 optimizer steps and keeps the last 2.

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/feat-reconstruct}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
HOLD_JOB=${HOLD_JOB:-462486}
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_NAME=${RUN_NAME:-reconstruct_decoder_sft2_full_4epoch}
OUT=${OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/${RUN_DATE}/${RUN_NAME}}
LOG=${LOG:-${OUT}/train.log}
RESUME=${RESUME:-0}

SFT2_OUT=${SFT2_OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2}
SFT2_MODEL=${SFT2_MODEL:-${SFT2_OUT}/export_best_hf}
SFT2_BEST=${SFT2_BEST:-${SFT2_OUT}/best}
RECORDS_ROOT=${RECORDS_ROOT:-/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format}
WANDB_ENV=${WANDB_ENV:-/project/peilab/atst/flower/.env}

mkdir -p "${OUT}"

TRAIN_TRANSITIONS=$(${PY} - <<'PY'
from pathlib import Path
import json
p = Path('/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format/train_all.jsonl')
# Cheap estimate: converted Nimloth records store one action per transition candidate.
# Exact TransitionJsonlDataset count was measured as 54702 for this dataset.
print(54702)
PY
)
EPOCHS=${EPOCHS:-4}
TOTAL_STEPS=$((TRAIN_TRANSITIONS * EPOCHS))

cat > "${OUT}/README.md" <<EOF
# Reconstruction decoder full training

- Purpose: post-hoc WM reconstruction decoder training with wandb logging.
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- Hold job: ${HOLD_JOB}
- Model: ${SFT2_MODEL}
- WM checkpoint: ${SFT2_BEST}/wm_predictor
- State projector: ${SFT2_BEST}/state_proj.pt
- Train jsonl: ${RECORDS_ROOT}/train_all.jsonl
- Val jsonl: ${RECORDS_ROOT}/val_all.jsonl
- Train transitions: ${TRAIN_TRANSITIONS}
- Epochs: ${EPOCHS}
- Planned optimizer steps: ${TOTAL_STEPS}
- Output: ${OUT}
- Trainable: WMImageDecoder only.
- Frozen: Qwen, StateProjector, LatentWMPredictor.
- Checkpoint: every 500 steps, keep last 2 step checkpoints; use RESUME=1 to continue.
EOF

for p in \
  "${SFT2_MODEL}/config.json" \
  "${SFT2_BEST}/state_proj.pt" \
  "${SFT2_BEST}/wm_predictor/predictor.pt" \
  "${RECORDS_ROOT}/train_all.jsonl" \
  "${RECORDS_ROOT}/val_all.jsonl"; do
  if [ ! -e "$p" ]; then
    echo "ERROR missing required path: $p" | tee -a "${LOG}"
    exit 1
  fi
done

if [ -f "${WANDB_ENV}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${WANDB_ENV}"
  set +a
fi
export WANDB_PROJECT=${WANDB_PROJECT:-nimloth}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}}
export WANDB_DIR=${WANDB_DIR:-${OUT}/wandb}
mkdir -p "${WANDB_DIR}"

CMD=(
  "${PY}" -m nimloth.training.reconstruction.cli
  --model "${SFT2_MODEL}"
  --state-proj-checkpoint "${SFT2_BEST}/state_proj.pt"
  --wm-checkpoint "${SFT2_BEST}/wm_predictor"
  --train-jsonl "${RECORDS_ROOT}/train_all.jsonl"
  --val-jsonl "${RECORDS_ROOT}/val_all.jsonl"
  --output-dir "${OUT}"
  --epochs "${EPOCHS}"
  --batch-size 1
  --max-train-records -1
  --max-val-records -1
  --max-val-batches 256
  --image-size 128
  --patch-size 16
  --hidden-dim 128
  --depth 1
  --heads 4
  --lr 1e-4
  --loss l1
  --log-interval 500
  --save-interval 500
  --keep-last-checkpoints 2
  --wandb-image-samples 5
  --wandb-image-interval 500
  --wandb-run-name "${WANDB_RUN_NAME}"
  --attn-implementation sdpa
)
if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

{
  echo "=== reconstruction decoder full train $(date) ==="
  echo "host=$(hostname) HOLD_JOB=${HOLD_JOB} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "OUT=${OUT}"
  echo "EPOCHS=${EPOCHS} TRAIN_TRANSITIONS=${TRAIN_TRANSITIONS} TOTAL_STEPS=${TOTAL_STEPS} RESUME=${RESUME}"
  echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_RUN_NAME=${WANDB_RUN_NAME} WANDB_DIR=${WANDB_DIR}"
  printf 'CMD:'; printf ' %q' "${CMD[@]}"; echo
} | tee -a "${LOG}"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm:${PYTHONPATH:-}"
module load slurm >/dev/null 2>&1 || true

exec srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  bash -lc "cd '${REPO}' && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ${CMD[*]} 2>&1 | tee -a '${LOG}'"
