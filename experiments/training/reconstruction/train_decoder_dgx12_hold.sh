#!/usr/bin/env bash
set -euo pipefail

# Run post-hoc reconstruction decoder training inside an existing Slurm hold allocation.
# Default target is the current dgx-12 hold job; override HOLD_JOB/RUN_NAME as needed.

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/feat-reconstruct}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
HOLD_JOB=${HOLD_JOB:-462486}
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_NAME=${RUN_NAME:-reconstruct_decoder_sft2_smoke_long}
OUT=${OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/reconstruction/${RUN_DATE}/${RUN_NAME}}
LOG=${LOG:-${OUT}/train.log}

SFT2_OUT=${SFT2_OUT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2}
SFT2_MODEL=${SFT2_MODEL:-${SFT2_OUT}/export_best_hf}
SFT2_BEST=${SFT2_BEST:-${SFT2_OUT}/best}
RECORDS_ROOT=${RECORDS_ROOT:-/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format}
WANDB_ENV=${WANDB_ENV:-/project/peilab/atst/flower/.env}

mkdir -p "${OUT}"

cat > "${OUT}/README.md" <<EOF
# Reconstruction decoder long smoke

- Purpose: post-hoc WM reconstruction decoder training smoke with wandb logging.
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- Hold job: ${HOLD_JOB}
- Model: ${SFT2_MODEL}
- WM checkpoint: ${SFT2_BEST}/wm_predictor
- State projector: ${SFT2_BEST}/state_proj.pt
- Train jsonl: ${RECORDS_ROOT}/train_all.jsonl
- Val jsonl: ${RECORDS_ROOT}/val_all.jsonl
- Output: ${OUT}
- Trainable: WMImageDecoder only.
- Frozen: Qwen, StateProjector, LatentWMPredictor.
- Scope: bounded smoke, max_train_records=64, max_val_records=32, max_val_batches=2, epochs=1.
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
  --epochs 1
  --batch-size 1
  --max-train-records 64
  --max-val-records 32
  --max-val-batches 2
  --image-size 255
  --patch-size 15
  --hidden-dim 1024
  --depth 4
  --heads 16
  --lr 1e-4
  --loss l1
  --log-interval 1
  --wandb-run-name "${WANDB_RUN_NAME}"
  --attn-implementation sdpa
)

{
  echo "=== reconstruction decoder long smoke $(date) ==="
  echo "host=$(hostname) HOLD_JOB=${HOLD_JOB} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "OUT=${OUT}"
  echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_RUN_NAME=${WANDB_RUN_NAME} WANDB_DIR=${WANDB_DIR}"
  printf 'CMD:'; printf ' %q' "${CMD[@]}"; echo
} | tee -a "${LOG}"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm:${PYTHONPATH:-}"

exec srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  bash -lc "cd '${REPO}' && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ${CMD[*]} 2>&1 | tee -a '${LOG}'"
