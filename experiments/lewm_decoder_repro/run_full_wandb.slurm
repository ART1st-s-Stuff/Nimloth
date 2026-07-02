#!/bin/bash
#SBATCH --job-name=lewm-dec-full
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/logs/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/logs/%x_%j.err

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/nimloth-lewm-repro}
PY=${PY:-${REPO}/.venv-lewm/bin/python}
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_NAME=${RUN_NAME:-nimloth_decoder_cube_full_ep1_eval1k}
OUT=${OUT:-/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/${RUN_DATE}/${RUN_NAME}_${SLURM_JOB_ID}}
STABLEWM_HOME=${STABLEWM_HOME:-/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/stablewm_home}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth}
WANDB_NAME=${WANDB_NAME:-${RUN_NAME}_${SLURM_JOB_ID}}

mkdir -p /project/peilab/atst/nimloth/outputs/experiments/lewm_repro/logs
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm:${PYTHONPATH:-}"
export STABLEWM_HOME

# W&B credentials live in the shared Flower env file on superpod.
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi

WANDB_ENTITY_ARGS=()
if [ -n "${WANDB_ENTITY:-}" ]; then
  WANDB_ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
fi

cat <<EOF
=== LeWM Nimloth decoder full train ===
date: $(date)
host: $(hostname)
repo: ${REPO}
commit: $(git rev-parse HEAD)
python: ${PY}
out: ${OUT}
stablewm_home: ${STABLEWM_HOME}
wandb_project: ${WANDB_PROJECT}
wandb_name: ${WANDB_NAME}
train_limit: ${TRAIN_LIMIT:-0}
val_limit: ${VAL_LIMIT:-4096}
epochs: ${EPOCHS:-1}
eval_every_steps: ${EVAL_EVERY_STEPS:-1000}
EOF

"${PY}" experiments/lewm_decoder_repro/train_nimloth_decoder_on_lewm_cube.py \
  --output-dir "${OUT}" \
  --stablewm-home "${STABLEWM_HOME}" \
  --download-dataset \
  --epochs "${EPOCHS:-1}" \
  --train-limit "${TRAIN_LIMIT:-0}" \
  --val-limit "${VAL_LIMIT:-4096}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --eval-every-steps "${EVAL_EVERY_STEPS:-1000}" \
  --decoder-hidden-dim "${DECODER_HIDDEN_DIM:-192}" \
  --decoder-depth "${DECODER_DEPTH:-4}" \
  --decoder-heads "${DECODER_HEADS:-3}" \
  --wandb-project "${WANDB_PROJECT}" \
  "${WANDB_ENTITY_ARGS[@]}" \
  --wandb-name "${WANDB_NAME}" \
  --wandb-preview-images "${WANDB_PREVIEW_IMAGES:-8}"

echo "=== finished $(date) ==="
