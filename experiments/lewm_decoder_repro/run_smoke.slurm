#!/bin/bash
#SBATCH --job-name=lewm-dec-smoke
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/logs/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/logs/%x_%j.err

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/nimloth-lewm-repro}
PY=${PY:-${REPO}/.venv-lewm/bin/python}
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_NAME=${RUN_NAME:-nimloth_decoder_cube_smoke}
OUT=${OUT:-/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/${RUN_DATE}/${RUN_NAME}_${SLURM_JOB_ID}}
STABLEWM_HOME=${STABLEWM_HOME:-/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/stablewm_home}

mkdir -p /project/peilab/atst/nimloth/outputs/experiments/lewm_repro/logs
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm:${PYTHONPATH:-}"
export STABLEWM_HOME

cat <<EOF
=== LeWM Nimloth decoder smoke ===
date: $(date)
host: $(hostname)
repo: ${REPO}
commit: $(git rev-parse HEAD)
python: ${PY}
out: ${OUT}
stablewm_home: ${STABLEWM_HOME}
EOF

"${PY}" experiments/lewm_decoder_repro/train_nimloth_decoder_on_lewm_cube.py \
  --output-dir "${OUT}" \
  --stablewm-home "${STABLEWM_HOME}" \
  --download-dataset \
  --epochs "${EPOCHS:-5}" \
  --train-limit "${TRAIN_LIMIT:-2048}" \
  --val-limit "${VAL_LIMIT:-256}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --decoder-hidden-dim "${DECODER_HIDDEN_DIM:-192}" \
  --decoder-depth "${DECODER_DEPTH:-4}" \
  --decoder-heads "${DECODER_HEADS:-3}"

echo "=== finished $(date) ==="
