#!/bin/bash
#SBATCH --job-name=sft2-v3-ws24-preflight
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${REPO:?}"
: "${EXPECTED_COMMIT:?}"
: "${RUN_ROOT:?}"
: "${RUN_OUTPUT:?}"
: "${MODEL_PATH:?}"
: "${TRAIN_JSONL:?}"
: "${VAL_JSONL:?}"
: "${TRAIN_SHA256:?}"
: "${VAL_SHA256:?}"
: "${PREPROCESS_CACHE:?}"
: "${DINO_CACHE:?}"
: "${WANDB_ENTITY:?}"
: "${WANDB_PROJECT_NAME:?}"
: "${WANDB_RUN_NAME:?}"
: "${WANDB_RUN_ID:?}"
: "${TRAIN_PARTITION:?}"
: "${TRAIN_NODES:?}"
: "${TRAIN_GPUS_PER_NODE:?}"

test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
test ! -e "${RUN_OUTPUT}"
mkdir -p "${RUN_ROOT}"
LOG="${RUN_OUTPUT}.preflight_${SLURM_JOB_ID}.log"
exec > >(tee -a "${LOG}") 2>&1

WANDB_ENTITY_REQUESTED=${WANDB_ENTITY}
WANDB_PROJECT_REQUESTED=${WANDB_PROJECT_NAME}
WANDB_RUN_NAME_REQUESTED=${WANDB_RUN_NAME}
WANDB_RUN_ID_REQUESTED=${WANDB_RUN_ID}
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
fi
export WANDB_ENTITY=${WANDB_ENTITY_REQUESTED}
export WANDB_PROJECT=${WANDB_PROJECT_REQUESTED}
export WANDB_RUN_NAME=${WANDB_RUN_NAME_REQUESTED}
export WANDB_RUN_ID=${WANDB_RUN_ID_REQUESTED}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export PYTHONPATH=${REPO}/src:${REPO}/external/le-wm
export PYTHONPYCACHEPREFIX=/tmp/nimloth-ws24-preflight-${SLURM_JOB_ID}

/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/sft2/preflight_dino_grid_h1_t4.py" \
  --repo "${REPO}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --config "${REPO}/configs/training/sft2/dino_grid_k16_h1_t4.yaml" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --train-sha256 "${TRAIN_SHA256}" \
  --val-sha256 "${VAL_SHA256}" \
  --preprocess-cache "${PREPROCESS_CACHE}" \
  --dino-cache "${DINO_CACHE}" \
  --run-output "${RUN_OUTPUT}" \
  --result-json "${RUN_ROOT}/preflight.json" \
  --world-size 24 \
  --grad-accum 4 \
  --epochs 2 \
  --partition "${TRAIN_PARTITION}" \
  --nodes "${TRAIN_NODES}" \
  --gpus-per-node "${TRAIN_GPUS_PER_NODE}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT_NAME}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}" \
  --min-free-gib 80

test -s "${RUN_ROOT}/preflight.json"
echo "preflight_finished=$(date --iso-8601=seconds) job=${SLURM_JOB_ID}"
