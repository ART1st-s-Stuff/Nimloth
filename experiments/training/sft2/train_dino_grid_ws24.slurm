#!/bin/bash
#SBATCH --job-name=sft2-value-v3-ws24-normal
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=08:00:00

set -euo pipefail

: "${REPO:?}"
: "${EXPECTED_COMMIT:?}"
: "${RUN_ROOT:?}"
: "${RUN_OUTPUT:?}"
: "${MODEL_PATH:?}"
: "${TRAIN_JSONL:?}"
: "${VAL_JSONL:?}"
: "${PREPROCESS_CACHE:?}"
: "${WANDB_ENTITY:?}"
: "${WANDB_PROJECT_NAME:?}"
: "${WANDB_RUN_NAME:?}"
: "${WANDB_RUN_ID:?}"
: "${EXPECTED_STEPS:?}"

test "${SLURM_JOB_PARTITION}" = "normal"
test "${SLURM_NNODES}" -eq 6
test "${SLURM_NTASKS}" -eq 6
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
for required in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_JSONL}" \
  "${VAL_JSONL}" \
  "${PREPROCESS_CACHE}/cache_done.flag" \
  "${RUN_ROOT}/preflight.json"; do
  test -s "${required}"
done
if [ -d "${RUN_OUTPUT}" ] && [ -n "$(find "${RUN_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "ERROR fresh SFT2 output is non-empty: ${RUN_OUTPUT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
CONTROLLER_LOG="${RUN_OUTPUT}.controller_${SLURM_JOB_ID}.log"
exec > >(tee -a "${CONTROLLER_LOG}") 2>&1

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
test "${#HOSTS[@]}" -eq 6
export MASTER_ADDR="${HOSTS[0]}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export REPO EXPECTED_COMMIT RUN_ROOT RUN_OUTPUT MODEL_PATH TRAIN_JSONL VAL_JSONL
export PREPROCESS_CACHE WANDB_ENTITY WANDB_PROJECT_NAME WANDB_RUN_NAME WANDB_RUN_ID
export MASTER_ADDR MASTER_PORT

{
  echo "# SFT2 value-v3 formal run"
  echo
  echo "- job: ${SLURM_JOB_ID}"
  echo "- commit: ${EXPECTED_COMMIT}"
  echo "- nodes: ${SLURM_JOB_NODELIST}"
  echo "- rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
  echo "- topology: 6 nodes x 4 H800; world size 24; B1; GA4; effective global batch 96"
  echo "- objective: H=1/T=4, decision-state executed-action MC ValueHead, no rank loss"
  echo "- init: ${MODEL_PATH}"
  echo "- preprocess cache: ${PREPROCESS_CACHE} (read-only reuse)"
  echo "- W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}/${WANDB_RUN_NAME} (${WANDB_RUN_ID})"
  echo "- output: ${RUN_OUTPUT}"
  echo "- resume: false; fresh optimizer; 20-minute latest checkpoints"
} > "${RUN_ROOT}/README.md"

echo "start=$(date --iso-8601=seconds) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
srun \
  --nodes=6 \
  --ntasks=6 \
  --ntasks-per-node=1 \
  --cpus-per-task=32 \
  --gres=gpu:4 \
  --kill-on-bad-exit=1 \
  bash "${REPO}/experiments/training/sft2/run_dino_grid_ws24_node.sh"

"${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}/bin/python3" \
  "${REPO}/experiments/training/sft2/validate_dino_grid_training_output.py" \
  --output-dir "${RUN_OUTPUT}" \
  --expected-step "${EXPECTED_STEPS}" \
  --expected-world-size 24 \
  --expected-wandb-run-id "${WANDB_RUN_ID}" \
  --result-json "${RUN_ROOT}/completion_validation.json"
echo "finished=$(date --iso-8601=seconds) job=${SLURM_JOB_ID}"
