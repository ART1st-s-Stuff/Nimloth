#!/bin/bash
# Resource components are supplied on the sbatch command line:
#   component 0: 2 physical nodes, 2 tasks/node, 2 GPUs/task (4 GPUs/node)
#   component 1: 2 physical nodes, 1 task/node, 2 GPUs/task (2 GPUs/node)
#SBATCH --job-name=sft2-value-v3-ws12-het
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err

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

test "${SLURM_HET_SIZE}" -eq 2
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

mapfile -t HOSTS0 < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}")
mapfile -t HOSTS1 < <(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}")
test "${#HOSTS0[@]}" -eq 2
test "${#HOSTS1[@]}" -eq 2
test "$(printf '%s\n' "${HOSTS0[@]}" "${HOSTS1[@]}" | sort -u | wc -l)" -eq 4
export MASTER_ADDR="${HOSTS0[0]}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export REPO EXPECTED_COMMIT RUN_ROOT RUN_OUTPUT MODEL_PATH TRAIN_JSONL VAL_JSONL
export PREPROCESS_CACHE WANDB_ENTITY WANDB_PROJECT_NAME WANDB_RUN_NAME WANDB_RUN_ID
export MASTER_ADDR MASTER_PORT

{
  echo "# SFT2 value-v3 formal heterogeneous world12 run"
  echo
  echo "- job: ${SLURM_JOB_ID}"
  echo "- commit: ${EXPECTED_COMMIT}"
  echo "- physical topology: group0=${SLURM_JOB_NODELIST_HET_GROUP_0} 4+4; group1=${SLURM_JOB_NODELIST_HET_GROUP_1} 2+2"
  echo "- logical topology: 6 agents x 2 H800; world size 12; B1; GA4; effective global batch 48"
  echo "- objective: H=1/T=4, decision-state executed-action MC ValueHead, no rank loss"
  echo "- rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
  echo "- init: ${MODEL_PATH}"
  echo "- preprocess cache: ${PREPROCESS_CACHE} (read-only reuse)"
  echo "- W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}/${WANDB_RUN_NAME} (${WANDB_RUN_ID})"
  echo "- output: ${RUN_OUTPUT}"
  echo "- resume: false; fresh optimizer; 20-minute latest checkpoints"
} > "${RUN_ROOT}/README.md"

echo "probe_start=$(date --iso-8601=seconds) job=${SLURM_JOB_ID}"
NODE_MODE=probe srun \
  --het-group=0,1 \
  --gpus-per-task=2 \
  --gpu-bind=closest \
  --cpus-per-task=32 \
  --kill-on-bad-exit=1 \
  bash "${REPO}/experiments/training/sft2/run_dino_grid_ws12_node.sh"

/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/sft2/validate_ws12_heterogeneous_allocation.py" \
  --run-root "${RUN_ROOT}" \
  --job-id "${SLURM_JOB_ID}"

echo "train_start=$(date --iso-8601=seconds) job=${SLURM_JOB_ID}"
NODE_MODE=train srun \
  --het-group=0,1 \
  --gpus-per-task=2 \
  --gpu-bind=closest \
  --cpus-per-task=32 \
  --kill-on-bad-exit=1 \
  bash "${REPO}/experiments/training/sft2/run_dino_grid_ws12_node.sh"

/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/sft2/validate_dino_grid_training_output.py" \
  --output-dir "${RUN_OUTPUT}" \
  --expected-step "${EXPECTED_STEPS}" \
  --expected-world-size 12 \
  --expected-wandb-run-id "${WANDB_RUN_ID}" \
  --result-json "${RUN_ROOT}/completion_validation.json"
echo "finished=$(date --iso-8601=seconds) job=${SLURM_JOB_ID}"
