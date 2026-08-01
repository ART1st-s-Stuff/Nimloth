#!/usr/bin/env bash
set -euo pipefail

: "${HOLD_JOB_ID:?}"
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

test "${HOLD_JOB_ID}" = "500950"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
test -s "${RUN_ROOT}/preflight.json"
test -f "${PREPROCESS_CACHE}/cache_done.flag"
test ! -e "${RUN_OUTPUT}"

export MASTER_ADDR=dgx-26
export MASTER_PORT=$((20000 + HOLD_JOB_ID % 20000))
export REPO EXPECTED_COMMIT RUN_ROOT RUN_OUTPUT MODEL_PATH TRAIN_JSONL VAL_JSONL
export PREPROCESS_CACHE WANDB_ENTITY WANDB_PROJECT_NAME WANDB_RUN_NAME WANDB_RUN_ID
export MASTER_ADDR MASTER_PORT
CONTROLLER_LOG="${RUN_OUTPUT}.controller_${HOLD_JOB_ID}.log"
exec > >(tee -a "${CONTROLLER_LOG}") 2>&1

run_agents() {
  local node_mode=$1
  NODE_MODE=${node_mode} srun --jobid="${HOLD_JOB_ID}" --overlap \
    --het-group=0 --nodes=3 --ntasks=12 --ntasks-per-node=4 \
    --gpus-per-task=1 --gpu-bind=closest --cpus-per-task=20 \
    --kill-on-bad-exit=1 bash "${REPO}/experiments/training/sft2/run_dino_grid_ws16_single_gpu_agent.sh" \
    : --het-group=1 --nodes=1 --ntasks=2 --ntasks-per-node=2 \
    --gpus-per-task=1 --gpu-bind=closest --cpus-per-task=20 \
    --kill-on-bad-exit=1 bash "${REPO}/experiments/training/sft2/run_dino_grid_ws16_single_gpu_agent.sh" \
    : --het-group=2 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-task=1 --gpu-bind=closest --cpus-per-task=20 \
    --kill-on-bad-exit=1 bash "${REPO}/experiments/training/sft2/run_dino_grid_ws16_single_gpu_agent.sh"
}

echo "probe_start=$(date --iso-8601=seconds) hold=${HOLD_JOB_ID}"
run_agents probe
/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/sft2/validate_ws16_444211_allocation.py" \
  --run-root "${RUN_ROOT}" --job-id "${HOLD_JOB_ID}"

echo "train_start=$(date --iso-8601=seconds) hold=${HOLD_JOB_ID}"
run_agents train
/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/sft2/validate_dino_grid_training_output.py" \
  --output-dir "${RUN_OUTPUT}" --expected-step "${EXPECTED_STEPS}" \
  --expected-world-size 16 --expected-wandb-run-id "${WANDB_RUN_ID}" \
  --result-json "${RUN_ROOT}/completion_validation.json"
