#!/usr/bin/env bash

# Run terminal-CoT generation, compact-cache construction, and the DINO-grid
# SFT2 world8 launcher inside one already-held eight-GPU Slurm allocation.

set -euo pipefail

REPO=${REPO:?set REPO to the committed server worktree}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new experiment root}
WANDB_RUN_NAME=${WANDB_RUN_NAME:?set WANDB_RUN_NAME}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/dino_grid_k16_h4.yaml}
MODEL_PATH=${MODEL_PATH:-/project/peilab/atst/nimloth/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged}
SOURCE_TRAIN_JSONL=${SOURCE_TRAIN_JSONL:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c/train_all.jsonl}
SOURCE_VAL_JSONL=${SOURCE_VAL_JSONL:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c/val_all.jsonl}
TRAIN_RECORDS=${TRAIN_RECORDS:-3217}
VAL_RECORDS=${VAL_RECORDS:-355}

DATA_DIR=${RUN_ROOT}/data
CACHE_DIR=${RUN_ROOT}/cache/preprocess
CACHE_METADATA_DIR=${RUN_ROOT}/cache/build_metadata
TRAIN_OUT=${RUN_ROOT}/train
TRAIN_JSONL=${DATA_DIR}/train_terminal_cot.jsonl
VAL_JSONL=${DATA_DIR}/val_terminal_cot.jsonl
CONTROLLER_LOG=${RUN_ROOT}/controller.log
EXPERIMENT_README=${RUN_ROOT}/README.md

if [[ -e "${RUN_ROOT}" ]]; then
  echo "ERROR RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 1
fi
for required in "${CONFIG}" "${MODEL_PATH}/config.json" \
  "${SOURCE_TRAIN_JSONL}" "${SOURCE_VAL_JSONL}"; do
  [[ -s "${required}" ]] || { echo "ERROR missing input: ${required}" >&2; exit 1; }
done
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "ERROR committed worktree does not match EXPECTED_COMMIT" >&2
  exit 1
}
[[ "${SLURM_JOB_NUM_NODES:-0}" == "1" ]] || {
  echo "ERROR pipeline requires one allocated node" >&2
  exit 1
}
VISIBLE_GPU_COUNT=$("${PYTHON_ENV}/bin/python3" -c 'import torch; print(torch.cuda.device_count())')
[[ "${VISIBLE_GPU_COUNT}" == "8" ]] || {
  echo "ERROR pipeline requires exactly eight visible GPUs; got ${VISIBLE_GPU_COUNT}" >&2
  exit 1
}

mkdir -p "${DATA_DIR}" "${CACHE_METADATA_DIR}"
exec > >(tee -a "${CONTROLLER_LOG}") 2>&1

export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:${PATH}
export VIRTUAL_ENV=${PYTHON_ENV}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=false

echo "pipeline_start=$(date --iso-8601=seconds)"
echo "commit=${EXPECTED_COMMIT} job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
echo "run_root=${RUN_ROOT} wandb=nimloth-sft2/${WANDB_RUN_NAME}"

printf '%s\n' \
  "# ${WANDB_RUN_NAME}" \
  "" \
  "- 状态：pipeline 已启动；terminal CoT、preprocess cache、SFT2 依次执行。" \
  "- 代码：dev@${EXPECTED_COMMIT}" \
  "- Slurm：job ${SLURM_JOB_ID}，1 node，8 visible GPUs，node ${SLURM_JOB_NODELIST}" \
  "- W&B：nimloth-sft2/${WANDB_RUN_NAME}" \
  "- 输出：${RUN_ROOT}" \
  "- 初始化模型：${MODEL_PATH}" \
  "- auxiliary warm start：ID33（由 ${CONFIG} 固定）；新 optimizer，不 resume ID46。" \
  "- 原始数据：train ${SOURCE_TRAIN_JSONL} (${TRAIN_RECORDS})；val ${SOURCE_VAL_JSONL} (${VAL_RECORDS})。" \
  "- terminal CoT：temperature=0，top_p=1.0，top_k=-1，do_sample=false，n=1，max_reasoning_tokens=128，seed=42，max_pixels=602112，flash_attention_2。" \
  "- SFT2：2 epochs，world_size=8，per-rank batch_size=1，gradient_accumulation=8，history_size=4；history_size 不是 planning.horizon。" \
  "- 调参：Qwen LLM freeze，vision full，vision EMA=0.999；grid EMA=0.99。" \
  "- loss：CE=1，WM=0.1->1，DINO=0.5，value=1，SIGReg=0.1。" \
  "- checkpoint：每20分钟；选择指标 val_wm_mse。" \
  "- 训练数据与 preprocess cache 均在本 run 内新建，旧 fixed-terminal cache 不复用。" \
  > "${EXPERIMENT_README}"

echo "phase=terminal_cot_train_start"

generate_terminal_cot() {
  local source_jsonl=$1
  local output_jsonl=$2
  "${PYTHON_ENV}/bin/python3" \
    "${REPO}/experiments/training/sft2/generate_terminal_cot.py" \
    --model "${MODEL_PATH}" \
    --input-jsonl "${source_jsonl}" \
    --output-jsonl "${output_jsonl}" \
    --max-pixels 602112 \
    --max-reasoning-tokens 128 \
    --temperature 0 \
    --top-p 1.0 \
    --top-k -1 \
    --no-do-sample \
    --n 1 \
    --seed 42 \
    --attn-implementation flash_attention_2
}

generate_terminal_cot "${SOURCE_TRAIN_JSONL}" "${TRAIN_JSONL}"
echo "phase=terminal_cot_val_start"
generate_terminal_cot "${SOURCE_VAL_JSONL}" "${VAL_JSONL}"
[[ "$(wc -l < "${TRAIN_JSONL}")" -eq "${TRAIN_RECORDS}" ]]
[[ "$(wc -l < "${VAL_JSONL}")" -eq "${VAL_RECORDS}" ]]
[[ -s "${TRAIN_JSONL}.manifest.json" && -s "${VAL_JSONL}.manifest.json" ]]
echo "terminal_cot_data=ready train=${TRAIN_RECORDS} val=${VAL_RECORDS}"

echo "phase=preprocess_cache_start"
"${PYTHON_ENV}/bin/python3" \
  "${REPO}/experiments/training/sft2/build_preprocess_cache.py" \
  --config "${CONFIG}" \
  --model "${MODEL_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${CACHE_METADATA_DIR}" \
  --preprocess-cache-dir "${CACHE_DIR}" \
  --preprocess-cache-processor-source "${MODEL_PATH}" \
  --preprocess-workers 16
[[ -s "${CACHE_DIR}/train/manifest.json" ]]
[[ -s "${CACHE_DIR}/val/manifest.json" ]]
echo "preprocess_cache=ready dir=${CACHE_DIR}"

echo "phase=sft2_train_start"
REPO=${REPO} \
CONFIG=${CONFIG} \
MODEL_PATH=${MODEL_PATH} \
TRAIN_JSONL=${TRAIN_JSONL} \
VAL_JSONL=${VAL_JSONL} \
PREPROCESS_CACHE_DIR_OVERRIDE=${CACHE_DIR} \
OUTPUT_DIR=${TRAIN_OUT} \
WANDB_RUN_NAME=${WANDB_RUN_NAME} \
WANDB_PROJECT_NAME=nimloth-sft2 \
NPROC_PER_NODE=8 \
BATCH_SIZE=1 \
GRAD_ACCUM=8 \
EXTRA_TRAIN_ARGS="--preprocess-cache-processor-source ${MODEL_PATH}" \
bash "${REPO}/experiments/training/sft2/train_dino_grid_world8.sh"

echo "pipeline_done=$(date --iso-8601=seconds)"
