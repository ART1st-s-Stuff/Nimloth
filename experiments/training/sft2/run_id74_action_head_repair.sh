#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${TRAIN_EXAMPLES_PER_ACTION:?TRAIN_EXAMPLES_PER_ACTION is required}"
: "${VALIDATION_EXAMPLES_PER_ACTION:?VALIDATION_EXAMPLES_PER_ACTION is required}"
: "${SELECTION_SEED:?SELECTION_SEED is required}"
: "${FIT_LEARNING_RATE:?FIT_LEARNING_RATE is required}"
: "${FIT_WEIGHT_DECAY:?FIT_WEIGHT_DECAY is required}"
: "${FIT_MAX_EPOCHS:?FIT_MAX_EPOCHS is required}"
: "${FIT_EARLY_STOPPING_PATIENCE:?FIT_EARLY_STOPPING_PATIENCE is required}"
: "${MINIMUM_VALIDATION_NLL_IMPROVEMENT:?MINIMUM_VALIDATION_NLL_IMPROVEMENT is required}"
: "${MINIMUM_BF16_MEDIAN_SPREAD:?MINIMUM_BF16_MEDIAN_SPREAD is required}"
: "${EXPECTED_WORLD_SIZE:?EXPECTED_WORLD_SIZE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
MODEL=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
DATA=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data
CACHE=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/53_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/cache/preprocess
MODEL_INDEX_SHA=32acf7bf413e8b87f295e816fe3d68c965e0ab196fbf30b32858b52df41cc97e
TRAIN_SHA=d43ada06d66c0b5cafa50e9da8ecc354445ca3b9686d1639b18050a981247b97
VALIDATION_SHA=4c092fb4069fb71ad92bca73566d5f20f572569a093bf4712467ca137615212e
TRAIN_CACHE_SHA=3e501a0ccee9193676d69dd3590ae0d592c4fdee298810df2abff47d9f36a943
VALIDATION_CACHE_SHA=acd10994cff947c365f95da69d81219fde0e97a30a7f574bb395e8169b93da58
RUN_OUT=${OUTPUT_ROOT}/${RUN_NAME}
CONTROL=${OUTPUT_ROOT}/slurm/${RUN_NAME}-${SLURM_JOB_ID}

[[ -x "${PY}" ]]
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=all)" ]]
for nested in external/VAGEN external/VAGEN/verl external/le-wm external/RCDM; do
  [[ -z "$(git -C "${REPO}/${nested}" status --porcelain --untracked-files=all)" ]]
done
[[ "${EXPECTED_WORLD_SIZE}" == "8" ]]
[[ "${TRAIN_EXAMPLES_PER_ACTION}" == "271" ]]
[[ "${VALIDATION_EXAMPLES_PER_ACTION}" == "40" ]]
[[ "${SELECTION_SEED}" == "42002" ]]
[[ "${FIT_LEARNING_RATE}" == "0.0001" ]]
[[ "${FIT_WEIGHT_DECAY}" == "0.0" ]]
[[ "${FIT_MAX_EPOCHS}" == "500" ]]
[[ "${FIT_EARLY_STOPPING_PATIENCE}" == "50" ]]
[[ "${MINIMUM_VALIDATION_NLL_IMPROVEMENT}" == "0.05" ]]
[[ "${MINIMUM_BF16_MEDIAN_SPREAD}" == "0.001" ]]
mkdir -p "${CONTROL}"
[[ ! -e "${CONTROL}/complete.marker" ]]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/project/peilab/atst/flower/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

COMMAND=(
  "${PY}" -m torch.distributed.run
  --standalone --nnodes=1 --nproc_per_node="${EXPECTED_WORLD_SIZE}"
  -m nimloth.training.sft2.action_head_repair_cli
  --model "${MODEL}"
  --train-jsonl "${DATA}/train_terminal_cot_migrated.jsonl"
  --validation-jsonl "${DATA}/val_terminal_cot_migrated.jsonl"
  --cache-root "${CACHE}"
  --output-dir "${RUN_OUT}"
  --expected-model-index-sha256 "${MODEL_INDEX_SHA}"
  --expected-train-sha256 "${TRAIN_SHA}"
  --expected-validation-sha256 "${VALIDATION_SHA}"
  --expected-train-cache-manifest-sha256 "${TRAIN_CACHE_SHA}"
  --expected-validation-cache-manifest-sha256 "${VALIDATION_CACHE_SHA}"
  --model-dtype bfloat16
  --attn-implementation flash_attention_2
  --resume-mode allow-exact
  --git-commit "${EXPECTED_COMMIT}"
  --experiment-purpose "Repair only ID74 action-token LM-head rows before resuming optimizer-free K4 RL calibration"
  --train-examples-per-action "${TRAIN_EXAMPLES_PER_ACTION}"
  --validation-examples-per-action "${VALIDATION_EXAMPLES_PER_ACTION}"
  --selection-seed "${SELECTION_SEED}"
  --fit-learning-rate "${FIT_LEARNING_RATE}"
  --fit-weight-decay "${FIT_WEIGHT_DECAY}"
  --fit-max-epochs "${FIT_MAX_EPOCHS}"
  --fit-early-stopping-patience "${FIT_EARLY_STOPPING_PATIENCE}"
  --minimum-validation-nll-improvement "${MINIMUM_VALIDATION_NLL_IMPROVEMENT}"
  --minimum-bf16-median-spread "${MINIMUM_BF16_MEDIAN_SPREAD}"
  --extraction-batch-size 1
  --max-length 12000
  --latent-token-count 16
  --expected-action-count 8
  --expected-world-size "${EXPECTED_WORLD_SIZE}"
)
printf '%q ' "${COMMAND[@]}" >"${CONTROL}/command.sh"
printf '\n' >>"${CONTROL}/command.sh"
{
  echo "parent_commit=$(git -C "${REPO}" rev-parse HEAD)"
  echo "vagen_commit=$(git -C "${REPO}/external/VAGEN" rev-parse HEAD)"
  echo "verl_commit=$(git -C "${REPO}/external/VAGEN/verl" rev-parse HEAD)"
  echo "lewm_commit=$(git -C "${REPO}/external/le-wm" rev-parse HEAD)"
  echo "rcdm_commit=$(git -C "${REPO}/external/RCDM" rev-parse HEAD)"
} >"${CONTROL}/source_commits.txt"

"${COMMAND[@]}" 2>&1 | tee "${CONTROL}/run.log"

"${PY}" - "${RUN_OUT}" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
summary=json.loads((out/'summary.json').read_text())
assert summary['schema']=='nimloth_id74_action_head_repair_v1'
assert summary['status']=='passed'
assert summary['world_size']==8
assert summary['train_examples']==2168
assert summary['validation_examples']==320
assert summary['examples_per_action']=={'train':271,'validation':40}
assert summary['fit']['validation_nll_improvement_fp32']>=0.05
assert summary['fit']['validation_bfloat16_median_action_spread']>0.001
assert summary['frozen_components']==[
 'qwen_transformer','qwen_vision','all_non_action_lm_head_rows',
 'state_proj','wm_predictor','value_head']
assert (out/'checkpoint'/'action_head_repair.pt').is_file()
assert (out/'train_step_log.csv').is_file()
assert (out/'complete.marker').read_text()=='complete\n'
print(json.dumps({'status':'ALL_OK','summary':summary['fit']},sort_keys=True))
PY

install -m 0644 "${CONTROL}/command.sh" "${CONTROL}/source_commits.txt" "${CONTROL}/run.log" "${RUN_OUT}/"
touch "${CONTROL}/complete.marker"
for source_repo in "${REPO}" "${REPO}/external/VAGEN" "${REPO}/external/VAGEN/verl" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
