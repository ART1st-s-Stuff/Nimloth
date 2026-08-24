#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "commit mismatch: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
  exit 2
}
for source_repo in \
  "${REPO}" "${REPO}/external/VAGEN" "${REPO}/external/VAGEN/verl" \
  "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]] || {
    echo "production source must be clean: ${source_repo}" >&2
    exit 2
  }
done
[[ -x "${PY}" ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]]
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${GPU_NAME}" == *H800* ]] || { echo "expected H800, got ${GPU_NAME}" >&2; exit 2; }

SFT1=${ROOT}/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged
ACTOR=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint
DATA=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data
TRAIN_JSONL=${DATA}/train_terminal_cot_migrated.jsonl
VAL_JSONL=${DATA}/val_terminal_cot_migrated.jsonl
ASSETS=${REPO}/external/VAGEN/vagen/envs/navigation/assets
DINO_CACHE=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe
PROBE_OUT=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/60_id176_sft1_frozen_goal_probe_early4
CANARY_OUT=${ROOT}/outputs/experiments/training/sft2/2026-08-23/75_frozen_sft1_residual_t1_canary_early4
PROBE_WANDB_PROJECT=nimloth-recon
PROBE_WANDB_NAME=60_id176_sft1_frozen_goal_probe_early4_k16
PROBE_WANDB_ID=nimloth-recon-id60-frozen-state-goal-probe
CANARY_WANDB_PROJECT=nimloth-sft2
CANARY_WANDB_NAME=75_frozen_sft1_residual_t1_canary_early4_k16
CANARY_WANDB_ID=nimloth-sft2-id75-frozen-sft1-residual-t1-canary

for path in \
  "${SFT1}/config.json" "${SFT1}/slot_projector.pt" \
  "${ACTOR}/config.json" "${ACTOR}/model.safetensors.index.json" \
  "${TRAIN_JSONL}" "${VAL_JSONL}" \
  "${ASSETS}/base_train.json" "${ASSETS}/common_sense_train.json" \
  "${ASSETS}/long_horizon_train.json" \
  "${DINO_CACHE}/train/dino_grid4/manifest.json" \
  "${DINO_CACHE}/val/dino_grid4/manifest.json"; do
  [[ -r "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${PROBE_OUT}")" "$(dirname "${CANARY_OUT}")"
[[ ! -e "${PROBE_OUT}" ]] || { echo "fresh PROBE_OUT exists: ${PROBE_OUT}" >&2; exit 2; }
[[ ! -e "${CANARY_OUT}" ]] || { echo "fresh CANARY_OUT exists: ${CANARY_OUT}" >&2; exit 2; }

set -a
source /project/peilab/atst/flower/.env
set +a
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir "${PROBE_OUT}"
cat >"${PROBE_OUT}/README.md" <<EOF
# ID60 frozen ID176+SFT1 goal probe

- runtime commit: ${EXPECTED_COMMIT}
- data: pre-RL ID52 train/validation; exact early steps 0--3 and persisted real CoT/terminal CoT
- split: archive-level pre-RL train/validation files; row-level config/seed task identity is unavailable; exact-image grouped inner split and cross-split exclusion
- trainable: diagnostic linear state and matched DINO goal readouts only
- frozen: ID176 actor/Qwen/vision, SFT1 SharedSlotProjector and DINO
- output: immutable float32 state cache, diagnostic probe weights, result and summary
- resume/overwrite: forbidden; any retry needs fresh output and W&B identity
EOF
export WANDB_PROJECT="${PROBE_WANDB_PROJECT}"
export WANDB_NAME="${PROBE_WANDB_NAME}"
export WANDB_RUN_ID="${PROBE_WANDB_ID}"
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export WANDB_DIR="${PROBE_OUT}/wandb"
"${PY}" -m nimloth.eval.frozen_state_goal_probe \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --asset-root "${ASSETS}" \
  --dino-grid-cache-root "${DINO_CACHE}" \
  --sft1-checkpoint "${SFT1}" \
  --actor-checkpoint "${ACTOR}" \
  --output-dir "${PROBE_OUT}" \
  --max-step-index 3 \
  --batch-size 8 \
  --max-length 12000 \
  --max-pixels 100352 \
  --probe-max-epochs 500 \
  --probe-patience 50 \
  --seed 42060 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${PROBE_WANDB_PROJECT}" \
  --wandb-run-name "${PROBE_WANDB_NAME}" \
  --wandb-run-id "${PROBE_WANDB_ID}"

for product in frozen_state_cache.npz frozen_state_cache_metadata.json result.json; do
  [[ -s "${PROBE_OUT}/${product}" ]] || { echo "ID60 missing ${product}" >&2; exit 2; }
done
[[ ! -e "${CANARY_OUT}" ]] || { echo "fresh CANARY_OUT appeared early" >&2; exit 2; }
mkdir "${CANARY_OUT}"
cat >"${CANARY_OUT}/README.md" <<EOF
# ID75 frozen-SFT1 Residual-T1 canary

- runtime commit: ${EXPECTED_COMMIT}
- source: immutable ID60 frozen state cache
- trainable: zero-copy-initialized ResidualTemporalSpatialGridPredictor only
- frozen/absent: actor/Qwen/vision/SFT1 projector/DINO; no ValueHead, policy, planner or RL
- loss: fixed-state next-state MSE only; raw DINO training loss is exactly zero
- gate: each validation action with N>=20 must beat copy, plus macro/overall/std/DINO checks
- checkpoint: diagnostic canary only; downstream use is not authorized
- resume/overwrite: forbidden; any retry needs fresh output and W&B identity
EOF
export WANDB_PROJECT="${CANARY_WANDB_PROJECT}"
export WANDB_NAME="${CANARY_WANDB_NAME}"
export WANDB_RUN_ID="${CANARY_WANDB_ID}"
export WANDB_DIR="${CANARY_OUT}/wandb"
"${PY}" -m nimloth.training.sft2.residual_t1_canary \
  --state-cache "${PROBE_OUT}/frozen_state_cache.npz" \
  --state-cache-metadata "${PROBE_OUT}/frozen_state_cache_metadata.json" \
  --probe-result "${PROBE_OUT}/result.json" \
  --output-dir "${CANARY_OUT}" \
  --batch-size 64 \
  --max-epochs 20 \
  --patience 4 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --minimum-primary-count 20 \
  --raw-dino-loss-weight 0 \
  --seed 42075 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${CANARY_WANDB_PROJECT}" \
  --wandb-run-name "${CANARY_WANDB_NAME}" \
  --wandb-run-id "${CANARY_WANDB_ID}"
