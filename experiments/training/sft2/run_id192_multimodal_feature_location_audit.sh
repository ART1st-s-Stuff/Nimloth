#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
: "${REPO:?REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
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

ACTOR=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint
DATA=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data
TRAIN_JSONL=${DATA}/train_terminal_cot_migrated.jsonl
VAL_JSONL=${DATA}/val_terminal_cot_migrated.jsonl
ID60=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/60_id176_sft1_frozen_goal_probe_early4
ID191=${ROOT}/outputs/experiments/training/sft2/2026-08-24/191_state_interface_direction_canary_retry1
STATE_CACHE=${ID60}/frozen_state_cache.npz
STATE_METADATA=${ID60}/frozen_state_cache_metadata.json
ID60_RESULT=${ID60}/result.json
ID191_RESULT=${ID191}/result.json
ID191_HIDDEN=${ID191}/frozen_same_generation_hidden.npz
OUT=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-24/192_frozen_multimodal_feature_location_audit
RUN_WANDB_PROJECT=nimloth-recon
RUN_WANDB_NAME=192_frozen_multimodal_feature_location_audit
RUN_WANDB_ID=nimloth-recon-id192-feature-location-audit

for spec in \
  "${TRAIN_JSONL}:d43ada06d66c0b5cafa50e9da8ecc354445ca3b9686d1639b18050a981247b97" \
  "${VAL_JSONL}:4c092fb4069fb71ad92bca73566d5f20f572569a093bf4712467ca137615212e" \
  "${STATE_CACHE}:0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6" \
  "${STATE_METADATA}:b25163d390930d1ccdc172e4f4401a97cbea3dba561a8ec25ef33b9a09911682" \
  "${ID60_RESULT}:37243c37e265691cc0cd3acdbc03a35241661a765ce8efc5fb2b6a7995bcd0ea" \
  "${ID191_RESULT}:1e1307c24b0d0187191476c87dee570ad261b98ee51facfd77cb38aab35006bb" \
  "${ID191_HIDDEN}:e676a870fc200175f761691bd449aac4fa4d529471dc24f17f3a9358b1fddc93"; do
  path=${spec%%:*}; expected=${spec##*:}
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "input hash mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
done
mkdir -p "$(dirname "${OUT}")"
[[ ! -e "${OUT}" ]] || { echo "fresh OUT exists: ${OUT}" >&2; exit 2; }
mkdir "${OUT}"
cat >"${OUT}/README.md" <<EOF
# ID192 frozen multimodal feature-location audit

- runtime commit: ${EXPECTED_COMMIT}
- purpose: locate visual collision and instruction-goal evidence before designing another state interface
- data: exact pre-RL ID52/ID60 train-validation split, exact-image decontamination, actual archived observation-conditioned CoT
- same-forward capture: current-image Qwen visual output before LLM, current-image final LLM tokens, exact instruction embedding/final tokens, and final K16 identity
- trainable: fresh diagnostic linear readouts only
- frozen: ID176 actor/Qwen/vision, SFT1 projector, DINO and every WM/ValueHead/policy/planner/RL module
- output: float32 same-forward feature cache, readouts, metrics and direction decision; no deployable checkpoint
- resume/overwrite: forbidden; any failure requires a fresh output and W&B identity
EOF

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_PROJECT="${RUN_WANDB_PROJECT}"
export WANDB_NAME="${RUN_WANDB_NAME}"
export WANDB_RUN_ID="${RUN_WANDB_ID}"
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export WANDB_DIR="${OUT}/wandb"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "${PY}" -m nimloth.eval.multimodal_feature_location_audit \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --state-cache "${STATE_CACHE}" \
  --state-cache-metadata "${STATE_METADATA}" \
  --id60-result "${ID60_RESULT}" \
  --id191-result "${ID191_RESULT}" \
  --id191-hidden-cache "${ID191_HIDDEN}" \
  --actor-checkpoint "${ACTOR}" \
  --output-dir "${OUT}" \
  --encode-batch-size 8 \
  --max-length 12000 \
  --max-pixels 100352 \
  --probe-max-epochs 300 \
  --probe-patience 30 \
  --bootstrap-draws 10000 \
  --seed 42192 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${RUN_WANDB_PROJECT}" \
  --wandb-run-name "${RUN_WANDB_NAME}" \
  --wandb-run-id "${RUN_WANDB_ID}"
