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

DATA=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data
TRAIN_JSONL=${DATA}/train_terminal_cot_migrated.jsonl
VAL_JSONL=${DATA}/val_terminal_cot_migrated.jsonl
ID60=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/60_id176_sft1_frozen_goal_probe_early4
ID61=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-24/61_id75_action_outcome_audit_retry1
STATE_CACHE=${ID60}/frozen_state_cache.npz
STATE_METADATA=${ID60}/frozen_state_cache_metadata.json
ID60_RESULT=${ID60}/result.json
ID61_RESULT=${ID61}/result.json
OUT=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-24/71_frozen_state_action_outcome_probe
RUN_WANDB_PROJECT=nimloth-recon
RUN_WANDB_NAME=71_frozen_state_action_outcome_probe
RUN_WANDB_ID=nimloth-recon-id71-action-outcome-probe

for spec in \
  "${TRAIN_JSONL}:d43ada06d66c0b5cafa50e9da8ecc354445ca3b9686d1639b18050a981247b97" \
  "${VAL_JSONL}:4c092fb4069fb71ad92bca73566d5f20f572569a093bf4712467ca137615212e" \
  "${STATE_CACHE}:0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6" \
  "${STATE_METADATA}:b25163d390930d1ccdc172e4f4401a97cbea3dba561a8ec25ef33b9a09911682" \
  "${ID60_RESULT}:37243c37e265691cc0cd3acdbc03a35241661a765ce8efc5fb2b6a7995bcd0ea" \
  "${ID61_RESULT}:bace6fcbc5ec85fdeed59e6ba30ff61b58bbe382f88af51f9dd591a8105a28e4"; do
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
# ID71 frozen-state action-outcome predictability probe

- runtime commit: ${EXPECTED_COMMIT}
- trainable: matched action-specific linear diagnostic readouts only
- frozen: ID176 actor/Qwen/vision, SFT1 projector, DINO and ID75 WM
- inputs: full flattened K16 state or matched DINO; fit-only per-dimension standardization
- labels: exact archived per-step environment success feedback
- split: pre-RL archive train/external, ID60 exact-image decontamination, initial-image-grouped inner selection
- no projector/WM update and no production checkpoint
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
export TOKENIZERS_PARALLELISM=false

exec "${PY}" -m nimloth.eval.action_outcome_predictability_probe \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --state-cache "${STATE_CACHE}" \
  --state-cache-metadata "${STATE_METADATA}" \
  --id60-result "${ID60_RESULT}" \
  --id61-result "${ID61_RESULT}" \
  --output-dir "${OUT}" \
  --learning-rate 3e-3 \
  --weight-decay 1e-2 \
  --max-epochs 300 \
  --patience 30 \
  --bootstrap-draws 10000 \
  --seed 42071 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${RUN_WANDB_PROJECT}" \
  --wandb-run-name "${RUN_WANDB_NAME}" \
  --wandb-run-id "${RUN_WANDB_ID}"
