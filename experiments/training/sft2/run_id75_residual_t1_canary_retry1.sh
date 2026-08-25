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

PROBE_OUT=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/60_id176_sft1_frozen_goal_probe_early4
STATE_CACHE=${PROBE_OUT}/frozen_state_cache.npz
STATE_METADATA=${PROBE_OUT}/frozen_state_cache_metadata.json
PROBE_RESULT=${PROBE_OUT}/result.json
EXPECTED_CACHE_SHA=0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6
EXPECTED_METADATA_SHA=b25163d390930d1ccdc172e4f4401a97cbea3dba561a8ec25ef33b9a09911682
EXPECTED_RESULT_SHA=37243c37e265691cc0cd3acdbc03a35241661a765ce8efc5fb2b6a7995bcd0ea
CANARY_OUT=${ROOT}/outputs/experiments/training/sft2/2026-08-23/75_frozen_sft1_residual_t1_canary_early4_retry1
CANARY_WANDB_PROJECT=nimloth-sft2
CANARY_WANDB_NAME=75_frozen_sft1_residual_t1_canary_early4_k16_retry1
CANARY_WANDB_ID=nimloth-sft2-id75-frozen-sft1-residual-t1-canary-retry1

for spec in \
  "${STATE_CACHE}:${EXPECTED_CACHE_SHA}" \
  "${STATE_METADATA}:${EXPECTED_METADATA_SHA}" \
  "${PROBE_RESULT}:${EXPECTED_RESULT_SHA}"; do
  path=${spec%%:*}; expected=${spec##*:}
  [[ -s "${path}" ]] || { echo "missing ID60 input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "ID60 input hash mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
done
mkdir -p "$(dirname "${CANARY_OUT}")"
[[ ! -e "${CANARY_OUT}" ]] || { echo "fresh CANARY_OUT exists: ${CANARY_OUT}" >&2; exit 2; }
mkdir "${CANARY_OUT}"
cat >"${CANARY_OUT}/README.md" <<EOF
# ID75 frozen-SFT1 Residual-T1 canary retry1

- runtime commit: ${EXPECTED_COMMIT}
- retry reason: Job 528931 completed ID60, then failed before ID75 because the dated parent output directory was absent (E0148)
- source: immutable validated ID60 cache SHA256 ${EXPECTED_CACHE_SHA}
- trainable: zero-copy-initialized ResidualTemporalSpatialGridPredictor only
- frozen/absent: actor/Qwen/vision/SFT1 projector/DINO; no ValueHead, policy, planner or RL
- loss: fixed-state next-state MSE only; raw DINO training loss is exactly zero
- gate: each validation action with N>=20 must beat copy, plus macro/overall/std/DINO checks
- checkpoint: diagnostic canary only; downstream use is not authorized
- resume/overwrite: forbidden; any further retry needs fresh output and W&B identity
EOF

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_PROJECT="${CANARY_WANDB_PROJECT}"
export WANDB_NAME="${CANARY_WANDB_NAME}"
export WANDB_RUN_ID="${CANARY_WANDB_ID}"
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export WANDB_DIR="${CANARY_OUT}/wandb"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

exec "${PY}" -m nimloth.training.sft2.residual_t1_canary \
  --state-cache "${STATE_CACHE}" \
  --state-cache-metadata "${STATE_METADATA}" \
  --probe-result "${PROBE_RESULT}" \
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
