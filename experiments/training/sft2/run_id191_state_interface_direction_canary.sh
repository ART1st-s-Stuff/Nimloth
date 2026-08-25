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

SFT1=${ROOT}/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged
ACTOR=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint
DATA=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data
TRAIN_JSONL=${DATA}/train_terminal_cot_migrated.jsonl
VAL_JSONL=${DATA}/val_terminal_cot_migrated.jsonl
ID60=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/60_id176_sft1_frozen_goal_probe_early4
ID71=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-24/71_frozen_state_action_outcome_probe
STATE_CACHE=${ID60}/frozen_state_cache.npz
STATE_METADATA=${ID60}/frozen_state_cache_metadata.json
ID60_RESULT=${ID60}/result.json
ID71_RESULT=${ID71}/result.json
OUT=${ROOT}/outputs/experiments/training/sft2/2026-08-24/191_state_interface_direction_canary_retry1
RUN_WANDB_PROJECT=nimloth-sft2
RUN_WANDB_NAME=191_state_interface_direction_canary_retry1
RUN_WANDB_ID=nimloth-sft2-id191-state-interface-canary-retry1

for spec in \
  "${TRAIN_JSONL}:d43ada06d66c0b5cafa50e9da8ecc354445ca3b9686d1639b18050a981247b97" \
  "${VAL_JSONL}:4c092fb4069fb71ad92bca73566d5f20f572569a093bf4712467ca137615212e" \
  "${STATE_CACHE}:0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6" \
  "${STATE_METADATA}:b25163d390930d1ccdc172e4f4401a97cbea3dba561a8ec25ef33b9a09911682" \
  "${ID60_RESULT}:37243c37e265691cc0cd3acdbc03a35241661a765ce8efc5fb2b6a7995bcd0ea" \
  "${ID71_RESULT}:2e2e1675317d252bc6e503ac78507328c81bf1925aceb487ed8e506f8b70c113" \
  "${SFT1}/slot_projector.pt:340d90a84a17f7aba3525f2f49e20921fd4f73a6534149587de2b3c875542ce0"; do
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
# ID191 state-interface direction canary

- runtime commit: ${EXPECTED_COMMIT}
- authorization: diagnostic direction canary only; no old checkpoint repair or downstream use
- data: immutable pre-RL ID60 state/DINO cache plus same-generation ID176 hidden recomputed from actual archived CoT
- frozen: ID176 actor/Qwen/vision, SFT1 projector, DINO, all old WM/ValueHead/policy/planner/RL
- trainable: zero-initialized rank-64 hidden-to-state residual bounded to 10%, training-only goal/outcome heads, fresh diagnostic readouts
- state semantics: one unified K16 visual-goal state; success is supervision only and never a deployment input
- selection: exact-initial-image grouped inner split; external validation preserves ID60 exact-image decontamination
- output: diagnostic adapter, hidden cache, external candidate states, full gates; downstream use forbidden
- resume/overwrite: forbidden; any retry requires fresh output and W&B identity
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

exec "${PY}" -m nimloth.training.sft2.id191_state_interface_canary \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --state-cache "${STATE_CACHE}" \
  --state-cache-metadata "${STATE_METADATA}" \
  --id60-result "${ID60_RESULT}" \
  --id71-result "${ID71_RESULT}" \
  --sft1-checkpoint "${SFT1}" \
  --actor-checkpoint "${ACTOR}" \
  --output-dir "${OUT}" \
  --batch-size 64 \
  --encode-batch-size 8 \
  --max-length 12000 \
  --max-pixels 100352 \
  --adapter-rank 64 \
  --max-residual-fraction 0.1 \
  --max-epochs 12 \
  --patience 3 \
  --learning-rate 3e-4 \
  --weight-decay 1e-3 \
  --anchor-weight 0.25 \
  --probe-max-epochs 300 \
  --probe-patience 30 \
  --bootstrap-draws 10000 \
  --seed 42191 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${RUN_WANDB_PROJECT}" \
  --wandb-run-name "${RUN_WANDB_NAME}" \
  --wandb-run-id "${RUN_WANDB_ID}"
