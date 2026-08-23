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
    git -C "${source_repo}" status --short >&2
    exit 2
  }
done
[[ -x "${PY}" ]] || { echo "missing server Python: ${PY}" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
  echo "ID58 requires exactly one Slurm-visible GPU" >&2
  exit 2
}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${GPU_NAME}" == *H800* ]] || { echo "expected H800, got ${GPU_NAME}" >&2; exit 2; }

SFT1=${ROOT}/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged
ID74=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
VAL_JSONL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data/val_terminal_cot_migrated.jsonl
DINO_CACHE=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe
RUN_OUT=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/58_sft1_id74_checkpoint_state_matrix_val96_early4
WANDB_PROJECT=nimloth-recon
WANDB_RUN_NAME=58_sft1id74_state_matrix_val96_early4_k16
WANDB_RUN_ID=nimloth-recon-id58-sft1id74-state-matrix

for path in \
  "${SFT1}/config.json" "${SFT1}/slot_projector.pt" \
  "${ID74}/config.json" "${ID74}/state_proj.pt" "${ID74}/vision_ema.pt" \
  "${ID74}/wm_predictor/predictor.pt" "${ID74}/value_head/value_head.pt" \
  "${VAL_JSONL}" "${DINO_CACHE}/val/manifest.json" "${DINO_CACHE}/val/dino_grid4/manifest.json"; do
  [[ -r "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
[[ ! -e "${RUN_OUT}" ]] || { echo "fresh RUN_OUT already exists: ${RUN_OUT}" >&2; exit 2; }
mkdir "${RUN_OUT}"
cat >"${RUN_OUT}/README.md" <<EOF
# ID58 SFT1 / ID74 checkpoint state matrix

- purpose: read-only isolation of backbone/vision, projector, and ID74 vision-EMA drift
- git commit: ${EXPECTED_COMMIT}
- W&B project: ${WANDB_PROJECT}
- W&B run: ${WANDB_RUN_NAME} (${WANDB_RUN_ID})
- data: ID74 pre-RL validation JSONL; deterministic early-step (0--3) transitions, 32 each from Base/Common/Long Horizon, at most one transition per trajectory
- checkpoint matrix: SFT1 and ID74-online/EMA backbones crossed with SFT1 and ID74 projectors
- downstream readouts: frozen ID74 one-step WM and ValueHead; cross-component cells are compatibility diagnostics only
- state semantics: actual recorded observation-conditioned CoT; original-observation cached DINO
- freeze boundary: all modules are frozen; training is forbidden; no optimizer, backward, update, resume, or checkpoint write
- output: result.json, summary.html, and float32 matrix_states.npz
- resource: normal 1xH800, Slurm walltime 01:45:00
EOF

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DIR="${RUN_OUT}/wandb"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

exec "${PY}" -m nimloth.eval.sft_checkpoint_state_matrix \
  --val-jsonl "${VAL_JSONL}" \
  --dino-grid-cache-root "${DINO_CACHE}" \
  --sft1-checkpoint "${SFT1}" \
  --id74-checkpoint "${ID74}" \
  --output-dir "${RUN_OUT}" \
  --samples-per-source 32 \
  --max-step-index 3 \
  --batch-size 2 \
  --max-length 12000 \
  --max-pixels 100352 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-run-id "${WANDB_RUN_ID}"
