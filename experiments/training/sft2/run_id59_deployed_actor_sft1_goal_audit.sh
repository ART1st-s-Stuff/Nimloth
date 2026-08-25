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
RUN_OUT=${ROOT}/outputs/experiments/evaluation/state_alignment/2026-08-23/59_id176_sft1_goal_audit_train3211_val355
RUN_WANDB_PROJECT=nimloth-recon
RUN_WANDB_NAME=59_id176_sft1_goal_audit_train3211_val355_k16
RUN_WANDB_ID=nimloth-recon-id59-id176-sft1-goal-audit

for path in \
  "${SFT1}/config.json" "${SFT1}/slot_projector.pt" \
  "${ACTOR}/config.json" "${ACTOR}/state_proj.pt" \
  "${TRAIN_JSONL}" "${VAL_JSONL}" \
  "${ASSETS}/base_train.json" "${ASSETS}/common_sense_train.json" \
  "${ASSETS}/long_horizon_train.json" \
  "${DINO_CACHE}/train/dino_grid4/manifest.json" \
  "${DINO_CACHE}/val/dino_grid4/manifest.json"; do
  [[ -r "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
[[ ! -e "${RUN_OUT}" ]] || { echo "fresh RUN_OUT exists: ${RUN_OUT}" >&2; exit 2; }
mkdir "${RUN_OUT}"
cat >"${RUN_OUT}/README.md" <<EOF
# ID59 deployed actor + SFT1 projector visual/goal audit

- purpose: check whether the deployed ID176 actor remains compatible with the SFT1 visual-goal state anchor
- source commit: ${EXPECTED_COMMIT}
- W&B: ${RUN_WANDB_PROJECT}/${RUN_WANDB_NAME} (${RUN_WANDB_ID})
- data: all 3,211 archived pre-RL train trajectories as retrieval gallery and all 355 archived validation trajectories as queries; first decision state only
- goal labels: exact archived instruction matched to unique targetObjectType in the actual source NavigationEnvConfig asset
- exact-image leakage: excluded from train-to-validation retrieval
- CoT boundary: actual archived observation-conditioned assistant responses; controlled checkpoint forward, not newly generated ID176 behavior-time CoT
- freeze boundary: all actor/backbone/vision/projector/DINO modules frozen; training is forbidden; no optimizer, backward, generation, update, resume, or checkpoint
- output: result.json, summary.html, float32 state_goal_audit.npz
- resource: normal 1xH800, 30-minute hard walltime
EOF

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_PROJECT="${RUN_WANDB_PROJECT}"
export WANDB_NAME="${RUN_WANDB_NAME}"
export WANDB_RUN_ID="${RUN_WANDB_ID}"
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${RUN_OUT}/wandb"

exec "${PY}" -m nimloth.eval.deployed_actor_sft1_goal_audit \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --asset-root "${ASSETS}" \
  --dino-grid-cache-root "${DINO_CACHE}" \
  --sft1-checkpoint "${SFT1}" \
  --actor-checkpoint "${ACTOR}" \
  --output-dir "${RUN_OUT}" \
  --batch-size 8 \
  --max-length 12000 \
  --max-pixels 100352 \
  --git-commit "${EXPECTED_COMMIT}" \
  --wandb-project "${RUN_WANDB_PROJECT}" \
  --wandb-run-name "${RUN_WANDB_NAME}" \
  --wandb-run-id "${RUN_WANDB_ID}"
