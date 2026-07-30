#!/usr/bin/env bash
#SBATCH --job-name=id56-align-recon
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
PYTHON=${ROOT}/.venv-vagen-main/bin/python3
WANDB_ENV=/project/peilab/atst/flower/.env

: "${WORKTREE:?set WORKTREE to the clean remote worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact evaluator commit}"
: "${ID56_DINO_GRID_CACHE:?set ID56_DINO_GRID_CACHE to the completed cache root}"
: "${ID56_DINO_GRID_CFM_CHECKPOINT:?set ID56_DINO_GRID_CFM_CHECKPOINT to ID48 best.pt}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new run-owned directory}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME to the ID49 run identity}"

SFT2_CHECKPOINT=${SFT2_CHECKPOINT:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-29/sft2/56_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16_px100352/train_ws16/epoch_002}
VAL_JSONL=${VAL_JSONL:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data/val_terminal_cot_migrated.jsonl}
OLD_DINO_GRID_CACHE=${OLD_DINO_GRID_CACHE:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/cache_sft1e5_dinogrid16x1024_all3217_ws8/dino_grid_cache/val}
QWEN_CACHE=${QWEN_CACHE:-${ROOT}/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/query_state_ablation/12_qwenpositive_stateadapter_all3217_s10000_b128/positive_cache/val}
OLD_DINO_GRID_CFM_CHECKPOINT=${OLD_DINO_GRID_CFM_CHECKPOINT:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt}
QWEN_CFM_CHECKPOINT=${QWEN_CFM_CHECKPOINT:-${ROOT}/outputs/experiments/representation_ablation/2026-07-07/token_cfm_decoder_full_b32_e10_lr1em4_rollout5_467041/vit_token_cfm_low_lr_cont_b32_e10_lr1em5_from_dropcfg_467261/vit_token_cfm_low_lr_best_val.pt}
SELECTIONS=${SELECTIONS:-${WORKTREE}/configs/eval/reconstruction/id56_wm_predicted_diverse40.json}

actual_commit=$(git -C "${WORKTREE}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "commit mismatch: expected=${EXPECTED_COMMIT}, actual=${actual_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${WORKTREE}" status --porcelain)" ]]; then
  echo "remote worktree is dirty" >&2
  git -C "${WORKTREE}" status --short >&2
  exit 2
fi
if [[ ! -f "${WORKTREE}/external/le-wm/module.py" ]]; then
  echo "external/le-wm is not initialized in ${WORKTREE}" >&2
  exit 2
fi
for path in \
  "${PYTHON}" \
  "${SFT2_CHECKPOINT}/model.safetensors.index.json" \
  "${SFT2_CHECKPOINT}/state_proj.pt" \
  "${SFT2_CHECKPOINT}/wm_predictor/predictor.pt" \
  "${SFT2_CHECKPOINT}/value_head/value_head.pt" \
  "${VAL_JSONL}" \
  "${OLD_DINO_GRID_CACHE}/manifest.json" \
  "${QWEN_CACHE}/manifest.json" \
  "${ID56_DINO_GRID_CACHE}/val/manifest.json" \
  "${OLD_DINO_GRID_CFM_CHECKPOINT}" \
  "${ID56_DINO_GRID_CFM_CHECKPOINT}" \
  "${QWEN_CFM_CHECKPOINT}" \
  "${SELECTIONS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 2
  fi
done

explicit_wandb_project=nimloth-recon
if [[ -f "${WANDB_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${WANDB_ENV}"
  set +a
fi
export WANDB_PROJECT=${explicit_wandb_project}
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing after loading ${WANDB_ENV}" >&2
  exit 2
fi

export PYTHONPATH=${WORKTREE}/src:${WORKTREE}/external/le-wm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}

"${PYTHON}" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GPU, got {torch.cuda.device_count()}")
p = torch.cuda.get_device_properties(0)
if p.total_memory < 75 * 1024**3:
    raise SystemExit(f"expected 80GB-class GPU, got {p.total_memory / 1024**3:.1f} GiB")
print(torch.cuda.get_device_name(0), round(p.total_memory / 1024**3, 2), flush=True)
PY

resume_args=()
if [[ "${RESUME:-0}" == 1 ]]; then
  resume_args+=(--resume)
fi

exec "${PYTHON}" -u -m nimloth.eval.dino_grid_wm_reconstruction \
  --sft2-checkpoint "${SFT2_CHECKPOINT}" \
  --val-jsonl "${VAL_JSONL}" \
  --old-dino-grid-cache "${OLD_DINO_GRID_CACHE}" \
  --qwen-cache "${QWEN_CACHE}" \
  --dino-grid-cfm-checkpoint "${OLD_DINO_GRID_CFM_CHECKPOINT}" \
  --id56-dino-grid-cache "${ID56_DINO_GRID_CACHE}/val" \
  --id56-dino-grid-cfm-checkpoint "${ID56_DINO_GRID_CFM_CHECKPOINT}" \
  --qwen-cfm-checkpoint "${QWEN_CFM_CHECKPOINT}" \
  --selections "${SELECTIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --git-commit "${EXPECTED_COMMIT}" \
  --horizon 4 \
  --steps 50 \
  --cfg-scale 2.0 \
  --chunk-size 8 \
  --encode-batch-size 1 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  "${resume_args[@]}"
