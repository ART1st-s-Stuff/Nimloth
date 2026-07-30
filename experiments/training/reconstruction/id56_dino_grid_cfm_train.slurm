#!/usr/bin/env bash
#SBATCH --job-name=id56-grid-cfm
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
PYTHON=${ROOT}/.venv-vagen-main/bin/python3
WANDB_ENV=/project/peilab/atst/flower/.env

: "${WORKTREE:?set WORKTREE to the clean remote worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact trainer commit}"
: "${STATE_CACHE_DIR:?set STATE_CACHE_DIR to the completed ID56 cache root}"
: "${RUN_ROOT:?set RUN_ROOT to a new run-owned directory}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME to the ID48 run identity}"

SFT2_CHECKPOINT=${SFT2_CHECKPOINT:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-29/sft2/56_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16_px100352/train_ws16/epoch_002}
INIT_CFM=${INIT_CFM:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt}
OUTPUT_DIR=${RUN_ROOT}/train

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
for path in \
  "${PYTHON}" \
  "${STATE_CACHE_DIR}/summary.json" \
  "${STATE_CACHE_DIR}/train/manifest.json" \
  "${STATE_CACHE_DIR}/val/manifest.json" \
  "${SFT2_CHECKPOINT}/model.safetensors.index.json" \
  "${INIT_CFM}"; do
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

if [[ "${RESUME:-0}" != 1 ]]; then
  if [[ -e "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "new CFM run root is not empty: ${RUN_ROOT}" >&2
    exit 2
  fi
  mkdir -p "${RUN_ROOT}"
  cat > "${RUN_ROOT}/README.md" <<EOF
# ID56-aligned DINO-grid CFM (ID48)

- Git commit: ${EXPECTED_COMMIT}
- Source checkpoint: ${SFT2_CHECKPOINT}
- State cache: ${STATE_CACHE_DIR}
- Initialization: strict model weights from ${INIT_CFM}; fresh optimizer
- Trainable: TokenConditionedFlowUNet only
- Frozen: ID56 Qwen, DINO-grid projector, WM, ValueHead
- Split: cache/train for optimization; disjoint cache/val for model selection
- Recipe: 128px, 16x1024 condition, batch32, 15 epochs, lr3e-5, condition dropout0.15
- Resume: checkpoint_*.pt plus the same output/cache/invariants and W&B identity
- W&B: ${WANDB_PROJECT}/${WANDB_RUN_NAME}
EOF
else
  test -f "${RUN_ROOT}/README.md"
fi
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH=${WORKTREE}/src:${WORKTREE}/external/le-wm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

exec "${PYTHON}" -u -m nimloth.training.reconstruction.cfm_sft2 \
  --state-cache-dir "${STATE_CACHE_DIR}" \
  --source-checkpoint "${SFT2_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --latent-token-count 16 \
  --image-size 128 \
  --base-channels 64 \
  --condition-dim 256 \
  --time-dim 512 \
  --epochs 15 \
  --batch-size 32 \
  --lr 3e-5 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --condition-dropout 0.15 \
  --init-cfm-checkpoint "${INIT_CFM}" \
  --log-interval 100 \
  --eval-interval 1000 \
  --eval-max-items 1024 \
  --save-interval 2000 \
  --validation-cache-split val \
  --skip-samples \
  --seed 20260730 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --require-wandb \
  "${resume_args[@]}"
