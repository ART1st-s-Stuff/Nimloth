#!/usr/bin/env bash
#SBATCH --job-name=id56-cfm-sweep
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
PYTHON=${ROOT}/.venv-vagen-main/bin/python3
WANDB_ENV=/project/peilab/atst/flower/.env

: "${WORKTREE:?set WORKTREE to the clean remote worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact sweep commit}"
: "${CFM_RUN_ROOT:?set CFM_RUN_ROOT to the completed ID50 run}"
: "${SOURCE_EVAL_DIR:?set SOURCE_EVAL_DIR to the completed ID51 evaluation}"
: "${RUN_ROOT:?set RUN_ROOT to a new sweep-owned directory}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME to a new formal diagnostic identity}"

INIT_CFM=${INIT_CFM:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction/45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt}

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
  "${CFM_RUN_ROOT}/train/best.pt" \
  "${CFM_RUN_ROOT}/train/checkpoint_000029640.pt" \
  "${SOURCE_EVAL_DIR}/metadata.json" \
  "${SOURCE_EVAL_DIR}/samples.json" \
  "${SOURCE_EVAL_DIR}/states.pt" \
  "${INIT_CFM}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "new sweep run root is not empty: ${RUN_ROOT}" >&2
  exit 2
fi

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

mkdir -p "${RUN_ROOT}"
cat > "${RUN_ROOT}/README.md" <<EOF
# ID56 CFM checkpoint reconstruction sweep

- Status: running
- Git commit: ${EXPECTED_COMMIT}
- Source evaluation: ${SOURCE_EVAL_DIR}
- Checkpoints: ${CFM_RUN_ROOT}/train/checkpoint_*.pt, best.pt, and the strict ID45 initialization
- Dataset: the existing held-out diverse40 rows and cached actual/autoregressive predicted states
- Protocol: matched noise, Euler50, CFG2 for every checkpoint and both state types
- Primary selection metric: WM-predicted reconstruction to GT L1
- Module updates: none; all checkpoints are frozen
- W&B: ${WANDB_PROJECT}/${WANDB_RUN_NAME}
- Resume: none; this short immutable sweep restarts into a new empty output if interrupted
EOF

export PYTHONPATH=${WORKTREE}/src:${WORKTREE}/external/le-wm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
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

"${PYTHON}" -u -m nimloth.eval.cfm_checkpoint_sweep \
  --evaluation-dir "${SOURCE_EVAL_DIR}" \
  --checkpoint-dir "${CFM_RUN_ROOT}/train" \
  --best-checkpoint "${CFM_RUN_ROOT}/train/best.pt" \
  --initialization-checkpoint "${INIT_CFM}" \
  --output-dir "${RUN_ROOT}" \
  --val-cache-fingerprint 5eb34b5263a0afbd \
  --git-commit "${EXPECTED_COMMIT}" \
  --steps 50 \
  --cfg-scale 2 \
  --chunk-size 32 \
  --seed 20260730 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}"

"${PYTHON}" -c "import json; p=json.load(open('${RUN_ROOT}/summary.json')); assert p['status']=='completed'; print({'selected': p['selected']['label'], 'predicted_l1': p['selected']['image_predicted_to_gt_l1']})"
