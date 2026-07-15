#!/usr/bin/env bash
#SBATCH --job-name=wm-dynamics-dim
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

REPO="${REPO:-/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon}"
PYTHON_BIN="${PYTHON_BIN:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}"
CONFIG="${CONFIG:-configs/training/reconstruction/wm_dynamics_dim_ablation.json}"
STAGE="${STAGE:-all}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/external/le-wm"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
fi

nvidia-smi
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  TRAIN_ARGS=(--config "$CONFIG")
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
  fi
  "$PYTHON_BIN" -m nimloth.training.wm_heads.dynamics_dim_train_cli "${TRAIN_ARGS[@]}"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
  OUTPUT_DIR="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$CONFIG")"
  "$PYTHON_BIN" -m nimloth.eval.dynamics_dim_cli \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/train/best"
fi
