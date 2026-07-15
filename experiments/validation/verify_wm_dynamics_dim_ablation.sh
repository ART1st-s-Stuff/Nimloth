#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_REPO="/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon"
SERVER_PYTHON="/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3"
CONFIG="configs/training/reconstruction/wm_dynamics_dim_ablation.json"
cd "$ROOT"

bash experiments/validation/wm_head_ablation_dev.sh
if [[ "${1:-}" == "--bootstrap-only" ]]; then
  echo "dynamics_dim_release_bootstrap=PASS"
  exit 0
fi

COMMIT="$(git rev-parse HEAD)"
ssh superpod-csejzhang "cd '$SERVER_REPO' && test \"\$(git rev-parse HEAD)\" = '$COMMIT'"
ssh superpod-csejzhang "cd '$SERVER_REPO' && WM_HEAD_PYTHON='$SERVER_PYTHON' bash experiments/validation/wm_head_ablation_dev.sh"
ssh superpod-csejzhang "cd '$SERVER_REPO' && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=\$PWD/src:\$PWD/external/le-wm '$SERVER_PYTHON' experiments/validation/verify_wm_dynamics_dim_artifacts.py --config '$CONFIG'"
echo "dynamics_dim_release_suite=PASS commit=$COMMIT"
