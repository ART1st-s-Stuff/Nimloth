#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_REPO="/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon"
SERVER_PYTHON="/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3"
CONFIG="configs/training/reconstruction/frozen_wm_head_shape_ablation.json"
cd "$ROOT"

bash experiments/validation/wm_head_ablation_dev.sh
if [[ "${1:-}" == "--bootstrap-only" ]]; then
  echo "release_bootstrap=PASS"
  exit 0
fi

COMMIT="$(git rev-parse HEAD)"
ssh superpod-csejzhang "cd '$SERVER_REPO' && test \"\$(git rev-parse HEAD)\" = '$COMMIT'"
ssh superpod-csejzhang "cd '$SERVER_REPO' && WM_HEAD_PYTHON='$SERVER_PYTHON' bash experiments/validation/wm_head_ablation_dev.sh"
ssh superpod-csejzhang "cd '$SERVER_REPO' && PYTHONPATH=\$PWD/src:\$PWD/external/le-wm '$SERVER_PYTHON' experiments/validation/verify_wm_head_artifacts.py --config '$CONFIG'"
echo "release_suite=PASS commit=$COMMIT"
