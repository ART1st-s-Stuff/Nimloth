#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash experiments/validation/wm_head_ablation_dev.sh
if [[ "${1:-}" == "--bootstrap-only" ]]; then
  echo "release_bootstrap=PASS"
  exit 0
fi

echo "artifact_verifier=NOT_IMPLEMENTED" >&2
exit 2
