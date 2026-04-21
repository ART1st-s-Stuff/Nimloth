#!/usr/bin/env bash
set -euo pipefail

manifest_path="${WM_MANIFEST_PATH:-}"
collection_root="${WM_COLLECTION_ROOT:-datasets}"
collection_phase="${WM_COLLECTION_PHASE:-phase1}"
collection_task="${WM_COLLECTION_TASK:-wm_data_collection}"
if [[ -z "${manifest_path}" ]]; then
  manifest_path="$(
    WM_COLLECTION_ROOT="${collection_root}" WM_COLLECTION_PHASE="${collection_phase}" WM_COLLECTION_TASK="${collection_task}" python - <<'PY'
from pathlib import Path
import os
base = Path(os.environ["WM_COLLECTION_ROOT"]) / os.environ["WM_COLLECTION_PHASE"] / os.environ["WM_COLLECTION_TASK"]
runs = sorted([p for p in base.glob("*/*") if (p / "manifest.jsonl").exists()], reverse=True)
print((runs[0] / "manifest.jsonl") if runs else "")
PY
)"
fi

if [[ -z "${manifest_path}" ]]; then
  echo "未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh 或设置 WM_MANIFEST_PATH"
  exit 1
fi

python -m src.train.train_wm dataset.manifest_path="${manifest_path}" "$@"

