#!/usr/bin/env bash
set -euo pipefail

manifest_path="${WM_MANIFEST_PATH:-}"
if [[ -z "${manifest_path}" ]]; then
  manifest_path="$(
    python - <<'PY'
from pathlib import Path
base = Path("outputs/phase1/wm_data_collection")
runs = sorted([p for p in base.glob("*/*") if (p / "manifest.jsonl").exists()], reverse=True)
print((runs[0] / "manifest.jsonl") if runs else "")
PY
)"
fi

if [[ -z "${manifest_path}" ]]; then
  echo "未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh 或设置 WM_MANIFEST_PATH"
  exit 1
fi

python -m src.train.train_wm dataset.dataset.manifest_path="${manifest_path}" "$@"

