#!/usr/bin/env bash
set -euo pipefail

manifest_path="${WM_MANIFEST_PATH:-}"
collection_root="${WM_COLLECTION_ROOT:-datasets}"
dataset_name="${WM_DATASET_NAME:-ai2thor}"
if [[ -z "${manifest_path}" ]]; then
  manifest_path="$(
    WM_COLLECTION_ROOT="${collection_root}" WM_DATASET_NAME="${dataset_name}" python - <<'PY'
from pathlib import Path
import json
import os
base = Path(os.environ["WM_COLLECTION_ROOT"]) / os.environ["WM_DATASET_NAME"]
meta = base / "metadata.json"
latest = None
if meta.exists():
    try:
        latest = json.loads(meta.read_text(encoding="utf-8")).get("latest")
    except Exception:
        latest = None
if latest and (base / latest / "manifest.jsonl").exists():
    print(base / latest / "manifest.jsonl")
else:
    runs = sorted([p for p in base.iterdir() if p.is_dir() and (p / "manifest.jsonl").exists()], reverse=True)
    print((runs[0] / "manifest.jsonl") if runs else "")
PY
)"
fi

if [[ -z "${manifest_path}" ]]; then
  echo "未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh 或设置 WM_MANIFEST_PATH"
  exit 1
fi

python -m src.train.train_wm dataset.manifest_path="${manifest_path}" "$@"

