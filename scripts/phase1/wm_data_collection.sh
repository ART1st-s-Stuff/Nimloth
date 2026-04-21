#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "clean" ]]; then
  shift
  if pgrep -f "src.train.collect_data" >/dev/null 2>&1; then
    echo "[warn] 检测到采集进程仍在运行，请先停止后再 clean。"
    exit 1
  fi
  collection_root="${WM_COLLECTION_ROOT:-datasets}"
  dataset_name="${WM_DATASET_NAME:-ai2thor}"
  WM_COLLECTION_ROOT="${collection_root}" WM_DATASET_NAME="${dataset_name}" python - <<'PY'
from pathlib import Path
import shutil
import os

target = Path(os.environ["WM_COLLECTION_ROOT"]) / os.environ["WM_DATASET_NAME"]
if target.exists():
    shutil.rmtree(target)
print(f"[ok] 已清理 {target}")
PY
  exit 0
fi

python -m src.train.collect_data "$@"

