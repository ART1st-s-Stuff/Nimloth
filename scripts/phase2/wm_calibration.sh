#!/usr/bin/env bash
set -euo pipefail

ckpt_path="${WM_CKPT_PATH:-}"
wm_model_root="${WM_MODEL_ROOT:-models}"
wm_model_name="${WM_MODEL_NAME:-${WM_NAME:-cfm_dinov2m}}"
if [[ -z "${ckpt_path}" ]]; then
  ckpt_path="$(
    WM_MODEL_ROOT="${wm_model_root}" WM_MODEL_NAME="${wm_model_name}" python - <<'PY'
import os
from pathlib import Path
from src.utils.model_provider import resolve_latest_model_file
base = Path(os.environ["WM_MODEL_ROOT"]) / "wm" / os.environ["WM_MODEL_NAME"]
resolved = resolve_latest_model_file(base, ["wm_ema.pt", "wm.pt"])
print(resolved or "")
PY
)"
fi

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

if [[ -z "${ckpt_path}" ]]; then
  echo "未找到可用 checkpoint，请先执行 scripts/phase2/wm_training.sh 或设置 WM_CKPT_PATH"
  exit 1
fi

if [[ -z "${manifest_path}" ]]; then
  echo "未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh 或设置 WM_MANIFEST_PATH"
  exit 1
fi

python -m src.train.calibrate_wm wm="${wm_model_name}" pipeline.calib.input_ckpt_path="${ckpt_path}" dataset.manifest_path="${manifest_path}" "$@"

