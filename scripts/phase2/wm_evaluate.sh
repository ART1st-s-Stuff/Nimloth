#!/usr/bin/env bash
set -euo pipefail

wm_model_root="${WM_MODEL_ROOT:-models}"
wm_model_name="${WM_MODEL_NAME:-${WM_NAME:-cfm_dinov2m}}"

resolve_latest_model_file() {
  local file_name="$1"
  WM_MODEL_ROOT="${wm_model_root}" WM_MODEL_NAME="${wm_model_name}" TARGET_FILE="${file_name}" python - <<'PY'
from pathlib import Path
import json
import os
base = Path(os.environ["WM_MODEL_ROOT"]) / "wm" / os.environ["WM_MODEL_NAME"]
target_file = os.environ["TARGET_FILE"]
meta = base / "metadata.json"
latest = None
if meta.exists():
    try:
        latest = json.loads(meta.read_text(encoding="utf-8")).get("latest")
    except Exception:
        latest = None
if latest and (base / latest / target_file).exists():
    print(base / latest / target_file)
else:
    runs = sorted([p for p in base.iterdir() if p.is_dir() and (p / target_file).exists()], reverse=True)
    print((runs[0] / target_file) if runs else "")
PY
}

wm_ckpt_path="${WM_CKPT_PATH:-$(resolve_latest_model_file "wm.pt")}"
idm_ckpt_path="${WM_IDM_CKPT_PATH:-$(resolve_latest_model_file "inverse_dynamics.pt")}"
mapper_ckpt_path="${WM_MAPPER_CKPT_PATH:-$(resolve_latest_model_file "action_mapper.pt")}"
theta_div_path="${WM_THETA_DIV_PATH:-$(resolve_latest_model_file "theta_div.json")}"

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

if [[ -z "${wm_ckpt_path}" ]]; then
  echo "未找到可用 wm.pt，请先执行 scripts/phase2/wm_training.sh 或设置 WM_CKPT_PATH"
  exit 1
fi

if [[ -z "${manifest_path}" ]]; then
  echo "未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh 或设置 WM_MANIFEST_PATH"
  exit 1
fi

cmd=(
  python -m src.train.evaluate_wm
  "wm=${wm_model_name}"
  "pipeline.eval.wm_ckpt_path=${wm_ckpt_path}"
  "pipeline.eval.idm_ckpt_path=${idm_ckpt_path}"
  "pipeline.eval.action_mapper_ckpt_path=${mapper_ckpt_path}"
  "dataset.manifest_path=${manifest_path}"
)
if [[ -n "${theta_div_path}" ]]; then
  cmd+=("pipeline.eval.theta_div_path=${theta_div_path}")
fi
cmd+=("$@")
"${cmd[@]}"

