#!/usr/bin/env bash
set -euo pipefail

ckpt_path="${WM_CKPT_PATH:-}"
if [[ -z "${ckpt_path}" ]]; then
  ckpt_path="$(
    python - <<'PY'
from pathlib import Path
base = Path("outputs/phase2/wm_training")
runs = sorted([p for p in base.glob("*/*") if (p / "wm.pt").exists()], reverse=True)
print((runs[0] / "wm.pt") if runs else "")
PY
)"
fi

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

if [[ -z "${ckpt_path}" ]]; then
  echo "未找到可用 checkpoint，请先执行 scripts/phase2/wm_training.sh 或设置 WM_CKPT_PATH"
  exit 1
fi

if [[ -z "${manifest_path}" ]]; then
  echo "未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh 或设置 WM_MANIFEST_PATH"
  exit 1
fi

python -m src.train.calibrate_wm calib.calib.input_ckpt_path="${ckpt_path}" dataset.dataset.manifest_path="${manifest_path}" "$@"

