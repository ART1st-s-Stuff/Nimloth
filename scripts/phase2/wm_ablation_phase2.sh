#!/usr/bin/env bash
set -euo pipefail

# Phase2 塌缩改进对比实验：
# 1) baseline
# 2) +stride
# 3) +stride+sigreg
# 4) +stride+sigreg+idm联调
#
# 默认执行快速烟雾实验（epochs=1）。如需完整实验可覆盖 WM_ABLATION_EPOCHS。

wm_name="${WM_NAME:-cfm_dinov2m}"
manifest_path="${WM_MANIFEST_PATH:-}"
collection_root="${WM_COLLECTION_ROOT:-datasets}"
dataset_name="${WM_DATASET_NAME:-ai2thor}"
epochs="${WM_ABLATION_EPOCHS:-1}"
batch_size="${WM_ABLATION_BATCH_SIZE:-16}"
stride_val="${WM_ABLATION_STRIDE:-2}"
sigreg_weight="${WM_ABLATION_SIGREG_WEIGHT:-0.1}"
idm_weight_tuned="${WM_ABLATION_IDM_WEIGHT:-1.5}"
dry_run="${WM_ABLATION_DRY_RUN:-0}"

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
  echo "未找到可用 manifest，请先执行 phase1 数据采集或设置 WM_MANIFEST_PATH"
  exit 1
fi

run_case() {
  local case_name="$1"
  shift
  echo "========== [${case_name}] =========="
  local cmd=(
    python -m src.train.train_wm
    "wm=${wm_name}"
    "dataset.manifest_path=${manifest_path}"
    "pipeline.train.epochs=${epochs}"
    "pipeline.train.batch_size=${batch_size}"
    "$@"
  )
  if [[ "${dry_run}" == "1" ]]; then
    printf 'DRY_RUN: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi
  "${cmd[@]}"
}

run_case "baseline" \
  "pipeline.train.temporal_stride=1" \
  "pipeline.train.sigreg.enabled=false" \
  "pipeline.train.semi_supervised_weight=1.0"

run_case "stride" \
  "pipeline.train.temporal_stride=${stride_val}" \
  "pipeline.train.sigreg.enabled=false" \
  "pipeline.train.semi_supervised_weight=1.0"

run_case "stride_sigreg" \
  "pipeline.train.temporal_stride=${stride_val}" \
  "pipeline.train.sigreg.enabled=true" \
  "pipeline.train.sigreg.weight=${sigreg_weight}" \
  "pipeline.train.semi_supervised_weight=1.0"

run_case "stride_sigreg_idm_tuned" \
  "pipeline.train.temporal_stride=${stride_val}" \
  "pipeline.train.sigreg.enabled=true" \
  "pipeline.train.sigreg.weight=${sigreg_weight}" \
  "pipeline.train.semi_supervised_weight=${idm_weight_tuned}"
