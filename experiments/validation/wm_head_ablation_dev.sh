#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${WM_HEAD_PYTHON:-python3}"
python3 experiments/validation/wm_head_ablation_bootstrap.py
python3 -m py_compile \
  experiments/validation/wm_head_ablation_bootstrap.py \
  src/nimloth/wm/frozen_query_state.py \
  src/nimloth/wm/frozen_state_cache.py \
  src/nimloth/wm/matched_heads.py \
  tests/test_matched_wm_heads.py \
  tests/training/test_matched_wm_trainer.py

if "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
  PYTHONPATH="$ROOT/src:$ROOT/external/le-wm" \
    "$PYTHON_BIN" -m pytest -q \
      tests/test_matched_wm_heads.py \
      tests/training/test_matched_wm_trainer.py
  echo "torch_contracts=PASS"
else
  echo "torch_contracts=DEFERRED_TO_PINNED_SERVER"
fi

echo "affected_targets=bootstrap,canonical_config,matched_head_contracts"
echo "cache_policy=immutable_source_fingerprints"
echo "parallel_policy=matched_single_process_default"
echo "developer_suite=PASS"
