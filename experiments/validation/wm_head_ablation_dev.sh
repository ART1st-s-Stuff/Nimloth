#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${WM_HEAD_PYTHON:-python3}"
python3 experiments/validation/wm_head_ablation_bootstrap.py
python3 -m py_compile \
  experiments/validation/verify_wm_head_artifacts.py \
  experiments/validation/wm_head_ablation_bootstrap.py \
  src/nimloth/eval/matched_wm_ablation.py \
  src/nimloth/eval/matched_wm_cli.py \
  src/nimloth/eval/matched_wm_metrics.py \
  src/nimloth/eval/matched_wm_render.py \
  src/nimloth/eval/matched_wm_turns.py \
  src/nimloth/training/wm_heads/cache_cli.py \
  src/nimloth/training/wm_heads/config.py \
  src/nimloth/training/wm_heads/data.py \
  src/nimloth/training/wm_heads/train_cli.py \
  src/nimloth/training/wm_heads/trainer.py \
  src/nimloth/wm/frozen_query_state.py \
  src/nimloth/wm/frozen_state_cache.py \
  src/nimloth/wm/matched_heads.py \
  tests/eval/test_matched_wm_ablation.py \
  tests/test_matched_wm_heads.py \
  tests/training/test_matched_wm_trainer.py

bash -n experiments/training/reconstruction/frozen_wm_head_ablation.slurm
if "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
  for module in nimloth.training.wm_heads.cache_cli nimloth.training.wm_heads.train_cli nimloth.eval.matched_wm_cli; do
    PYTHONPATH="$ROOT/src:$ROOT/external/le-wm" "$PYTHON_BIN" -m "$module" --help >/dev/null
  done
  PYTHONPATH="$ROOT/src:$ROOT/external/le-wm" \
    "$PYTHON_BIN" -m pytest -q \
      tests/eval/test_matched_wm_ablation.py \
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
