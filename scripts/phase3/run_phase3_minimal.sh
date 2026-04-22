#!/usr/bin/env bash
set -euo pipefail

# Phase 3 最小回归顺序：
# 1) 语义对齐训练
# 2) 语义对齐评估
# 3) PM-ready 特征导出

bash scripts/phase3/semantic_align_train.sh "$@"
bash scripts/phase3/semantic_align_eval.sh
bash scripts/phase3/export_pm_ready_features.sh
