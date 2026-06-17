#!/bin/bash
# Deprecated — forwards to experiments/training/sft2/submit_llmvis_lora_preempt6g_dgx11.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../training/sft2" && pwd)"
exec "${ROOT}/submit_llmvis_lora_preempt6g_dgx11.sh" "$@"
