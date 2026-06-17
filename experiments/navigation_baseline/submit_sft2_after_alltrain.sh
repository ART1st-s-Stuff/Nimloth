#!/bin/bash
# Deprecated — forwards to experiments/training/sft2/submit_default_8gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../training/sft2" && pwd)"
exec "${ROOT}/submit_default_8gpu.sh" "$@"
