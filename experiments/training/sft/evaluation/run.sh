#!/usr/bin/env bash
# Formal success-rate entry; all model/stage/episode choices are CLI arguments.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
PYTHON=${PYTHON:-python3}
export PYTHONPATH="${REPO}/src${VAGEN_DIR:+:${VAGEN_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m nimloth.training.sft.evaluation "$@"
