#!/usr/bin/env bash
set -euo pipefail

python -m src.train.calibrate_wm "$@"

