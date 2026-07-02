#!/bin/bash
# Create a Python environment for the official LeWM reproduction scripts.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-${REPO}/.venv-lewm}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.10}

cd "${REPO}"
uv venv --python "${PYTHON_BIN}" "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install 'stable-worldmodel[train,env]' pillow huggingface_hub einops
"${VENV}/bin/python" - <<'PY'
import torch
import stable_worldmodel
import stable_pretraining
import PIL
print('torch', torch.__version__)
print('stable_worldmodel ok')
print('stable_pretraining ok')
print('PIL ok')
PY
