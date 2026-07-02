#!/bin/bash
# Create a Python environment for the official LeWM reproduction scripts.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-${REPO}/.venv-lewm}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.10}

cd "${REPO}"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "${PYTHON_BIN}" "${VENV}"
else
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install 'stable-worldmodel[train]' 'transformers==4.55.4' pillow huggingface_hub einops h5py hdf5plugin zstandard imageio
# The cluster driver is CUDA 12.8-era; newer PyPI torch/cu130 fails at CUDA init.
"${VENV}/bin/python" -m pip install --force-reinstall 'torch==2.8.0' 'torchvision==0.23.0'
"${VENV}/bin/python" -m pip install --force-reinstall --no-deps 'torchvision==0.23.0'
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
