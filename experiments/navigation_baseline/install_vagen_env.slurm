#!/bin/bash
#SBATCH --job-name=vagen-install
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --exclude=dgx-03,dgx-16,dgx-20,dgx-23,dgx-31,dgx-34,dgx-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --requeue

set -euo pipefail

echo "=== VAGEN uv env install started at $(date) ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

ROOT=/project/peilab/atst/nimloth
VENV=${ROOT}/.venv
UV=${ROOT}/.local/bin/uv

export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export HOME=${ROOT}/.home
export PATH=${VENV}/bin:${ROOT}/.local/bin:$PATH
mkdir -p "$HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

if [ ! -x "$VENV/bin/python" ]; then
  cd "$ROOT"
  "$UV" venv --python 3.12 .venv
fi

cd ${ROOT}/external/VAGEN/verl

echo "=== Python ==="
python --version
which python

echo "=== Install core VERL/VAGEN runtime dependencies ==="
uv pip install --prerelease=allow --python "$VENV/bin/python" \
  "sglang[all]==0.5.3.post3" \
  "transformers[hf_xet]>=4.51.0" \
  accelerate datasets peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
  "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
  "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "=== Install flash-attn prebuilt if available; otherwise skip with warning ==="
set +e
uv pip install --python "$VENV/bin/python" \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
FLASH_STATUS=$?
set -e
if [ "$FLASH_STATUS" -ne 0 ]; then
  echo "WARNING: prebuilt flash-attn wheel install failed; training may fall back or fail depending on model config."
fi

echo "=== Install editable verl and VAGEN ==="
cd ${ROOT}/external/VAGEN/verl
uv pip install --python "$VENV/bin/python" --no-deps -e .
cd ${ROOT}/external/VAGEN
uv pip install --python "$VENV/bin/python" -e .

echo "=== Install navigation extras ==="
uv pip install --python "$VENV/bin/python" "trl==0.26.2" ai2thor httpx

echo "=== Pre-download navigation scenes ==="
source ${ROOT}/experiments/navigation_baseline/setup_ai2thor_env.sh
python -m vagen.envs.navigation.pre_download_scenes || echo "WARNING: scene pre-download failed or already completed"

echo "=== Verify packages ==="
python - <<'PY'
import importlib
for name in ['vagen', 'verl', 'sglang', 'ray', 'ai2thor', 'torch']:
    m = importlib.import_module(name)
    print(name, getattr(m, '__version__', 'no_version'), getattr(m, '__file__', 'builtin'))
PY

echo "=== VAGEN uv env install finished at $(date) ==="
