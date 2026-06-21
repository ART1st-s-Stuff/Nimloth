#!/usr/bin/env bash
# Shared environment for VAGEN navigation baseline jobs on superpod.
# Source from Slurm scripts: source "${SCRIPTDIR}/common_env.sh"

REPO="${REPO:-/project/peilab/atst/nimloth}"

export UV_CACHE_DIR="${REPO}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${REPO}/.local/python"
export XDG_CACHE_HOME="${REPO}/.cache"
export HOME="${REPO}/.home"
export FLASHINFER_WORKSPACE_DIR="${REPO}/.cache/flashinfer"
export WANDB_DIR="${REPO}/.cache/wandb"
# Triton/torch compile caches on local /tmp avoid NFS stale-file-handle crashes.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${SLURM_JOB_ID:-local}_$(hostname)}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_ext_${SLURM_JOB_ID:-local}_$(hostname)}"
mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
export PATH="${REPO}/.venv/bin:${REPO}/.local/bin:$PATH"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export PYTHONPATH="${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${PYTHONPATH:-}"

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/.env
  set +a
fi

# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=true
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export VERL_SGLANG_MASTER_PORT_BASE=50000
export VERL_SGLANG_MASTER_PORT_STRIDE=256
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker0,virbr0"
