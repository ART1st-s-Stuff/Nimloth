#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}" "${TINY_PROBE_OUT:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"
: "${RANK_OFFSET:?}" "${WORLD_SIZE:?}"
export RANK=$((RANK_OFFSET + SLURM_PROCID))
export LOCAL_RANK=${SLURM_LOCALID}
export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/rl/probe_external_checkpoint_fsdp.py" \
  --output-dir "${TINY_PROBE_OUT}"
