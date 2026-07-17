#!/usr/bin/env bash
set -euo pipefail
: "${RANK_OFFSET:?must set RANK_OFFSET}"
: "${REPO:?must set REPO}"
: "${MODEL:?must set MODEL}"
: "${TRAJECTORIES:?must set TRAJECTORIES}"
: "${PROBE_OUT:?must set PROBE_OUT}"
: "${MASTER_ADDR:?must set MASTER_ADDR}"
: "${MASTER_PORT:?must set MASTER_PORT}"

export RANK=$((RANK_OFFSET + SLURM_PROCID))
export WORLD_SIZE=8
export LOCAL_RANK=${SLURM_LOCALID}
cd "${REPO}"
exec /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  "${REPO}/experiments/training/rl/probe_actor_recompute_memory.py" \
  --model "${MODEL}" \
  --trajectories "${TRAJECTORIES}" \
  --output-dir "${PROBE_OUT}"
