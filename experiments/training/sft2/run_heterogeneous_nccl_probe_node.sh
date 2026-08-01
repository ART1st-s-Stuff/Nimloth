#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"
: "${SLURM_PROCID:?}"

case "${SLURM_PROCID}" in
  0) LOCAL_WORLD_SIZE=8 ;;
  1|2) LOCAL_WORLD_SIZE=4 ;;
  *) echo "unexpected physical-node rank: ${SLURM_PROCID}" >&2; exit 1 ;;
esac

exec /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  -m torch.distributed.run \
  --nnodes=3 \
  --node_rank="${SLURM_PROCID}" \
  --nproc_per_node="${LOCAL_WORLD_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${REPO}/experiments/training/sft2/probe_heterogeneous_nccl.py"
