#!/bin/bash
# Submit post-SFT train/val/test rollouts (full 151676 vocab) for epoch012 and final.
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch

for tag in epoch012 final; do
  job=$($SLURM --job-name="sft1-roll-sft-${tag}" \
    --export=ALL,SFT_TAG="${tag}" \
    "${ROOT}/sft1_rollouts_vagen48_sft_ws2_2node_externalenv.slurm")
  echo "${tag}: rollout=${job}"
done
