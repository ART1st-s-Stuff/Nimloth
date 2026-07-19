#!/bin/bash
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err

set -euo pipefail
REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-query-align}
cd "${REPO}"
exec bash experiments/training/sft2/run_dino_world8_4x2.sh
