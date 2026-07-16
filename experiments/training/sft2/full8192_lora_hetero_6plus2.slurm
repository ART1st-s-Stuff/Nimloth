#!/bin/bash
#SBATCH --job-name=sft2-full8192-lora
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-27
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/17_state8192_fullwm8192_ws8_ga4_ep5_dgx27x6_dgx54x2/logs/formal-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/17_state8192_fullwm8192_ws8_ga4_ep5_dgx27x6_dgx54x2/logs/formal-%j.err
#SBATCH hetjob
#SBATCH --job-name=sft2-full8192-lora
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-54
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

set -euo pipefail
REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon}
cd "$REPO"
bash experiments/training/sft2/run_full8192_lora_hetero_6plus2.sh
