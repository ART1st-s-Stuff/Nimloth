#!/bin/bash
#SBATCH --job-name=sft2-id27-resume
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-27
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed/logs/resume-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed/logs/resume-%j.err
#SBATCH hetjob
#SBATCH --job-name=sft2-id27-resume
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-54
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail
REPO=/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon
ROOT=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97
OUT_ROOT=$ROOT/sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed
CONFIG=$REPO/configs/training/sft2/latent_wm_value_k8_state8192_ep5_img8.yaml
RUN_NAME=27_state8192_fullwm8192_llmlora64a128_vislora64a128_pair2_ws4_ga8_ep5_bucketed
export OUT_ROOT CONFIG RUN_NAME
cd "$REPO"
echo "resume_commit=$(git rev-parse HEAD) job=${SLURM_JOB_ID:-unset}"
bash experiments/training/sft2/run_full8192_lora_pair2_3plus1.sh
