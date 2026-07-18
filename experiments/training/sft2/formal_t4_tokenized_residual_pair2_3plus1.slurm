#!/bin/bash
#SBATCH --job-name=sft2-t4tok8res
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-27
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/29_state8192_t4tok8res_llmlora_vislora_pair2_ws4_ga8_ep5/logs/formal-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/29_state8192_t4tok8res_llmlora_vislora_pair2_ws4_ga8_ep5/logs/formal-%j.err
#SBATCH hetjob
#SBATCH --job-name=sft2-t4tok8res
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-54
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

set -euo pipefail
REPO=/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon
ROOT=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97
OUT_ROOT=$ROOT/sft2/29_state8192_t4tok8res_llmlora_vislora_pair2_ws4_ga8_ep5
CONFIG=$REPO/configs/training/sft2/latent_wm_value_k8_state8192_t4_tok8_residual_ep5_img8.yaml
RUN_NAME=29_state8192_t4tok8res_llmlora64a128_vislora64a128_pair2_ws4_ga8_ep5
PAIR_LAYOUT=hetero_3plus1
export REPO ROOT OUT_ROOT CONFIG RUN_NAME PAIR_LAYOUT
mkdir -p "$OUT_ROOT/logs"
cd "$REPO"
echo "formal_commit=$(git rev-parse HEAD) job=${SLURM_JOB_ID:-unset}"
exec bash experiments/training/sft2/run_full8192_lora_pair2_3plus1.sh
