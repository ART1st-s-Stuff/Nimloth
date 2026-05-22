#!/usr/bin/env bash
#SBATCH --job-name=qwen_fork_20ep_rerun
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-16
#SBATCH --requeue
#SBATCH --output=/project/peilab/atst/flower/running/slurm_logs/%x-%j.out
#SBATCH --error=/project/peilab/atst/flower/running/slurm_logs/%x-%j.err

set -euo pipefail

cd /project/peilab/atst/flower

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a
  source /project/peilab/atst/flower/.env
  set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a
  source /project/peilab/atst/.env
  set +a
fi

. .venv/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_PROJECT=${WANDB_PROJECT:-flower}
export HOME=/project/peilab/atst
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/ai2thor-fork-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

source running/setup_ai2thor_env.sh

OUT=outputs/superpod/qwen_value_rpe_headens8_rank_8192_5ep/fork_collect_pred_unc_e20_h20_nav_latest_rerun_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

python dev/collect_eb_nav_uncertainty_fork_rollouts.py \
  --dataset datasets/EB-Nav/eb-nav_dataset_single_step.json \
  --embodiedbench-root /project/peilab/atst/EmbodiedBench \
  --wm-checkpoint outputs/superpod/qwen_wm_s3_headens8_fixed_5ep/latest_for_fork_test.pt \
  --value-checkpoints outputs/superpod/qwen_value_rpe_headens8_rank_8192_5ep/fork_refine/best_value_ensemble.pt \
  --planner-lora models/qwen_planner_lora_preaction_leakfree_8192_step00008000 \
  --output-dir "$OUT" \
  --num-episodes 20 \
  --max-steps 30 \
  --fork-horizon 20 \
  --max-forks-per-episode 3 \
  --fork-margin-threshold 0.00005 \
  --fork-value-std-threshold 0.012 \
  --fork-pred-uncertainty-threshold 0.08 \
  --fork-actions selected,ucb1 \
  --fork-action-allowlist 0-5 \
  --ucb-beta 1.0 \
  --split all \
  --seed 20260520 \
  --history-len 4 \
  --resolution 500 \
  --fov 100 \
  --exp-name qwen_value_rpe_fork_pred_unc_e20_h20_nav_latest_rerun \
  --cuda-device 0 \
  --visual-encoder qwen \
  --mode hybrid \
  --risk-lambda 0.0 \
  --max-new-tokens 128 \
  --progress-weight 0.0 \
  --success-bonus 1.0 \
  --collision-penalty 0.1 \
  --reward-weight 1.0
