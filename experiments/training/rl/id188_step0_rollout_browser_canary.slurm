#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id188-step0-browser
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51

# ID188: same Base seed2 evidence canary as ID187, initialized directly from
# corrected SFT2 + ID74 planning weights before any joint PPO update.
set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
module load slurm

export ID185_VIS_RUN_NAME_OVERRIDE=188_smoke_rollout_browser_k4_dp8_tp8_step0_base_seed2_t20_s100
export ID185_VIS_RUN_DATE_OVERRIDE=2026-08-21
export ID185_VIS_SEED_OVERRIDE=2
export ID185_VIS_EXPECTED_OUTCOME=any
export ID185_VIS_ENABLE_WANDB=true
export ID185_VIS_WANDB_RUN_ID=nimloth-id188-smoke-rollout-browser-k4-step0-seed2
export ID185_VIS_EXPECTED_PARTITION=preempt
export ID185_VIS_SOURCE_BOUNDARY=0
export ID185_VIS_RUNNER_OVERRIDE=${REPO}/experiments/training/rl/run_vagen_k4_id188_step0_browser_canary.sh

printf 'id188_step0_rollout_browser_job=%s node=%s partition=%s cuda_visible_devices=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_JOB_PARTITION}" \
  "${CUDA_VISIBLE_DEVICES:-}"
exec "${REPO}/experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh" \
  "${SLURM_JOB_ID}"
