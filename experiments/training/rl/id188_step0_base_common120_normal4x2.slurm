#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id188-step0-bc120-n4x2
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51

# ID188: frozen pre-RL step0 evaluation on held-out Base60 + Common60.
# Every rollout stores full state and 100-simulation MCTS evidence. No backward,
# optimizer update, resume, or checkpoint write is allowed; topology is 4x2 H800.
set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
module load slurm
[[ -z "${SLURM_HET_SIZE:-}" ]]

export ID185_VIS_RUN_NAME_OVERRIDE=188_eval_rollout_browser_k4_dp8_tp8_step0_base_common120_t20_s100_normal_4x2
export ID185_VIS_RUN_DATE_OVERRIDE=2026-08-24
export ID185_VIS_EXPECTED_OUTCOME=any
export ID185_VIS_PHASE_NAME_OVERRIDE=base_common120
export ID185_VIS_PHASE_TAG_OVERRIDE=bc120-step0
export ID185_VIS_RUNNER_OVERRIDE="${REPO}/experiments/training/rl/run_vagen_k4_id188_step0_base_common120_normal4x2.sh"
export ID185_VIS_ENABLE_WANDB=true
export ID185_VIS_WANDB_RUN_ID=nimloth-id188-eval-rollout-browser-k4-step0-base-common120-normal-4x2
export ID185_VIS_EXPECTED_PARTITION=normal
export ID185_VIS_SOURCE_BOUNDARY=0
export VAGEN_ROLLOUT_BROWSER_PACK_WORKERS=8

printf 'id188_step0_base_common120_normal4x2_job=%s node=%s partition=%s cuda_visible_devices=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_JOB_PARTITION}" \
  "${CUDA_VISIBLE_DEVICES:-}"
exec "${REPO}/experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh" \
  "${SLURM_JOB_ID}"
