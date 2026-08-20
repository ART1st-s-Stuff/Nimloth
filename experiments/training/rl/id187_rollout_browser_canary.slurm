#!/usr/bin/env bash
#SBATCH --job-name=nimloth-id187-browser-canary
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

# ID187: one frozen, val-only K4 rollout. This validates the unified rollout
# browser on the production TP8/DP1 + actor DP8 restore path. It performs no
# optimizer update and writes no checkpoint. Resource components are supplied
# on sbatch CLI as one 6-GPU node plus one 2-GPU node.
set -euo pipefail
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
module load slurm
[[ "${SLURM_HET_SIZE:-}" == 2 ]]

export ID185_VIS_RUN_NAME_OVERRIDE=187_smoke_rollout_browser_k4_dp8_tp8_source20_base_seed2_t20_s100_preempt_retry14
export ID185_VIS_RUN_DATE_OVERRIDE=2026-08-21
export ID185_VIS_SEED_OVERRIDE=2
export ID185_VIS_EXPECTED_OUTCOME=any
export ID185_VIS_ENABLE_WANDB=true
export ID185_VIS_WANDB_RUN_ID=nimloth-id187-smoke-rollout-browser-k4-source20-seed2-preempt-r14
export ID185_VIS_EXPECTED_PARTITION=preempt
export ID185_VIS_SOURCE_BOUNDARY=20
export ID185_VIS_GPU_LAYOUT=6,2

printf 'id187_rollout_browser_canary_job=%s node=%s partition=%s cuda_visible_devices=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_JOB_PARTITION}" \
  "${CUDA_VISIBLE_DEVICES:-}"
exec "${REPO}/experiments/training/rl/launch_vagen_k4_heterogeneous_6x2_browser_on_hold.sh" \
  "${SLURM_JOB_ID}"
