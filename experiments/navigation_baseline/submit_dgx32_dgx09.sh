#!/bin/bash
set -euo pipefail
module load slurm >/dev/null 2>&1
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=/project/peilab/atst/nimloth/experiments/navigation_baseline
cd "$SCRIPTDIR"
chmod +x resume_retry2_train_from50_dgx32_4train_external_env.slurm convert_vagen50_to_world_size4_dgx32.slurm 2>/dev/null || true

$SL/scancel 452573 2>/dev/null || true
pkill -u csejzhang -f launch_dgx31_hold_and_2env4train 2>/dev/null || true

CKPT="$SCRIPTDIR/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_50/actor/model_world_size_4_rank_3.pt"
CONV_DEP=""
if [ ! -f "$CKPT" ]; then
  CONV_JOB=$($SL/sbatch convert_vagen50_to_world_size4_dgx32.slurm | awk '{print $NF}')
  echo "CONV_JOB=$CONV_JOB"
  CONV_DEP="--dependency=afterok:$CONV_JOB"
else
  echo "ws4_exists"
fi

ENV_JOB=$($SL/sbatch env_dgx09_2gpu_resume_retry2.slurm | awk '{print $NF}')
echo "ENV_JOB=$ENV_JOB"
TRAIN_JOB=$($SL/sbatch $CONV_DEP resume_retry2_train_from50_dgx32_4train_external_env.slurm | awk '{print $NF}')
echo "TRAIN_JOB=$TRAIN_JOB"
$SL/squeue -u csejzhang -o "%.10i %.20j %.2t %.6D %R"
