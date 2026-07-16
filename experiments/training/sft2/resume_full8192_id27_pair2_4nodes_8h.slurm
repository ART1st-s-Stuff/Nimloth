#!/bin/bash
#SBATCH --job-name=sft2-id27-4node
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --exclude=dgx-27
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed/logs/resume-4node-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed/logs/resume-4node-%j.err

set -euo pipefail
REPO=/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon
ROOT=/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97
OUT_ROOT=$ROOT/sft2/27_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5_bucketed
CONFIG=$REPO/configs/training/sft2/latent_wm_value_k8_state8192_ep5_img8.yaml
RUN_NAME=27_state8192_fullwm8192_llmlora64a128_vislora64a128_pair2_ws4_ga8_ep5_bucketed
PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
export REPO ROOT OUT_ROOT CONFIG RUN_NAME PY PAIR_LAYOUT=one_rank_per_node
export NCCL_DEBUG=INFO NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
cd "$REPO"
echo "resume_commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST socket_if=auto"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
SMOKE_PORT=$((30001 + SLURM_JOB_ID % 10000))
export MASTER_ADDR SMOKE_PORT
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -lc '
  export RANK=$SLURM_PROCID LOCAL_RANK=0 WORLD_SIZE=4
  export MASTER_ADDR MASTER_PORT=$SMOKE_PORT NIMLOTH_SMOKE_LOCAL_PAIR=1
  export PYTHONPATH="$REPO/src:$REPO/external/le-wm:$REPO/external/VAGEN"
  "$PY" experiments/validation/qwen_partitioned_grad_sync_smoke.py
'
unset MASTER_ADDR
bash experiments/training/sft2/run_full8192_lora_pair2_3plus1.sh
