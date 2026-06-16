#!/bin/bash
# Submit SFT1 greedy rollouts: 4-GPU env + 4-node x 2-GPU full parallel (no array packing).
set -euo pipefail

ROOT=/project/peilab/atst/nimloth/experiments/navigation_baseline
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

mkdir -p /project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm

RUN_DIR=${ROOT}/runs/sft1_rollouts_vagen79_greedy_parallel
ENV_READY=${RUN_DIR}/external_env_4gpu/ready

echo "=== Submit SFT1 vagen79 greedy rollouts at $(date) ==="
echo "Checkpoint: global_step_79/actor/huggingface"
echo "Decode: greedy (do_sample=False, temperature=0, n=1)"
echo "Parallel: 4 nodes x 2 GPU rollout workers + 4 GPU env (12 GPU total)"
echo "Output: ${RUN_DIR}/validation/{train,val,test}/shard_*/79.jsonl"

if [ -f "${ENV_READY}" ]; then
  echo "Reusing existing env (ready file present)"
  J_ENV=""
else
  J_ENV=$($SLURM "${ROOT}/sft1_env_vagen79_4gpu.slurm" | awk '{print $NF}')
  echo "env job: ${J_ENV}"
fi

if [ -n "${J_ENV}" ]; then
  J_ROLL=$($SLURM --dependency=after:${J_ENV} "${ROOT}/sft1_rollouts_vagen79_greedy_4node.slurm" | awk '{print $NF}')
else
  J_ROLL=$($SLURM "${ROOT}/sft1_rollouts_vagen79_greedy_4node.slurm" | awk '{print $NF}')
fi
echo "rollout 4-node job: ${J_ROLL}"

echo "Monitor:"
echo "  squeue -u \$USER | grep sft1"
echo "  tail -f ${RUN_DIR}/sft1_rollouts_vagen79_greedy_parallel.log"
