#!/bin/bash
# Hold 8 GPUs on NODE, then launch SFT2 default training on the same node.
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/sft2
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
SCANCEL=/cm/shared/apps/slurm/current/bin/scancel
SQUEUE=/cm/shared/apps/slurm/current/bin/squeue
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

NODE=${SFT2_NODE:?set SFT2_NODE for hold+train}
SFT1_EPOCH=${SFT1_EPOCH:-2}
SFT2_LLM_TUNE=${SFT2_LLM_TUNE:-freeze}
SFT2_VISION_TUNE=${SFT2_VISION_TUNE:-full}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft2_latentwm_default_8gpu}
TRAIN_OUT=${TRAIN_OUT:-${REPO}/outputs/experiments/training/sft2/${RUN_DATE}/${EXPERIMENT_NAME}}

echo "=== Hold + SFT2 launch ==="
echo "node: ${NODE}"
echo "tune: llm=${SFT2_LLM_TUNE} vision=${SFT2_VISION_TUNE}"
echo "output: ${TRAIN_OUT}"

HOLD_JOB=$($SLURM --account=peilab --partition=normal --nodelist="${NODE}" \
  --job-name="sft2-hold-8g" \
  "${ROOT}/hold_node_8gpu.slurm" | awk '{print $NF}')
echo "hold job: ${HOLD_JOB}"

empty_streak=0
for _ in $(seq 1 360); do
  state=$($SQUEUE -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
  if [ "${state}" = "RUNNING" ]; then
    echo "hold ${HOLD_JOB} RUNNING at $(date)"
    break
  fi
  if [ -z "${state}" ]; then
    empty_streak=$((empty_streak + 1))
    if [ "${empty_streak}" -ge 6 ]; then
      echo "ERROR hold ${HOLD_JOB} disappeared before RUNNING"
      exit 1
    fi
  else
    empty_streak=0
    echo "hold ${HOLD_JOB} state=${state} ..."
  fi
  sleep 10
done

state=$($SQUEUE -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
if [ "${state}" != "RUNNING" ]; then
  echo "ERROR hold ${HOLD_JOB} not RUNNING: ${state:-missing}"
  exit 1
fi

echo "cancelling hold ${HOLD_JOB} and submitting SFT2 train on ${NODE}"
$SCANCEL "${HOLD_JOB}" || true
sleep 3

TRAIN_JOB=$($SLURM --account=peilab --partition=normal --nodelist="${NODE}" \
  --job-name="sft2-default-8g" \
  --export=ALL,SFT2_LLM_TUNE="${SFT2_LLM_TUNE}",SFT2_VISION_TUNE="${SFT2_VISION_TUNE}",NGPUS=8,SFT1_EPOCH="${SFT1_EPOCH}",SKIP_SFT1_DONE=1,TRAIN_OUT_OVERRIDE="${TRAIN_OUT}",EXPERIMENT_NAME="${EXPERIMENT_NAME}",RUN_DATE="${RUN_DATE}" \
  "${ROOT}/train_vagen79_default.slurm" | awk '{print $NF}')

echo "sft2 train job: ${TRAIN_JOB}"
echo "log: ${TRAIN_OUT}/sft2_train_${TRAIN_JOB}.log"
