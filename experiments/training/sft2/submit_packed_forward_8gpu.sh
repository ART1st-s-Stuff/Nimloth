#!/bin/bash
# Submit SFT2 with optional packed-forward (opt-in; not enabled by default).
set -euo pipefail

REPO=/project/peilab/atst/nimloth
ROOT=${REPO}/experiments/training/sft2
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
mkdir -p "${REPO}/outputs/experiments/training/sft2/slurm"

SFT1_EPOCH=${SFT1_EPOCH:-2}
SFT2_NODE=${SFT2_NODE:-}
NGPUS=${NGPUS:-8}
SFT2_LLM_TUNE=${SFT2_LLM_TUNE:-freeze}
SFT2_VISION_TUNE=${SFT2_VISION_TUNE:-full}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft2_packed_forward_8gpu}
TRAIN_OUT=${TRAIN_OUT:-${REPO}/outputs/experiments/training/sft2/${RUN_DATE}/${EXPERIMENT_NAME}}
PACKED_FORWARD=${PACKED_FORWARD:-0}

echo "=== Submit SFT2 packed-forward (${NGPUS} GPU) ==="
echo "tune: llm=${SFT2_LLM_TUNE} vision=${SFT2_VISION_TUNE}"
echo "output: ${TRAIN_OUT}"
echo "packed_forward: ${PACKED_FORWARD}"
echo "preprocess cache: ${PREPROCESS_CACHE_DIR:-<disabled>}"
echo "node: ${SFT2_NODE:-<scheduler>}"

SBATCH_ARGS=(--account=peilab --job-name="sft2-packed-8g")
if [ -n "${SFT2_NODE}" ]; then
  SBATCH_ARGS+=(--nodelist="${SFT2_NODE}")
fi

EXPORT_VARS="ALL,SFT2_LLM_TUNE=${SFT2_LLM_TUNE},SFT2_VISION_TUNE=${SFT2_VISION_TUNE},NGPUS=${NGPUS},SFT1_EPOCH=${SFT1_EPOCH},SKIP_SFT1_DONE=1,TRAIN_OUT_OVERRIDE=${TRAIN_OUT},EXPERIMENT_NAME=${EXPERIMENT_NAME},RUN_DATE=${RUN_DATE},PACKED_FORWARD=${PACKED_FORWARD},STEP_TIMING=${STEP_TIMING:-0}"
if [ -n "${PREPROCESS_CACHE_DIR:-}" ]; then
  EXPORT_VARS="${EXPORT_VARS},PREPROCESS_CACHE_DIR=${PREPROCESS_CACHE_DIR}"
fi

J=$($SLURM "${SBATCH_ARGS[@]}" \
  --export="${EXPORT_VARS}" \
  "${ROOT}/train_vagen79_default.slurm" | awk '{print $NF}')
echo "sft2 job: ${J}"
echo "log: ${TRAIN_OUT}/sft2_train_${J}.log"
