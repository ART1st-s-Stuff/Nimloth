#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
ROOT=${REPO}/experiments/training/sft1
SLURM=/cm/shared/apps/slurm/current/bin/sbatch
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

: "${TRAIN_OUT:?TRAIN_OUT required}"
: "${TRAIN_JSONL:?TRAIN_JSONL required}"
: "${VAL_JSONL:?VAL_JSONL required}"
: "${INIT_HF:?INIT_HF required}"

cache_job=$(${SLURM} --parsable --export=ALL "${ROOT}/build_preprocess_cache.slurm")
echo "Submitted SFT1 cache job ${cache_job}"

export REQUIRE_PREBUILT_CACHE=1
export SFT1_DEPENDENCY="afterok:${cache_job}"
train_output=$(bash "${ROOT}/submit_train_8gpu.sh")
echo "${train_output}"
echo "SFT1 training is dependency-gated on cache job ${cache_job}."
