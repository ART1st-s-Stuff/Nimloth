#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
ROOT=${REPO}/experiments/training/sft2

cache_output=$(bash "${ROOT}/submit_compact_cache.sh")
echo "${cache_output}"
cache_job=$(awk '{print $NF}' <<<"${cache_output}")
if [[ ! "${cache_job}" =~ ^[0-9]+$ ]]; then
  echo "ERROR could not parse compact cache job id: ${cache_job}" >&2
  exit 2
fi

export REQUIRE_PREBUILT_CACHE=1
export SFT2_DEPENDENCY="afterok:${cache_job}"
train_output=$(bash "${ROOT}/submit_default_8gpu.sh")
echo "${train_output}"
echo "SFT2 training is dependency-gated on compact cache job ${cache_job}."
