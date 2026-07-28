#!/bin/bash
#SBATCH --job-name=sft2-terminal-migrate
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

REPO=${REPO:?set REPO to the clean server worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
SOURCE_TRAIN_JSONL=${SOURCE_TRAIN_JSONL:?set SOURCE_TRAIN_JSONL}
SOURCE_VAL_JSONL=${SOURCE_VAL_JSONL:?set SOURCE_VAL_JSONL}
SOURCE_TRAIN_SHA256=${SOURCE_TRAIN_SHA256:?set SOURCE_TRAIN_SHA256}
SOURCE_VAL_SHA256=${SOURCE_VAL_SHA256:?set SOURCE_VAL_SHA256}
OUTPUT_TRAIN_JSONL=${OUTPUT_TRAIN_JSONL:?set OUTPUT_TRAIN_JSONL}
OUTPUT_VAL_JSONL=${OUTPUT_VAL_JSONL:?set OUTPUT_VAL_JSONL}
MIGRATION_LOG=${MIGRATION_LOG:?set MIGRATION_LOG}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv-vagen-main}

actual_commit=$(git -C "${REPO}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "ERROR commit mismatch: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR remote worktree has tracked changes: ${REPO}" >&2
  exit 1
fi
for source in "${SOURCE_TRAIN_JSONL}" "${SOURCE_VAL_JSONL}"; do
  if [[ ! -s "${source}" ]]; then
    echo "ERROR source JSONL missing: ${source}" >&2
    exit 1
  fi
done
if [[ -e "${OUTPUT_TRAIN_JSONL}" || -e "${OUTPUT_VAL_JSONL}" ]]; then
  echo "ERROR refusing to overwrite migrated terminal-CoT output" >&2
  exit 1
fi

actual_train_sha=$(sha256sum "${SOURCE_TRAIN_JSONL}" | awk '{print $1}')
actual_val_sha=$(sha256sum "${SOURCE_VAL_JSONL}" | awk '{print $1}')
if [[ "${actual_train_sha}" != "${SOURCE_TRAIN_SHA256}" ]]; then
  echo "ERROR train terminal-CoT SHA256 changed" >&2
  exit 1
fi
if [[ "${actual_val_sha}" != "${SOURCE_VAL_SHA256}" ]]; then
  echo "ERROR val terminal-CoT SHA256 changed" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_TRAIN_JSONL}")" "$(dirname "${MIGRATION_LOG}")"
export VIRTUAL_ENV=${PYTHON_ENV}
export PATH=${PYTHON_ENV}/bin:${REPO}/.local/bin:${PATH}
export PYTHONPATH=${REPO}/src:${REPO}/external/VAGEN:${REPO}/external/le-wm

{
  echo "migration_start=$(date --iso-8601=seconds)"
  echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
  echo "commit=${actual_commit}"
  echo "source_train=${SOURCE_TRAIN_JSONL} sha256=${actual_train_sha}"
  echo "source_val=${SOURCE_VAL_JSONL} sha256=${actual_val_sha}"
  echo "output_train=${OUTPUT_TRAIN_JSONL}"
  echo "output_val=${OUTPUT_VAL_JSONL}"
} | tee -a "${MIGRATION_LOG}"

for split in train val; do
  if [[ "${split}" == "train" ]]; then
    source_jsonl=${SOURCE_TRAIN_JSONL}
    output_jsonl=${OUTPUT_TRAIN_JSONL}
  else
    source_jsonl=${SOURCE_VAL_JSONL}
    output_jsonl=${OUTPUT_VAL_JSONL}
  fi
  "${PYTHON_ENV}/bin/python3" -m nimloth.rollout.migration \
    --source "${source_jsonl}" \
    --output "${output_jsonl}" \
    --missing-action-space-id navigation \
    --missing-action-space-version 1 \
    --missing-reward-provenance trajectory_terminal_reward \
    2>&1 | tee -a "${MIGRATION_LOG}"
  "${PYTHON_ENV}/bin/python3" -m nimloth.rollout.migration_validation \
    --source "${source_jsonl}" \
    --output "${output_jsonl}" \
    --missing-action-space-id navigation \
    --missing-action-space-version 1 \
    --missing-reward-provenance trajectory_terminal_reward \
    2>&1 | tee -a "${MIGRATION_LOG}"
done

for manifest in \
  "${OUTPUT_TRAIN_JSONL}.manifest.json" \
  "${OUTPUT_VAL_JSONL}.manifest.json"; do
  if [[ ! -s "${manifest}" ]]; then
    echo "ERROR migration manifest missing: ${manifest}" >&2
    exit 1
  fi
done
echo "migration_done=$(date --iso-8601=seconds)" | tee -a "${MIGRATION_LOG}"
