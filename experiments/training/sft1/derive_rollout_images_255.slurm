#!/bin/bash
#SBATCH --job-name=rollout-img255
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
PYTHON_ENV=${PYTHON_ENV:-${REPO}/.venv-vagen-main}
SOURCE_RECORDS_ROOT=${SOURCE_RECORDS_ROOT:?set SOURCE_RECORDS_ROOT}
OUTPUT_RECORDS_ROOT=${OUTPUT_RECORDS_ROOT:?set OUTPUT_RECORDS_ROOT}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}
LOG=${OUTPUT_RECORDS_ROOT}/derive_images_255.log

if [ "$(realpath -m "${SOURCE_RECORDS_ROOT}")" = "$(realpath -m "${OUTPUT_RECORDS_ROOT}")" ]; then
  echo "ERROR output records root must differ from source root" >&2
  exit 2
fi
mkdir -p "${OUTPUT_RECORDS_ROOT}" "${REPO}/outputs/experiments/training/sft1/slurm"
{
  echo "=== rollout image derivation starting at $(date -Iseconds) ==="
  echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
  echo "git=$(git -C "${REPO}" rev-parse HEAD)"
  echo "source=${SOURCE_RECORDS_ROOT}"
  echo "output=${OUTPUT_RECORDS_ROOT}"
  echo "workers=${WORKERS} target=255x255 mode=RGB resampling=BICUBIC"
} | tee "${LOG}"

"${PYTHON_ENV}/bin/python3" \
  "${REPO}/experiments/training/sft1/derive_rollout_images_255.py" \
  --source-records-root "${SOURCE_RECORDS_ROOT}" \
  --output-records-root "${OUTPUT_RECORDS_ROOT}" \
  --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"

echo "=== rollout image derivation finished at $(date -Iseconds) ===" | tee -a "${LOG}"
