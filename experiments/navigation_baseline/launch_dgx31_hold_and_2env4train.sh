#!/bin/bash
set -euo pipefail
module load slurm >/dev/null 2>&1 || true
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=/project/peilab/atst/nimloth/experiments/navigation_baseline
HOLD_JOB=452573
empty_streak=0
for i in $(seq 1 720); do
  state=$($SL/squeue -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
  if [ "$state" = "RUNNING" ]; then
    echo "hold ${HOLD_JOB} RUNNING at $(date)"
    break
  fi
  if [ -z "$state" ]; then
    empty_streak=$((empty_streak + 1))
    if [ "$empty_streak" -ge 6 ]; then
      echo "hold ${HOLD_JOB} gone after retries"
      exit 1
    fi
  else
    empty_streak=0
  fi
  sleep 10
done
state=$($SL/squeue -j "${HOLD_JOB}" -h -o '%T' 2>/dev/null || true)
if [ "$state" != "RUNNING" ]; then
  echo "hold ${HOLD_JOB} not RUNNING: ${state:-missing}"
  exit 1
fi
echo "launching train via srun --jobid=${HOLD_JOB} at $(date)"
$SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w dgx-31 \
  bash "${SCRIPTDIR}/resume_retry2_train_from50_dgx31_2env4train.slurm" \
  > "${SCRIPTDIR}/dgx31_2env4train_srun_${HOLD_JOB}.log" 2>&1
rc=$?
echo "train finished rc=${rc}; cancelling hold ${HOLD_JOB}"
$SL/scancel "${HOLD_JOB}" >/dev/null 2>&1 || true
exit $rc
