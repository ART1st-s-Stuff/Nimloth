#!/usr/bin/env bash
#SBATCH --job-name=rl-k8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --nodelist=dgx-09
#SBATCH hetjob
#SBATCH --job-name=rl-k8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH hetjob
#SBATCH --job-name=rl-k8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH hetjob
#SBATCH --job-name=rl-k8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --nodelist=dgx-27
#SBATCH hetjob
#SBATCH --job-name=rl-k8-env-hold
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=dgx-48

# Placeholder allocation. The batch shell keeps all heterogeneous resources and
# executes atomically-published next_stage.sh files in the same allocation.
set -euo pipefail
: "${HOLD_ROOT:?must set HOLD_ROOT}"
: "${REPO:?must set REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
mkdir -p "${HOLD_ROOT}/stages"
rm -f "${HOLD_ROOT}/READY" "${HOLD_ROOT}/RELEASE" "${HOLD_ROOT}/next_stage.sh"
{
  echo "job_id=${SLURM_JOB_ID}"
  for group in 0 1 2 3 4; do
    name="SLURM_JOB_NODELIST_HET_GROUP_${group}"
    echo "group${group}=${!name:-missing}"
  done
  echo "commit=${EXPECTED_COMMIT}"
  echo "started=$(date --iso-8601=seconds)"
} | tee "${HOLD_ROOT}/allocation.txt"
touch "${HOLD_ROOT}/READY"

cleanup() {
  touch "${HOLD_ROOT}/RELEASED"
}
trap cleanup EXIT TERM INT

stage_index=0
while [[ ! -f "${HOLD_ROOT}/RELEASE" ]]; do
  if [[ -s "${HOLD_ROOT}/next_stage.sh" ]]; then
    stage_index=$((stage_index + 1))
    stage_dir=$(printf "%s/stages/%02d" "${HOLD_ROOT}" "${stage_index}")
    mkdir -p "${stage_dir}"
    mv "${HOLD_ROOT}/next_stage.sh" "${stage_dir}/command.sh"
    chmod 700 "${stage_dir}/command.sh"
    set +e
    bash "${stage_dir}/command.sh" >"${stage_dir}/stage.log" 2>&1
    stage_rc=$?
    set -e
    printf '%s\n' "${stage_rc}" >"${stage_dir}/exit_code.txt"
    date --iso-8601=seconds >"${stage_dir}/finished_at.txt"
    ln -sfn "${stage_dir}" "${HOLD_ROOT}/last_stage"
    if [[ "${stage_rc}" != 0 ]]; then
      touch "${stage_dir}/FAILED"
    else
      touch "${stage_dir}/PASSED"
    fi
  fi
  sleep 5
done
