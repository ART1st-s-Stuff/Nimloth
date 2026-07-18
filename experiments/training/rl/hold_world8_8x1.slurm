#!/usr/bin/env bash
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
#SBATCH hetjob
#SBATCH --job-name=rl-world8-hold
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=dgx-29,dgx-32
# One bash allocation for all world8 resources. Actual probes are atomically
# published as next_stage.sh and run via nested srun without releasing GPUs.
set -euo pipefail
: "${HOLD_ROOT:?must set HOLD_ROOT}"
: "${REPO:?must set REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]
mkdir -p "${HOLD_ROOT}/stages"
rm -f "${HOLD_ROOT}/READY" "${HOLD_ROOT}/RELEASE"
# next_stage.sh may be pre-published before sbatch so an allocation can start
# and execute unattended without a second Slurm job or a login-node watcher.
{
  echo "job_id=${SLURM_JOB_ID}"
  for group in 0 1 2 3 4 5 6 7; do
    variable="SLURM_JOB_NODELIST_HET_GROUP_${group}"
    echo "group${group}=${!variable:-missing}"
  done
  echo "commit=${EXPECTED_COMMIT}"
  echo "started=$(date --iso-8601=seconds)"
} | tee "${HOLD_ROOT}/allocation.txt"
touch "${HOLD_ROOT}/READY"

cleanup() { touch "${HOLD_ROOT}/RELEASED"; }
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
    if [[ "${stage_rc}" == 0 ]]; then
      touch "${stage_dir}/PASSED"
    else
      touch "${stage_dir}/FAILED"
    fi
  fi
  sleep 3
done
