#!/usr/bin/env bash
#SBATCH --job-name=rl-dyn-2plus1
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH hetjob
#SBATCH --job-name=rl-dyn-2plus1
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-51
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Atomic heterogeneous allocation: trainer2GPU on het-group0 and env1GPU on
# het-group1. Slurm grants both components together, so account/node limits
# cannot leave a live trainer waiting on a separately queued env job.
set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO}"
: "${ENV_REPO:?must set ENV_REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
: "${MODEL:?must set MODEL}"
: "${WANDB_RUN_ID:?must set WANDB_RUN_ID}"
: "${WANDB_RUN_NAME:?must set WANDB_RUN_NAME}"

module load slurm >/dev/null 2>&1 || true
SRUN=/cm/shared/apps/slurm/current/bin/srun
export RUN_ROOT REPO ENV_REPO EXPECTED_COMMIT MODEL WANDB_RUN_ID WANDB_RUN_NAME
rm -f \
  "${RUN_ROOT}/trainer_done.flag" \
  "${RUN_ROOT}/env_url.txt" \
  "${RUN_ROOT}/env_url.txt.tmp" \
  "${RUN_ROOT}/env_ready.flag"

env_cleanup() {
  touch "${RUN_ROOT}/trainer_done.flag"
  if [[ -n "${ENV_STEP_PID:-}" ]]; then
    kill "${ENV_STEP_PID}" 2>/dev/null || true
    wait "${ENV_STEP_PID}" 2>/dev/null || true
  fi
}
trap env_cleanup EXIT TERM INT

"${SRUN}" --het-group=1 --overlap --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  bash "${REPO}/experiments/training/rl/dynamic_env_server.slurm" \
  >"${RUN_ROOT}/env_hetero_step.out" \
  2>"${RUN_ROOT}/env_hetero_step.err" &
ENV_STEP_PID=$!
echo "env_het_step_pid=${ENV_STEP_PID}"

set +e
AUTO_SUBMIT_ENV=0 "${SRUN}" --het-group=0 --overlap --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 \
  bash "${REPO}/experiments/training/rl/dynamic_fsdp_smoke.slurm"
TRAIN_RC=$?
set -e

touch "${RUN_ROOT}/trainer_done.flag"
wait "${ENV_STEP_PID}" || ENV_RC=$?
ENV_RC=${ENV_RC:-0}
echo "hetero_smoke trainer_rc=${TRAIN_RC} env_rc=${ENV_RC}"
[[ "${TRAIN_RC}" == 0 && "${ENV_RC}" == 0 ]]
