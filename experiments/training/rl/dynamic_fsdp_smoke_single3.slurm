#!/usr/bin/env bash
#SBATCH --job-name=rl-dyn-single3
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=192G

# One atomic single-node allocation: one dedicated env GPU plus two trainer GPUs.
set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${REPO:?must set REPO}"
: "${ENV_REPO:?must set ENV_REPO}"
: "${EXPECTED_COMMIT:?must set EXPECTED_COMMIT}"
: "${MODEL:?must set MODEL}"
: "${WANDB_RUN_ID:?must set WANDB_RUN_ID}"
: "${WANDB_RUN_NAME:?must set WANDB_RUN_NAME}"

VISIBLE=${CUDA_VISIBLE_DEVICES:-0,1,2}
IFS=',' read -r -a GPUS <<<"${VISIBLE}"
if (( ${#GPUS[@]} < 3 )); then
  echo "single3 requires three allocated GPUs; CUDA_VISIBLE_DEVICES=${VISIBLE}" >&2
  exit 1
fi
ENV_GPU=${GPUS[0]}
TRAIN_GPUS=${GPUS[1]},${GPUS[2]}
export RUN_ROOT REPO ENV_REPO EXPECTED_COMMIT MODEL WANDB_RUN_ID WANDB_RUN_NAME
rm -f \
  "${RUN_ROOT}/trainer_done.flag" \
  "${RUN_ROOT}/env_url.txt" \
  "${RUN_ROOT}/env_url.txt.tmp" \
  "${RUN_ROOT}/env_ready.flag"

env_cleanup() {
  touch "${RUN_ROOT}/trainer_done.flag"
  if [[ -n "${ENV_PID:-}" ]]; then
    kill "${ENV_PID}" 2>/dev/null || true
    wait "${ENV_PID}" 2>/dev/null || true
  fi
}
trap env_cleanup EXIT TERM INT

(
  export CUDA_VISIBLE_DEVICES="${ENV_GPU}"
  export ENV_CUDA_VISIBLE_DEVICES="${ENV_GPU}"
  exec bash "${REPO}/experiments/training/rl/dynamic_env_server.slurm"
) >"${RUN_ROOT}/env_single3_step.out" 2>"${RUN_ROOT}/env_single3_step.err" &
ENV_PID=$!
echo "single3 visible=${VISIBLE} env_gpu=${ENV_GPU} trainer_gpus=${TRAIN_GPUS} env_pid=${ENV_PID}"

set +e
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" AUTO_SUBMIT_ENV=0 \
  bash "${REPO}/experiments/training/rl/dynamic_fsdp_smoke.slurm"
TRAIN_RC=$?
set -e

touch "${RUN_ROOT}/trainer_done.flag"
wait "${ENV_PID}" || ENV_RC=$?
ENV_RC=${ENV_RC:-0}
echo "single3_smoke trainer_rc=${TRAIN_RC} env_rc=${ENV_RC}"
[[ "${TRAIN_RC}" == 0 && "${ENV_RC}" == 0 ]]
