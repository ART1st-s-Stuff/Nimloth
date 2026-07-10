#!/bin/bash
#SBATCH --job-name=vagen-k8-full-6g
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:6
#SBATCH --mem=180G

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/sft1
RUN_DIR=${ROLLOUT_RUN_DIR:?ROLLOUT_RUN_DIR required}
LOG=${RUN_DIR}/preempt_6gpu_${SLURM_JOB_ID}.log
mkdir -p "${RUN_DIR}" "${REPO}/outputs/experiments/training/sft1/slurm"

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES required}"
if [ "${#ALLOCATED_GPUS[@]}" -ne 6 ]; then
  echo "ERROR expected exactly 6 allocated GPUs, got ${#ALLOCATED_GPUS[@]}: ${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG}"
  exit 2
fi
ENV_GPUS=""
POLICY_GPUS=""
ENV_PID=""

cleanup() {
  set +e
  if [ -n "${ENV_PID}" ] && kill -0 "${ENV_PID}" 2>/dev/null; then
    kill "${ENV_PID}" 2>/dev/null || true
    wait "${ENV_PID}" 2>/dev/null || true
  fi
}
on_signal() {
  exit 143
}
trap cleanup EXIT
trap on_signal TERM INT

{
  echo "=== full rollout 6-GPU preempt orchestration ==="
  echo "time=$(date -Is) job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
  echo "commit=$(git -C "${REPO}" rev-parse HEAD)"
  echo "allocated_gpus=${CUDA_VISIBLE_DEVICES}"
  echo "run_dir=${RUN_DIR} init_hf=${INIT_HF} source_step=${INIT_HF_STEP} dump_step=${ROLLOUT_DUMP_STEP:-0}"
} | tee -a "${LOG}"

# AI2-THOR needs working Vulkan/CloudRendering GPUs, while policy vLLM only
# needs CUDA. Probe all six first so one bad Vulkan mapping does not reject an
# otherwise usable six-GPU allocation.
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/../baseline/setup_ai2thor_env.sh"
declare -a GOOD_ENV_GPUS=()
for gpu in "${ALLOCATED_GPUS[@]}"; do
  gpu=$(echo "${gpu}" | xargs)
  echo "=== orchestrator AI2-THOR smoke gpu=${gpu} ===" | tee -a "${LOG}"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" timeout 150s "${PYTHON_ENV}/bin/python3" - <<'PY' >> "${LOG}" 2>&1
import ai2thor.controller
from ai2thor.platform import CloudRendering

controller = ai2thor.controller.Controller(
    agentMode="default",
    gridSize=0.1,
    visibilityDistance=10,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=255,
    height=255,
    fieldOfView=100,
    platform=CloudRendering,
    gpu_device=0,
    server_timeout=60,
    server_start_timeout=60,
)
try:
    controller.reset(scene="FloorPlan1")
finally:
    controller.stop()
print("OK", flush=True)
PY
  rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    GOOD_ENV_GPUS+=("${gpu}")
  else
    echo "WARN gpu=${gpu} unsuitable for AI2-THOR rc=${rc}; reserve for CUDA policy" | tee -a "${LOG}"
  fi
done
if [ "${#GOOD_ENV_GPUS[@]}" -lt 4 ]; then
  echo "ERROR need at least four AI2-THOR-good GPUs among six, got ${#GOOD_ENV_GPUS[@]}" | tee -a "${LOG}"
  exit 4
fi
SELECTED_ENV=("${GOOD_ENV_GPUS[@]:0:4}")
declare -a SELECTED_POLICY=()
for gpu in "${ALLOCATED_GPUS[@]}"; do
  gpu=$(echo "${gpu}" | xargs)
  selected=0
  for env_gpu in "${SELECTED_ENV[@]}"; do
    if [ "${gpu}" = "${env_gpu}" ]; then selected=1; break; fi
  done
  if [ "${selected}" -eq 0 ]; then SELECTED_POLICY+=("${gpu}"); fi
done
if [ "${#SELECTED_POLICY[@]}" -ne 2 ]; then
  echo "ERROR expected two policy GPUs after env selection, got ${#SELECTED_POLICY[@]}" | tee -a "${LOG}"
  exit 4
fi
ENV_GPUS=$(IFS=,; echo "${SELECTED_ENV[*]}")
POLICY_GPUS=$(IFS=,; echo "${SELECTED_POLICY[*]}")
echo "selected env_gpus=${ENV_GPUS} policy_gpus=${POLICY_GPUS}" | tee -a "${LOG}"

CONTROL_DIR=${RUN_DIR}/external_env_4gpu
READY=${CONTROL_DIR}/ready
FAILED=${CONTROL_DIR}/failed
URLS=${CONTROL_DIR}/env_urls.txt
mkdir -p "${CONTROL_DIR}"
# The parent can observe a marker left by an earlier failed attempt before the
# child launcher gets scheduled and clears it. Clear attempt-scoped markers in
# the parent before spawning to remove that startup race.
rm -f "${READY}" "${FAILED}"
: > "${CONTROL_DIR}/env_hosts.txt"
: > "${URLS}"

(
  export CUDA_VISIBLE_DEVICES="${ENV_GPUS}"
  exec bash "${SCRIPTDIR}/env_external_4gpu.slurm"
) >> "${LOG}" 2>&1 &
ENV_PID=$!

for _ in $(seq 1 180); do
  if [ -f "${FAILED}" ]; then
    echo "ERROR external environment marked failed" | tee -a "${LOG}"
    exit 4
  fi
  if [ -f "${READY}" ] && [ -s "${URLS}" ]; then
    break
  fi
  if ! kill -0 "${ENV_PID}" 2>/dev/null; then
    echo "ERROR external environment launcher exited before ready" | tee -a "${LOG}"
    wait "${ENV_PID}" || true
    exit 4
  fi
  sleep 5
done
if [ ! -f "${READY}" ] || [ ! -s "${URLS}" ]; then
  echo "ERROR timed out waiting for external environment" | tee -a "${LOG}"
  exit 4
fi
while read -r url; do
  [ -z "${url}" ] && continue
  curl -fsS "${url}/health" >/dev/null
  echo "healthy ${url}" | tee -a "${LOG}"
done < "${URLS}"

# One 2-GPU policy task runs at a time. Completed non-empty shards are skipped
# by rollouts_greedy_parallel.slurm, so a later retry preserves shard progress.
for task_id in 0 1 2 3; do
  echo "=== starting rollout task ${task_id} at $(date -Is) ===" | tee -a "${LOG}"
  (
    export CUDA_VISIBLE_DEVICES="${POLICY_GPUS}"
    export SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
    export SLURM_ARRAY_TASK_ID="${task_id}"
    export SLURM_CPUS_PER_TASK=112
    exec bash "${SCRIPTDIR}/rollouts_greedy_parallel.slurm"
  ) >> "${LOG}" 2>&1
  echo "=== completed rollout task ${task_id} at $(date -Is) ===" | tee -a "${LOG}"
done

touch "${RUN_DIR}/preempt_6gpu_rollout_done.flag"
echo "=== all rollout tasks completed at $(date -Is) ===" | tee -a "${LOG}"
