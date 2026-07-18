#!/usr/bin/env bash
set -euo pipefail

: "${HOLD_JOB:?set HOLD_JOB}"
: "${REPO:?set REPO}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"
: "${EXPECTED_VAGEN_COMMIT:?set EXPECTED_VAGEN_COMMIT}"
: "${EXPECTED_VERL_COMMIT:?set EXPECTED_VERL_COMMIT}"
: "${MODEL:?set MODEL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${WANDB_PROJECT:?set WANDB_PROJECT}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME}"
: "${WANDB_RUN_ID:?set WANDB_RUN_ID}"

SLURM=/cm/shared/apps/slurm/current/bin
PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/trainer_done.flag" "${OUTPUT_DIR}/env_url.txt" \
  "${OUTPUT_DIR}/env_ready.flag"

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain)" ]]
[[ "$(git -C "${REPO}/external/VAGEN" rev-parse HEAD)" == "${EXPECTED_VAGEN_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/VAGEN/verl" rev-parse HEAD)" == "${EXPECTED_VERL_COMMIT}" ]]

# Both heterogeneous components must be running in normal, under this one hold.
for group in 0 1; do
  line=$(${SLURM}/scontrol show job "${HOLD_JOB}+${group}" -o)
  [[ "${line}" == *"JobState=RUNNING"* && "${line}" == *"Partition=normal"* ]] || {
    echo "hold group${group} is not RUNNING on normal: ${line}" >&2
    exit 2
  }
done
trainer_gpu_count=$(${SLURM}/srun --jobid="${HOLD_JOB}" --het-group=0 --overlap \
  --nodes=1 --ntasks=1 --gpus=8 bash -lc 'nvidia-smi -L | wc -l')
env_gpu_count=$(${SLURM}/srun --jobid="${HOLD_JOB}" --het-group=1 --overlap \
  --nodes=1 --ntasks=1 --gpus=1 bash -lc 'nvidia-smi -L | wc -l')
[[ "${trainer_gpu_count}" == 8 && "${env_gpu_count}" == 1 ]]

cleanup() {
  touch "${OUTPUT_DIR}/trainer_done.flag"
  if [[ -n "${ENV_STEP_PID:-}" ]]; then
    wait "${ENV_STEP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

${SLURM}/srun --jobid="${HOLD_JOB}" --het-group=1 --overlap \
  --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --kill-on-bad-exit=1 \
  --export="ALL,RUN_ROOT=${OUTPUT_DIR},ENV_REPO=${REPO},ENV_PORT=5000" \
  bash "${REPO}/experiments/training/rl/dynamic_env_server.slurm" \
  >"${OUTPUT_DIR}/env_step.log" 2>&1 &
ENV_STEP_PID=$!

for _ in $(seq 1 180); do
  [[ -s "${OUTPUT_DIR}/env_url.txt" ]] && break
  if ! kill -0 "${ENV_STEP_PID}" 2>/dev/null; then
    echo "environment step exited before readiness" >&2
    exit 1
  fi
  sleep 2
done
[[ -s "${OUTPUT_DIR}/env_url.txt" ]]
ENV_URL=$(<"${OUTPUT_DIR}/env_url.txt")
curl --fail --silent "${ENV_URL}/health" >/dev/null

export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm:${PYTHONPATH:-}"
"${PY}" "${REPO}/experiments/training/rl/preflight_dynamic_env.py" \
  --env-url "${ENV_URL}" \
  --output "${OUTPUT_DIR}/env_preflight.json" \
  --eval-set base_train --seed 30002 --timeout 300 \
  >"${OUTPUT_DIR}/env_preflight.log" 2>&1

echo "launch_time=$(date --iso-8601=seconds) hold=${HOLD_JOB} env=${ENV_URL}" \
  | tee "${OUTPUT_DIR}/topology.log"
set +e
${SLURM}/srun --jobid="${HOLD_JOB}" --het-group=0 --overlap \
  --nodes=1 --ntasks=1 --gpus=8 --cpus-per-task=64 --kill-on-bad-exit=1 \
  --export="ALL,REPO=${REPO},EXPECTED_COMMIT=${EXPECTED_COMMIT},EXPECTED_VAGEN_COMMIT=${EXPECTED_VAGEN_COMMIT},EXPECTED_VERL_COMMIT=${EXPECTED_VERL_COMMIT},MODEL=${MODEL},ENV_URL=${ENV_URL},OUTPUT_DIR=${OUTPUT_DIR},WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_NAME=${WANDB_RUN_NAME},WANDB_RUN_ID=${WANDB_RUN_ID}" \
  bash "${REPO}/experiments/training/rl/run_verl_online_world8_smoke.sh" \
  >"${OUTPUT_DIR}/trainer_step.log" 2>&1
TRAIN_RC=$?
set -e
touch "${OUTPUT_DIR}/trainer_done.flag"
wait "${ENV_STEP_PID}" || ENV_RC=$?
ENV_RC=${ENV_RC:-0}
if [[ ${TRAIN_RC} -ne 0 || ${ENV_RC} -ne 0 ]]; then
  echo "VERL_ONLINE_FAILED trainer_rc=${TRAIN_RC} env_rc=${ENV_RC}" >&2
  exit 1
fi

"${PY}" "${REPO}/experiments/training/rl/validate_verl_online_world8_smoke.py" \
  --output-dir "${OUTPUT_DIR}" \
  --commit "${EXPECTED_COMMIT}" \
  --vagen-commit "${EXPECTED_VAGEN_COMMIT}" \
  --verl-commit "${EXPECTED_VERL_COMMIT}" \
  --wandb-id "${WANDB_RUN_ID}" \
  >"${OUTPUT_DIR}/artifact_gate.log" 2>&1
cat "${OUTPUT_DIR}/artifact_gate.log"
echo VERL_ONLINE_WORLD8_ALL_OK
