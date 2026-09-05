#!/bin/bash
#SBATCH --job-name=step60-gate-8gpu
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:8
#SBATCH --mem=480G

set -euo pipefail

required=(
  NIMLOTH_ROOT EXPECTED_NIMLOTH_COMMIT PYTHON_EXECUTABLE VAGEN_RUNTIME_ROOT MODEL_PATH
  PARTITION_MANIFEST SOURCE_RUNTIME_CONTRACT RUN_ROOT RUN_ID_PREFIX RUN_MODE
  ENV_SERVICE_ENTRYPOINT COLLECTOR_ENTRYPOINT SHARD_STATE_ENTRYPOINT
  EXPECTED_RECONSTRUCTION_HEAD EXPECTED_RECONSTRUCTION_TREE
  EXPECTED_RECONSTRUCTION_DIFF_SHA256 EXPECTED_RUNTIME_CONTRACT_PAYLOAD_SHA256
  SMOKE_FORMAT_FAILURE_POLICY SMOKE_COLLECTOR_CONCURRENCY
  GATE_FORMAT_FAILURE_POLICY GATE_COLLECTOR_CONCURRENCY
  GPU_MEMORY_UTILIZATION ENGINE_SEED
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then echo "ERROR required environment variable is unset: ${name}" >&2; exit 2; fi
done
for name in NIMLOTH_ROOT PYTHON_EXECUTABLE VAGEN_RUNTIME_ROOT MODEL_PATH PARTITION_MANIFEST SOURCE_RUNTIME_CONTRACT RUN_ROOT ENV_SERVICE_ENTRYPOINT COLLECTOR_ENTRYPOINT SHARD_STATE_ENTRYPOINT; do
  if [[ "${!name}" != /* ]]; then echo "ERROR ${name} must be an absolute path" >&2; exit 2; fi
done
[[ "${RUN_MODE}" == fresh || "${RUN_MODE}" == resume ]] || { echo "ERROR RUN_MODE must be fresh or resume" >&2; exit 2; }
[[ "${SMOKE_FORMAT_FAILURE_POLICY}" == fail_shard && "${SMOKE_COLLECTOR_CONCURRENCY}" == 1 ]] || { echo "ERROR smoke requires fail_shard and concurrency 1" >&2; exit 2; }
[[ "${GATE_FORMAT_FAILURE_POLICY}" == exclude_trajectory ]] || { echo "ERROR gate requires exclude_trajectory" >&2; exit 2; }
[[ "${GATE_COLLECTOR_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR gate concurrency must be positive" >&2; exit 2; }
[[ -d "${NIMLOTH_ROOT}" && -d "${VAGEN_RUNTIME_ROOT}" && -d "${MODEL_PATH}" ]] || { echo "ERROR required directory is missing" >&2; exit 2; }
[[ -x "${PYTHON_EXECUTABLE}" && -x "${COLLECTOR_ENTRYPOINT}" && -x "${SHARD_STATE_ENTRYPOINT}" ]] || { echo "ERROR Python executable/entrypoint is not executable" >&2; exit 2; }
[[ -f "${ENV_SERVICE_ENTRYPOINT}" && ! -L "${ENV_SERVICE_ENTRYPOINT}" && -r "${ENV_SERVICE_ENTRYPOINT}" ]] || { echo "ERROR ENV_SERVICE_ENTRYPOINT must be a readable regular non-symlink file" >&2; exit 2; }
[[ -f "${PARTITION_MANIFEST}" && -f "${SOURCE_RUNTIME_CONTRACT}" ]] || { echo "ERROR required contract file is missing" >&2; exit 2; }
export PYTHONPATH="${NIMLOTH_ROOT}/src:${NIMLOTH_ROOT}:${PYTHONPATH:-}"
export PYTHON_ENV="$(dirname "$(dirname "${PYTHON_EXECUTABLE}")")"
[[ -d "${PYTHON_ENV}" ]] || { echo "ERROR derived PYTHON_ENV is missing: ${PYTHON_ENV}" >&2; exit 2; }

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES required}"
if [[ "${#ALLOCATED_GPUS[@]}" -ne 8 ]]; then echo "ERROR expected exactly eight allocated GPUs" >&2; exit 2; fi
for i in "${!ALLOCATED_GPUS[@]}"; do
  ALLOCATED_GPUS[$i]="${ALLOCATED_GPUS[$i]//[[:space:]]/}"
  [[ -n "${ALLOCATED_GPUS[$i]}" ]] || { echo "ERROR empty allocated GPU identity" >&2; exit 2; }
done
if [[ "$(printf '%s\n' "${ALLOCATED_GPUS[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "ERROR allocated GPU identities are not distinct" >&2; exit 2
fi
ENV_GPUS=("${ALLOCATED_GPUS[@]:0:4}")
POLICY_GPUS=("${ALLOCATED_GPUS[@]:4:4}")

IDENTITY_FIELDS=(
  "run_id_prefix=${RUN_ID_PREFIX}"
  "nimloth_root=${NIMLOTH_ROOT}"
  "expected_nimloth_commit=${EXPECTED_NIMLOTH_COMMIT}"
  "python_executable=${PYTHON_EXECUTABLE}"
  "vagen_runtime_root=${VAGEN_RUNTIME_ROOT}"
  "model_path=${MODEL_PATH}"
  "partition_manifest=${PARTITION_MANIFEST}"
  "source_runtime_contract=${SOURCE_RUNTIME_CONTRACT}"
  "env_service_entrypoint=${ENV_SERVICE_ENTRYPOINT}"
  "collector_entrypoint=${COLLECTOR_ENTRYPOINT}"
  "shard_state_entrypoint=${SHARD_STATE_ENTRYPOINT}"
  "expected_reconstruction_head=${EXPECTED_RECONSTRUCTION_HEAD}"
  "expected_reconstruction_tree=${EXPECTED_RECONSTRUCTION_TREE}"
  "expected_reconstruction_diff_sha256=${EXPECTED_RECONSTRUCTION_DIFF_SHA256}"
  "expected_runtime_contract_payload_sha256=${EXPECTED_RUNTIME_CONTRACT_PAYLOAD_SHA256}"
  "smoke_format_failure_policy=${SMOKE_FORMAT_FAILURE_POLICY}"
  "smoke_collector_concurrency=${SMOKE_COLLECTOR_CONCURRENCY}"
  "gate_format_failure_policy=${GATE_FORMAT_FAILURE_POLICY}"
  "gate_collector_concurrency=${GATE_COLLECTOR_CONCURRENCY}"
  "tensor_parallel_size=1"
  "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "engine_seed=${ENGINE_SEED}"
  "environment_service_count=4"
  "gate_shards=4"
  "gate_shard_size=100"
)
INIT_ARGS=(
  "${SHARD_STATE_ENTRYPOINT}" initialize-run "${RUN_ROOT}"
  --run-mode "${RUN_MODE}" --nimloth-root "${NIMLOTH_ROOT}"
  --expected-nimloth-commit "${EXPECTED_NIMLOTH_COMMIT}"
)
for field in "${IDENTITY_FIELDS[@]}"; do INIT_ARGS+=(--identity-field "${field}"); done
"${PYTHON_EXECUTABLE}" "${INIT_ARGS[@]}"

JOB_ID=${SLURM_JOB_ID:-local}
ATTEMPT_ID="${JOB_ID}-$(date -u +%Y%m%dT%H%M%S%NZ)-$$"
ATTEMPT_DIR="${RUN_ROOT}/attempts/${ATTEMPT_ID}"
LOG_DIR="${ATTEMPT_DIR}/logs"
CONTROL_DIR="${ATTEMPT_DIR}/control"
MAIN_LOG="${LOG_DIR}/gate-8gpu-${JOB_ID}.log"
PID_LOG="${CONTROL_DIR}/children.tsv"
STATUS_LOG="${ATTEMPT_DIR}/status.tsv"
ENV_PID=""
ORCHESTRATOR_SIGNAL=""
declare -a CHILD_PIDS=()

terminate_group() {
  local pid=$1
  # A dead group leader does not imply that its descendants have exited.
  kill -TERM -- "-${pid}" 2>/dev/null || true
}
cleanup() {
  local rc=$?
  trap - EXIT TERM INT
  set +e
  for pid in "${CHILD_PIDS[@]:-}"; do [[ -n "${pid}" ]] && terminate_group "${pid}"; done
  [[ -n "${ENV_PID}" ]] && terminate_group "${ENV_PID}"
  wait 2>/dev/null || true
  printf 'orchestrator\t%s\t%s\t%s\n' "$(date -Is)" "${rc}" "${ORCHESTRATOR_SIGNAL}" >> "${STATUS_LOG}" 2>/dev/null || true
  if [[ "${rc}" -ne 0 ]]; then
    local -a finalize_args=(
      "${SHARD_STATE_ENTRYPOINT}" finalize-run "${RUN_ROOT}"
      --exit-code "${rc}" --attempt-id "${ATTEMPT_ID}"
    )
    if [[ -n "${ORCHESTRATOR_SIGNAL}" ]]; then
      finalize_args+=(--orchestrator-signal "${ORCHESTRATOR_SIGNAL}")
    fi
    if ! "${PYTHON_EXECUTABLE}" "${finalize_args[@]}"; then
      printf 'finalize-run-error\t%s\n' "$(date -Is)" >> "${STATUS_LOG}" 2>/dev/null || true
    fi
  fi
  exit "${rc}"
}
record_signal_and_exit() {
  ORCHESTRATOR_SIGNAL=$1
  exit 143
}
trap cleanup EXIT
trap 'record_signal_and_exit TERM' TERM
trap 'record_signal_and_exit INT' INT

mkdir "${ATTEMPT_DIR}"
mkdir "${LOG_DIR}" "${CONTROL_DIR}"
(set -o noclobber; : > "${MAIN_LOG}"; : > "${PID_LOG}"; : > "${STATUS_LOG}")

{
  echo "time=$(date -Is) job=${JOB_ID} attempt=${ATTEMPT_ID} node=${SLURM_JOB_NODELIST:-local}"
  echo "nimloth_root=${NIMLOTH_ROOT} vagen_runtime_root=${VAGEN_RUNTIME_ROOT}"
  echo "model_path=${MODEL_PATH} partition_manifest=${PARTITION_MANIFEST} runtime_contract=${SOURCE_RUNTIME_CONTRACT}"
  echo "env_gpus=${ENV_GPUS[*]} policy_gpus=${POLICY_GPUS[*]}"
} | tee -a "${MAIN_LOG}"

COMMON_STATE_ARGS=(
  --partition-manifest "${PARTITION_MANIFEST}" --shard-size 100
  --source-runtime-root "${VAGEN_RUNTIME_ROOT}"
  --source-runtime-contract "${SOURCE_RUNTIME_CONTRACT}" --model-path "${MODEL_PATH}"
  --expected-runtime-contract-payload-sha256 "${EXPECTED_RUNTIME_CONTRACT_PAYLOAD_SHA256}"
  --expected-reconstruction-head "${EXPECTED_RECONSTRUCTION_HEAD}"
  --expected-reconstruction-tree "${EXPECTED_RECONSTRUCTION_TREE}"
  --expected-reconstruction-diff-sha256 "${EXPECTED_RECONSTRUCTION_DIFF_SHA256}"
  --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --engine-seed "${ENGINE_SEED}"
)
SMOKE_OUTPUT="${RUN_ROOT}/smoke/source-index-00000"
SMOKE_RUN_ID="${RUN_ID_PREFIX}-smoke-source-00000"
SMOKE_ITEM="smoke|${SMOKE_OUTPUT}|${SMOKE_RUN_ID}|source-index|0|${SMOKE_FORMAT_FAILURE_POLICY}|${SMOKE_COLLECTOR_CONCURRENCY}"
declare -a GATE_ITEMS=() GATE_STATES=()
for i in 0 1 2 3; do
  output=$(printf '%s/gate/shard-%03d' "${RUN_ROOT}" "$((i + 1))")
  run_id=$(printf '%s-gate-shard-%03d' "${RUN_ID_PREFIX}" "$((i + 1))")
  GATE_ITEMS[$i]="shard-${i}|${output}|${run_id}|shard-index|${i}|${GATE_FORMAT_FAILURE_POLICY}|${GATE_COLLECTOR_CONCURRENCY}"
done
INSPECTION_HANDOFF="${ATTEMPT_DIR}/inspection-handoff.json"
CLASSIFICATION_OUTPUT="${CONTROL_DIR}/classification.tsv"
CLASSIFY_ARGS=("${SHARD_STATE_ENTRYPOINT}" classify-batch "${COMMON_STATE_ARGS[@]}" --inspection-handoff "${INSPECTION_HANDOFF}" --item "${SMOKE_ITEM}")
for item in "${GATE_ITEMS[@]}"; do CLASSIFY_ARGS+=(--item "${item}"); done
(
  set -o noclobber
  "${PYTHON_EXECUTABLE}" "${CLASSIFY_ARGS[@]}" > "${CLASSIFICATION_OUTPUT}"
)
while IFS=$'\t' read -r label state; do
  [[ "${state}" == fresh || "${state}" == resume || "${state}" == complete ]] || { echo "ERROR invalid output state: ${state}" >&2; exit 3; }
  if [[ "${label}" == smoke ]]; then SMOKE_STATE=${state}; else GATE_STATES[${label#shard-}]=${state}; fi
done < "${CLASSIFICATION_OUTPUT}"
[[ -n "${SMOKE_STATE:-}" && "${#GATE_STATES[@]}" -eq 4 ]] || { echo "ERROR incomplete batch classification" >&2; exit 3; }
[[ -f "${INSPECTION_HANDOFF}" && ! -L "${INSPECTION_HANDOFF}" ]] || { echo "ERROR inspection handoff was not published" >&2; exit 3; }
INSPECTION_HANDOFF_SHA256=$(sha256sum "${INSPECTION_HANDOFF}" | awk '{print $1}')
[[ "${INSPECTION_HANDOFF_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR inspection handoff SHA256 is invalid" >&2; exit 3; }
VERIFY_HANDOFF_ARGS=("${SHARD_STATE_ENTRYPOINT}" verify-handoff "${COMMON_STATE_ARGS[@]}" --inspection-handoff "${INSPECTION_HANDOFF}" --expected-inspection-handoff-sha256 "${INSPECTION_HANDOFF_SHA256}" --item "${SMOKE_ITEM}")
for item in "${GATE_ITEMS[@]}"; do VERIFY_HANDOFF_ARGS+=(--item "${item}"); done
"${PYTHON_EXECUTABLE}" "${VERIFY_HANDOFF_ARGS[@]}"
printf 'classification\tcomplete\t%s\n' "${INSPECTION_HANDOFF_SHA256}" >> "${STATUS_LOG}"

ENV_CUDA=$(IFS=,; echo "${ENV_GPUS[*]}")
ENV_RUN_DIR="${ATTEMPT_DIR}/environment"
mkdir "${ENV_RUN_DIR}"
"${PYTHON_EXECUTABLE}" "${SHARD_STATE_ENTRYPOINT}" verify-nimloth \
  --nimloth-root "${NIMLOTH_ROOT}" --expected-nimloth-commit "${EXPECTED_NIMLOTH_COMMIT}"
export REPO="${NIMLOTH_ROOT}" VAGEN_DIR="${VAGEN_RUNTIME_ROOT}" PYTHON_ENV
export ROLLOUT_RUN_DIR="${ENV_RUN_DIR}" ROLLOUT_RUN_NAME="${RUN_ID_PREFIX}-environment-${ATTEMPT_ID}"
export CUDA_VISIBLE_DEVICES="${ENV_CUDA}" ENV_SERVICE_COUNT=4
setsid bash "${ENV_SERVICE_ENTRYPOINT}" > "${LOG_DIR}/environment-service.log" 2>&1 &
ENV_PID=$!
# Exported service inputs are inherited by the dedicated process group.
printf 'environment\t%s\n' "${ENV_PID}" >> "${PID_LOG}"

READY="${ENV_RUN_DIR}/external_env_4gpu/ready"
FAILED="${ENV_RUN_DIR}/external_env_4gpu/failed"
URLS_FILE="${ENV_RUN_DIR}/external_env_4gpu/env_urls.txt"
ready=0
for ((poll=0; poll<${ENV_READY_MAX_POLLS:-180}; poll++)); do
  [[ -e "${FAILED}" ]] && { echo "ERROR environment service failed" >&2; exit 4; }
  if [[ -f "${READY}" && -s "${URLS_FILE}" ]]; then ready=1; break; fi
  kill -0 "${ENV_PID}" 2>/dev/null || { echo "ERROR environment service exited before readiness" >&2; exit 4; }
  sleep "${ENV_READY_POLL_SECONDS:-5}"
done
[[ "${ready}" -eq 1 ]] || { echo "ERROR environment readiness timeout" >&2; exit 4; }
mapfile -t ENV_URLS < <(awk 'NF {print}' "${URLS_FILE}")
[[ "${#ENV_URLS[@]}" -eq 4 ]] || { echo "ERROR expected exactly four environment URLs" >&2; exit 4; }
[[ "$(printf '%s\n' "${ENV_URLS[@]}" | sort -u | wc -l)" -eq 4 ]] || { echo "ERROR environment URLs are not distinct" >&2; exit 4; }
for url in "${ENV_URLS[@]}"; do curl -fsS --max-time 10 "${url}/health" >/dev/null; done

run_collector() {
  local label=$1 output=$2 run_id=$3 kind=$4 index=$5 gpu=$6 url=$7 state=$8 policy=$9 concurrency=${10}
  [[ "${state}" == complete ]] && return 0
  local -a selector=("--${kind}" "${index}") resume=()
  [[ "${state}" == resume ]] && resume=(--resume)
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${NIMLOTH_ROOT}/src:${NIMLOTH_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_EXECUTABLE}" "${COLLECTOR_ENTRYPOINT}" \
    --model-path "${MODEL_PATH}" --partition-manifest "${PARTITION_MANIFEST}" \
    "${selector[@]}" --shard-size 100 --output-dir "${output}" --env-url "${url}" \
    --run-id "${run_id}" --source-runtime-root "${VAGEN_RUNTIME_ROOT}" \
    --source-runtime-contract "${SOURCE_RUNTIME_CONTRACT}" \
    --expected-reconstruction-head "${EXPECTED_RECONSTRUCTION_HEAD}" \
    --expected-reconstruction-tree "${EXPECTED_RECONSTRUCTION_TREE}" \
    --expected-reconstruction-diff-sha256 "${EXPECTED_RECONSTRUCTION_DIFF_SHA256}" \
    --expected-runtime-contract-payload-sha256 "${EXPECTED_RUNTIME_CONTRACT_PAYLOAD_SHA256}" \
    --format-failure-policy "${policy}" --concurrency "${concurrency}" \
    --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --engine-seed "${ENGINE_SEED}" \
    --inspection-handoff "${INSPECTION_HANDOFF}" \
    --expected-inspection-handoff-sha256 "${INSPECTION_HANDOFF_SHA256}" \
    "${resume[@]}" \
    > "${LOG_DIR}/${label}.log" 2>&1
}

remove_owned_child() {
  local completed_pid=$1 pid
  local -a remaining=()
  for pid in "${CHILD_PIDS[@]:-}"; do
    [[ -n "${pid}" && "${pid}" != "${completed_pid}" ]] && remaining+=("${pid}")
  done
  CHILD_PIDS=("${remaining[@]}")
}
wait_for_child() {
  local pid=$1 rc
  set +e
  wait "${pid}"
  rc=$?
  set -e
  return "${rc}"
}
export -f run_collector
export MODEL_PATH PARTITION_MANIFEST COLLECTOR_ENTRYPOINT VAGEN_RUNTIME_ROOT SOURCE_RUNTIME_CONTRACT
export EXPECTED_RECONSTRUCTION_HEAD EXPECTED_RECONSTRUCTION_TREE EXPECTED_RECONSTRUCTION_DIFF_SHA256
export EXPECTED_RUNTIME_CONTRACT_PAYLOAD_SHA256 GPU_MEMORY_UTILIZATION ENGINE_SEED
export INSPECTION_HANDOFF INSPECTION_HANDOFF_SHA256
export NIMLOTH_ROOT PYTHON_EXECUTABLE LOG_DIR PYTHONPATH
setsid bash -c 'run_collector "$@"' _ smoke "${SMOKE_OUTPUT}" "${SMOKE_RUN_ID}" source-index 0 "${POLICY_GPUS[0]}" "${ENV_URLS[0]}" "${SMOKE_STATE}" "${SMOKE_FORMAT_FAILURE_POLICY}" "${SMOKE_COLLECTOR_CONCURRENCY}" \
  > "${LOG_DIR}/smoke-launcher.log" 2>&1 &
SMOKE_PID=$!
CHILD_PIDS+=("${SMOKE_PID}")
printf 'smoke\t%s\n' "${SMOKE_PID}" >> "${PID_LOG}"
if wait_for_child "${SMOKE_PID}"; then
  remove_owned_child "${SMOKE_PID}"
else
  smoke_rc=$?
  echo "ERROR smoke collector failed rc=${smoke_rc}" >&2
  exit "${smoke_rc}"
fi
"${PYTHON_EXECUTABLE}" "${SHARD_STATE_ENTRYPOINT}" validate-batch "${COMMON_STATE_ARGS[@]}" \
  --inspection-handoff "${INSPECTION_HANDOFF}" \
  --expected-inspection-handoff-sha256 "${INSPECTION_HANDOFF_SHA256}" \
  --item "${SMOKE_ITEM}" >> "${LOG_DIR}/smoke-validation.log" 2>&1
printf 'smoke\tcomplete\n' >> "${STATUS_LOG}"

launch_gate_shards() {
  local i output run_id pid
  declare -A pid_names=()
  for i in 0 1 2 3; do
    output=$(printf '%s/gate/shard-%03d' "${RUN_ROOT}" "$((i + 1))")
    run_id=$(printf '%s-gate-shard-%03d' "${RUN_ID_PREFIX}" "$((i + 1))")
    if [[ "${GATE_STATES[$i]}" == complete ]]; then printf 'shard-%03d\tskipped-complete\n' "$((i + 1))" >> "${STATUS_LOG}"; continue; fi
    setsid bash -c 'run_collector "$@"' _ "shard-$i" "${output}" "${run_id}" shard-index "$i" "${POLICY_GPUS[$i]}" "${ENV_URLS[$i]}" "${GATE_STATES[$i]}" "${GATE_FORMAT_FAILURE_POLICY}" "${GATE_COLLECTOR_CONCURRENCY}" \
      > "${LOG_DIR}/shard-$i-launcher.log" 2>&1 &
    pid=$!
    CHILD_PIDS+=("${pid}")
    pid_names["${pid}"]=$(printf 'shard-%03d' "$((i + 1))")
    printf 'shard-%03d\t%s\n' "$((i + 1))" "${pid}" >> "${PID_LOG}"
  done
  local rc=0 completed_pid completed_label
  local -a active=("${CHILD_PIDS[@]}") next=()
  while [[ "${#active[@]}" -gt 0 ]]; do
    set +e
    wait -n -p completed_pid "${active[@]}"
    rc=$?
    set -e
    completed_label=unknown
    [[ -n "${completed_pid:-}" ]] && completed_label=${pid_names[$completed_pid]:-unknown}
    printf '%s\tchild-exit-%s\n' "${completed_label}" "${rc}" >> "${STATUS_LOG}"
    if [[ "${rc}" -ne 0 ]]; then
      echo "ERROR collector child ${completed_pid:-unknown} failed rc=${rc}" >&2
      return "${rc}"
    fi
    next=()
    for pid in "${active[@]}"; do [[ "${pid}" != "${completed_pid}" ]] && next+=("${pid}"); done
    active=("${next[@]}")
  done
  VALIDATE_ARGS=("${SHARD_STATE_ENTRYPOINT}" validate-batch "${COMMON_STATE_ARGS[@]}" --inspection-handoff "${INSPECTION_HANDOFF}" --expected-inspection-handoff-sha256 "${INSPECTION_HANDOFF_SHA256}")
  for item in "${GATE_ITEMS[@]}"; do VALIDATE_ARGS+=(--item "${item}"); done
  "${PYTHON_EXECUTABLE}" "${VALIDATE_ARGS[@]}" >> "${LOG_DIR}/gate-validation.log" 2>&1
  for i in 0 1 2 3; do printf 'shard-%03d\tcomplete\n' "$((i + 1))" >> "${STATUS_LOG}"; done
}
launch_gate_shards
