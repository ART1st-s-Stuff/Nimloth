#!/bin/bash
#SBATCH --job-name=vagen-legacy-wm
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:6
#SBATCH --mem=512G
#SBATCH --requeue

# Co-located VAGEN legacy reproduction:
# - node 0: first ENV_GPUS_PER_NODE GPUs for legacy BatchEnvServer.
# - remaining TRAIN_NODES nodes: TRAIN_GPUS_PER_NODE GPUs per node for Ray/VAGEN training.
# - prompt_format=wm, bi_level_gae, no LLM-as-a-judge state reward.
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/baseline
BASEDIR=${REPO}/external/VAGEN
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_legacy_nav_wm_bilevel}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
LOG_FILE=${RUN_DIR}/${EXPERIMENT_NAME}.log
ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-2}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
TRAIN_NODES=${TRAIN_NODES:-2}
TOTAL_NODES_NEEDED=$((TRAIN_NODES + 1))
SERVICE_PORT=${SERVICE_PORT:-5000}
RAY_PORT=${RAY_PORT:-6379}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv}

mkdir -p "${RUN_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

# shellcheck disable=SC1091
source "${PYTHON_ENV}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/setup_ai2thor_env.sh"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [ "${#NODES[@]}" -lt "${TOTAL_NODES_NEEDED}" ]; then
  echo "ERROR: need ${TOTAL_NODES_NEEDED} nodes (1 env + ${TRAIN_NODES} train), got ${#NODES[@]}" | tee -a "${LOG_FILE}"
  exit 3
fi
ENV_NODE=${NODES[0]}
TRAIN_NODES_LIST=("${NODES[@]:1:${TRAIN_NODES}}")
HEAD_NODE=${TRAIN_NODES_LIST[0]}
ENV_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${ENV_NODE}" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${ENV_IP}" ]; then ENV_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${ENV_NODE}" hostname -I | awk '{print $1}'); fi
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${HEAD_IP}" ]; then HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}'); fi
IP_HEAD=${HEAD_IP}:${RAY_PORT}
SERVICE_BASE_URL=http://${ENV_IP}:${SERVICE_PORT}

cleanup() {
  set +e
  echo "=== legacy reproduction cleanup $(date) ===" | tee -a "${LOG_FILE}"
  for node in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc "
      REPO='${REPO}'
      source '${PYTHON_ENV}/bin/activate'
      source '${SCRIPTDIR}/common_env.sh'
      ray stop --force >/dev/null 2>&1 || true
      pkill -u \$USER -f 'vagen.server.server|vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
    " >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

{
  echo "=== VAGEN legacy preempt reproduction $(date) ==="
  echo "nodes=${NODES[*]} env_node=${ENV_NODE} env_ip=${ENV_IP} train_nodes=${TRAIN_NODES_LIST[*]} head=${HEAD_NODE} head_ip=${HEAD_IP}"
  echo "run_dir=${RUN_DIR}"
  echo "service=${SERVICE_BASE_URL}"
  echo "env_gpus_per_node=${ENV_GPUS_PER_NODE} train_gpus_per_node=${TRAIN_GPUS_PER_NODE} train_nodes=${TRAIN_NODES}"
} | tee -a "${LOG_FILE}"

for node in "${NODES[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc "
    REPO='${REPO}'
    source '${PYTHON_ENV}/bin/activate'
    source '${SCRIPTDIR}/common_env.sh'
    ray stop --force >/dev/null 2>&1 || true
    pkill -u \$USER -f 'vagen.server.server|vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
  " || true
done
sleep 10

# Legacy BatchEnvServer on dedicated env node, using first ENV_GPUS_PER_NODE local GPU ordinals.
ENV_DEVICES=$(python3 - <<PY
print('[' + ','.join(str(i) for i in range(${ENV_GPUS_PER_NODE})) + ']')
PY
)
ENV_CUDA=$(python3 - <<PY
print(','.join(str(i) for i in range(${ENV_GPUS_PER_NODE})))
PY
)
echo "=== Starting legacy env service on ${ENV_NODE}, CUDA=${ENV_CUDA}, devices=${ENV_DEVICES} ===" | tee -a "${LOG_FILE}"
srun --overlap --nodes=1 --ntasks=1 -w "${ENV_NODE}" bash -lc "
  set -euo pipefail
  REPO='${REPO}'
  source '${PYTHON_ENV}/bin/activate'
  source '${SCRIPTDIR}/common_env.sh'
  source '${SCRIPTDIR}/setup_ai2thor_env.sh'
  cd '${BASEDIR}'
  CUDA_VISIBLE_DEVICES='${ENV_CUDA}' PYTHONUNBUFFERED=1 python -m vagen.server.server \
    server.host=0.0.0.0 \
    server.port=${SERVICE_PORT} \
    use_state_reward=False \
    navigation.devices='${ENV_DEVICES}' \
    navigation.max_workers=48
" > "${RUN_DIR}/legacy_env_server_${SLURM_JOB_ID}_${ENV_NODE}.log" 2>&1 &
ENV_PID=$!

for i in $(seq 1 180); do
  if curl -fsS "${SERVICE_BASE_URL}/health" >/dev/null 2>&1; then
    echo "legacy env service health OK after ${i} checks" | tee -a "${LOG_FILE}"
    break
  fi
  kill -0 "${ENV_PID}" >/dev/null 2>&1 || {
    echo "ERROR legacy env service exited" | tee -a "${LOG_FILE}"
    tail -200 "${RUN_DIR}/legacy_env_server_${SLURM_JOB_ID}_${ENV_NODE}.log" | tee -a "${LOG_FILE}" || true
    exit 4
  }
  sleep 5
  if [ "${i}" -eq 180 ]; then
    echo "ERROR timed out waiting for legacy env service" | tee -a "${LOG_FILE}"
    tail -200 "${RUN_DIR}/legacy_env_server_${SLURM_JOB_ID}_${ENV_NODE}.log" | tee -a "${LOG_FILE}" || true
    exit 4
  fi
done

TRAIN_CUDA=$(python3 - <<PY
print(','.join(str(i) for i in range(${TRAIN_GPUS_PER_NODE})))
PY
)
echo "=== Starting Ray head ${HEAD_NODE} CUDA=${TRAIN_CUDA} ===" | tee -a "${LOG_FILE}"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  set -euo pipefail
  REPO='${REPO}'
  source '${PYTHON_ENV}/bin/activate'
  source '${SCRIPTDIR}/common_env.sh'
  CUDA_VISIBLE_DEVICES='${TRAIN_CUDA}' ray start --head \
    --port=${RAY_PORT} --num-cpus=96 --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' --include-dashboard=false --disable-usage-stats --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 20

for node in "${TRAIN_NODES_LIST[@]:1}"; do
  echo "=== Starting Ray worker ${node} CUDA=${TRAIN_CUDA} ===" | tee -a "${LOG_FILE}"
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" bash -lc "
    set -euo pipefail
    REPO='${REPO}'
    source '${PYTHON_ENV}/bin/activate'
    source '${SCRIPTDIR}/common_env.sh'
    CUDA_VISIBLE_DEVICES='${TRAIN_CUDA}' ray start --address='${IP_HEAD}' \
      --num-cpus=96 --num-gpus=${TRAIN_GPUS_PER_NODE} --block
  " > "${RUN_DIR}/ray_worker_${SLURM_JOB_ID}_${node}.log" 2>&1 &
done

NEED_GPUS=$((TRAIN_GPUS_PER_NODE * TRAIN_NODES))
ray_ready=0
for i in $(seq 1 60); do
  ray_gpus=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
    REPO='${REPO}'
    source '${PYTHON_ENV}/bin/activate'
    source '${SCRIPTDIR}/common_env.sh'
    python3 - <<'PY'
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR')
    print(int(ray.cluster_resources().get('GPU', 0)))
finally:
    ray.shutdown()
PY
  " 2>/dev/null | tail -1 || true)
  [ -n "${ray_gpus}" ] && echo "ray gpu check ${i}: ${ray_gpus}/${NEED_GPUS}" | tee -a "${LOG_FILE}"
  if [ "${ray_gpus:-0}" -ge "${NEED_GPUS}" ]; then ray_ready=1; break; fi
  sleep 5
done
if [ "${ray_ready}" -ne 1 ]; then
  echo "ERROR Ray did not reach ${NEED_GPUS} GPUs" | tee -a "${LOG_FILE}"
  exit 4
fi

export REPO RUN_DATE EXPERIMENT_NAME RUN_DIR LOG_FILE SERVICE_BASE_URL TRAIN_GPUS_PER_NODE TRAIN_NODES
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash "${SCRIPTDIR}/run_legacy_reproduction.sh"
