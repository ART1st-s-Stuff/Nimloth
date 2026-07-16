#!/usr/bin/env bash
#SBATCH --job-name=rl-dyn-env
#SBATCH --time=00:45:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail
: "${RUN_ROOT:?must set RUN_ROOT}"
: "${ENV_REPO:?must set ENV_REPO to a worktree with verified *_train assets}"

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
ENV_PORT=${ENV_PORT:-5000}
mkdir -p "${RUN_ROOT}"
rm -f "${RUN_ROOT}/env_url.txt" "${RUN_ROOT}/env_ready.flag"

source "${ENV_REPO}/experiments/training/baseline/setup_ai2thor_env.sh"
export PYTHONPATH="${ENV_REPO}/external/VAGEN:${PYTHONPATH:-}"

HOST_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [[ -z "${HOST_IP}" ]]; then
  HOST_IP=$(hostname -I | awk '{print $1}')
fi
ENV_URL="http://${HOST_IP}:${ENV_PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

cd "${ENV_REPO}/external/VAGEN"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${PYTHON}" -m vagen.server.server \
  server.host=0.0.0.0 \
  server.port="${ENV_PORT}" \
  use_state_reward=False \
  'navigation.devices=[0]' \
  navigation.max_workers=2 \
  >"${RUN_ROOT}/env_server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  if curl --fail --silent "${ENV_URL}/health" >/dev/null; then
    printf '%s\n' "${ENV_URL}" >"${RUN_ROOT}/env_url.txt.tmp"
    mv "${RUN_ROOT}/env_url.txt.tmp" "${RUN_ROOT}/env_url.txt"
    touch "${RUN_ROOT}/env_ready.flag"
    echo "env_ready url=${ENV_URL} pid=${SERVER_PID} commit=$(git -C "${ENV_REPO}/external/VAGEN" rev-parse HEAD)"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "environment server exited before health check" >&2
    exit 1
  fi
  sleep 2
done
[[ -f "${RUN_ROOT}/env_ready.flag" ]] || { echo "environment health timeout" >&2; exit 1; }

for _ in $(seq 1 480); do
  [[ -f "${RUN_ROOT}/trainer_done.flag" ]] && exit 0
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "environment server exited while trainer was running" >&2
    exit 1
  fi
  sleep 5
done

echo "trainer completion timeout" >&2
exit 1
