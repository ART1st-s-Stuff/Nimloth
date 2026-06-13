#!/usr/bin/env bash
set -euo pipefail

PORT_BASE=${1:-8100}
MAX_ENVS=${2:-16}

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
RUN_DIR=${SCRIPTDIR}/runs/env_probe_451680/env_servers_$(hostname)_${PORT_BASE}
mkdir -p "$RUN_DIR"

export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export HOME=${ROOT}/.home
export FLASHINFER_WORKSPACE_DIR=${ROOT}/.cache/flashinfer
mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"
export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
source ${ROOT}/.venv/bin/activate
source ${SCRIPTDIR}/setup_ai2thor_env.sh
cd "$BASEDIR"

cleanup() {
  set +e
  if [ -f "$RUN_DIR/pids.txt" ]; then
    xargs -r kill < "$RUN_DIR/pids.txt" 2>/dev/null || true
  fi
}
trap cleanup EXIT
: > "$RUN_DIR/pids.txt"

echo "NODE=$(hostname) IP=$(hostname -I)"
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((PORT_BASE + gpu))
  echo "--- start server gpu=$gpu port=$port ---"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
    --port "$port" \
    --devices='[0]' \
    --max_envs "$MAX_ENVS" \
    --max_inflight "$MAX_ENVS" \
    --thread_pool_size "$MAX_ENVS" \
    --session_timeout 1200.0 \
    > "$RUN_DIR/server_gpu${gpu}.log" 2>&1 &
  echo $! >> "$RUN_DIR/pids.txt"
done

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((PORT_BASE + gpu))
  ok=0
  for i in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "HEALTH_OK gpu=$gpu port=$port tries=$i"
      ok=1
      break
    fi
    sleep 2
  done
  if [ "$ok" -ne 1 ]; then
    echo "HEALTH_FAIL gpu=$gpu port=$port"
    tail -80 "$RUN_DIR/server_gpu${gpu}.log" || true
  fi
done

echo "--- server log errors ---"
for gpu in 0 1 2 3 4 5 6 7; do
  echo "### gpu $gpu"
  grep -E "ERROR|Traceback|Exception|RuntimeError|failed|FAILED" "$RUN_DIR/server_gpu${gpu}.log" | tail -20 || true
done

echo "PROBE_DONE"
