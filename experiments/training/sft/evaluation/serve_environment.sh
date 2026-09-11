#!/usr/bin/env bash
# Reusable batch environment service for VAGEN and early-stage direct evaluation.
set -euo pipefail
: "${VAGEN_DIR:?set VAGEN_DIR to the verified batch-service VAGEN source root}"
: "${ENV_DEVICES:?set ENV_DEVICES to the device list, for example [0]}"
PYTHON=${PYTHON:-python3}
export PYTHONPATH="${VAGEN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -c 'from vagen.server.server import BatchEnvServer'
exec "${PYTHON}" -m vagen.server.server \
  server.host="${ENV_HOST:-127.0.0.1}" server.port="${ENV_PORT:-8000}" \
  use_state_reward=False navigation.devices="${ENV_DEVICES}" \
  navigation.max_workers="${ENV_MAX_WORKERS:-2}" "$@"
