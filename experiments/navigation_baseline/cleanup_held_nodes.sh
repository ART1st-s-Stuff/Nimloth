#!/usr/bin/env bash
set -euo pipefail
ray stop --force >/dev/null 2>&1 || true
pkill -u "${USER}" -f 'sglang' 2>/dev/null || true
pkill -u "${USER}" -f 'SGLang' 2>/dev/null || true
pkill -u "${USER}" -f 'vagen.envs.navigation.serve' 2>/dev/null || true
pkill -u "${USER}" -f 'vagen.main_ppo' 2>/dev/null || true
if [ -f /tmp/vagen_3node_451680_env_pids ]; then
  xargs -r kill < /tmp/vagen_3node_451680_env_pids 2>/dev/null || true
fi
