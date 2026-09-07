#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?REPO is required}"
: "${PYTHON:?PYTHON is required}"
: "${ARM_ROOT:?ARM_ROOT is required}"
: "${ARM_OUTPUT:?ARM_OUTPUT is required}"
: "${ENV_PORT:?ENV_PORT is required}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != *,* && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "environment arm requires exactly one visible GPU token" >&2
  exit 2
}

[[ -d /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases ]] || {
  echo "missing verified AI2-THOR release cache" >&2
  exit 2
}
mkdir -p "${ARM_ROOT}/ai2thor/.ai2thor" "${ARM_OUTPUT}"
ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases \
  "${ARM_ROOT}/ai2thor/.ai2thor/releases"
export AI2THOR_HOME_ROOT="${ARM_ROOT}/ai2thor"
OWNED_HOME=${AI2THOR_HOME_ROOT}
OWNED_XDG_CACHE=${ARM_ROOT}/xdg-cache
mkdir -p "${OWNED_HOME}" "${OWNED_XDG_CACHE}"
VULKAN_ROOT=/project/peilab/atst/flower/.local-vulkan
VULKAN_LIB_DIR=${VULKAN_ROOT}/extracted/usr/lib/x86_64-linux-gnu
VULKAN_RUNTIME_DIR=${VULKAN_ROOT}/runtime/extracted/usr/lib/x86_64-linux-gnu
VULKAN_TOOLS_DIR=${VULKAN_ROOT}/tools/extracted/usr/bin
[[ -f "${VULKAN_LIB_DIR}/libvulkan.so.1" ]] || {
  echo "missing verified Vulkan runtime" >&2
  exit 2
}
export LD_LIBRARY_PATH=${VULKAN_LIB_DIR}:${VULKAN_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${VULKAN_LIB_DIR}:${VULKAN_RUNTIME_DIR}:${LIBRARY_PATH:-}
export PATH=${VULKAN_TOOLS_DIR}:${PATH}
export VK_ICD_FILENAMES=${VULKAN_ROOT}/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=${VK_ICD_FILENAMES}

timeout --signal=TERM --kill-after=10s 150s \
  env HOME="${OWNED_HOME}" AI2THOR_HOME_ROOT="${AI2THOR_HOME_ROOT}" \
  XDG_CACHE_HOME="${OWNED_XDG_CACHE}" \
  "${PYTHON}" -m nimloth.environment.navigation.direct_render_probe \
  --gpu-device 0 >"${ARM_OUTPUT}/render_probe.json"

cd "${REPO}/external/VAGEN"
exec env HOME="${OWNED_HOME}" AI2THOR_HOME_ROOT="${AI2THOR_HOME_ROOT}" \
  XDG_CACHE_HOME="${OWNED_XDG_CACHE}" \
  "${PYTHON}" -m vagen.envs.navigation.serve \
  --host=127.0.0.1 --port="${ENV_PORT}" --devices='[0]' \
  --max_envs=24 --max_inflight=24 --thread_pool_size=24 --session_timeout=21600
