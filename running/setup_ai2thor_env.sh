#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VULKAN_ROOT="${ROOT_DIR}/.local-vulkan"
VULKAN_LIB_DIR="${VULKAN_ROOT}/extracted/usr/lib/x86_64-linux-gnu"
VULKAN_TOOLS_DIR="${VULKAN_ROOT}/tools/extracted/usr/bin"
AI2THOR_HOME_ROOT="${ROOT_DIR}/.ai2thor-home"

mkdir -p "${VULKAN_ROOT}" "${AI2THOR_HOME_ROOT}"

if ! ls "${VULKAN_ROOT}"/libvulkan1_*_amd64.deb >/dev/null 2>&1; then
  (cd "${VULKAN_ROOT}" && apt download libvulkan1)
fi

DEB_PATH="$(ls "${VULKAN_ROOT}"/libvulkan1_*_amd64.deb | head -n 1)"

if [[ ! -f "${VULKAN_LIB_DIR}/libvulkan.so.1" ]]; then
  rm -rf "${VULKAN_ROOT}/extracted"
  dpkg-deb -x "${DEB_PATH}" "${VULKAN_ROOT}/extracted"
fi

mkdir -p "${VULKAN_ROOT}/tools"
if ! ls "${VULKAN_ROOT}/tools"/vulkan-tools*_amd64.deb >/dev/null 2>&1; then
  (cd "${VULKAN_ROOT}/tools" && apt download vulkan-tools)
fi

TOOLS_DEB_PATH="$(ls "${VULKAN_ROOT}/tools"/vulkan-tools*_amd64.deb | head -n 1)"
if [[ ! -x "${VULKAN_TOOLS_DIR}/vulkaninfo" ]]; then
  rm -rf "${VULKAN_ROOT}/tools/extracted"
  dpkg-deb -x "${TOOLS_DEB_PATH}" "${VULKAN_ROOT}/tools/extracted"
fi

ln -sf libvulkan.so.1 "${VULKAN_LIB_DIR}/libvulkan.so"

export LD_LIBRARY_PATH="${VULKAN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${VULKAN_LIB_DIR}:${LIBRARY_PATH:-}"
export PATH="${VULKAN_TOOLS_DIR}:${PATH}"
export HOME="${AI2THOR_HOME_ROOT}"

echo "[setup_ai2thor_env] HOME=${HOME}"
echo "[setup_ai2thor_env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[setup_ai2thor_env] PATH=${PATH}"
python - <<'PY'
import ctypes.util
print(f"[setup_ai2thor_env] ctypes.find_library('vulkan')={ctypes.util.find_library('vulkan')}")
PY
