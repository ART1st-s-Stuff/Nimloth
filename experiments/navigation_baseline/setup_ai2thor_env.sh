#!/usr/bin/env bash
set -euo pipefail

NIMLOTH_ROOT="/project/peilab/atst/nimloth"
# Reuse flower's verified local Vulkan runtime by default to avoid sudo/system deps.
VULKAN_ROOT="${VULKAN_ROOT:-/project/peilab/atst/flower/.local-vulkan}"
VULKAN_LIB_DIR="${VULKAN_ROOT}/extracted/usr/lib/x86_64-linux-gnu"
VULKAN_RUNTIME_DIR="${VULKAN_ROOT}/runtime/extracted/usr/lib/x86_64-linux-gnu"
VULKAN_TOOLS_DIR="${VULKAN_ROOT}/tools/extracted/usr/bin"
# Reuse flower's verified AI2-THOR cache by default; Nimloth-local cache can be enabled by setting AI2THOR_HOME_ROOT.
AI2THOR_HOME_ROOT="${AI2THOR_HOME_ROOT:-/project/peilab/atst/flower/.ai2thor-home}"

if [[ ! -f "${VULKAN_LIB_DIR}/libvulkan.so.1" ]]; then
  echo "[setup_ai2thor_env] Missing ${VULKAN_LIB_DIR}/libvulkan.so.1" >&2
  echo "[setup_ai2thor_env] Expected flower local Vulkan runtime at ${VULKAN_ROOT}" >&2
  exit 1
fi

mkdir -p "${AI2THOR_HOME_ROOT}"
ln -sf libvulkan.so.1 "${VULKAN_LIB_DIR}/libvulkan.so"

export LD_LIBRARY_PATH="${VULKAN_LIB_DIR}:${VULKAN_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${VULKAN_LIB_DIR}:${VULKAN_RUNTIME_DIR}:${LIBRARY_PATH:-}"
export PATH="${VULKAN_TOOLS_DIR}:${PATH}"
export HOME="${AI2THOR_HOME_ROOT}"
export VK_ICD_FILENAMES="${VULKAN_ROOT}/icd.d/nvidia_icd.json"
export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"

echo "[setup_ai2thor_env] HOME=${HOME}"
echo "[setup_ai2thor_env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[setup_ai2thor_env] VK_ICD_FILENAMES=${VK_ICD_FILENAMES}"
python - <<'PY'
import ctypes.util
print(f"[setup_ai2thor_env] ctypes.find_library('vulkan')={ctypes.util.find_library('vulkan')}")
PY
