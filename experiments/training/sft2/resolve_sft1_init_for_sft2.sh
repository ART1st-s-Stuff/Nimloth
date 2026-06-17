#!/bin/bash
# Resolve SFT1 LoRA adapter + hf_merged for SFT2 init (success-rate early-stop or override).
set -euo pipefail

: "${SFT1_RUN:?SFT1_RUN required}"
: "${BASE_HF:?BASE_HF required}"
SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB="$(cd "${SCRIPTDIR}/../../navigation_baseline" && pwd)"
RUNS_ROOT="$(dirname "${SFT1_RUN}")"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-alltrain_8gpu_lora_cache}"
PICK_MARGIN="${PICK_MARGIN:-0.0}"
FORCE_EPOCH="${SFT1_EPOCH:-${FORCE_EPOCH:-}}"
ENV_OUT="${ENV_OUT:-${SFT1_RUN}/.sft2_init_env}"

pick_args=(--sft1-run "${SFT1_RUN}" --runs-root "${RUNS_ROOT}" --eval-tag-prefix "${EVAL_TAG_PREFIX}" --margin "${PICK_MARGIN}" --out "${SFT1_RUN}/sft2_init_pick.json")
if [ -n "${FORCE_EPOCH}" ]; then
  pick_args+=(--force-epoch "${FORCE_EPOCH}")
fi

if ! python3 "${SCRIPTDIR}/pick_sft1_ckpt_for_sft2.py" "${pick_args[@]}" >/dev/null; then
  if [ -n "${FORCE_EPOCH}" ]; then
    echo "ERROR pick_sft1_ckpt_for_sft2 failed with FORCE_EPOCH=${FORCE_EPOCH}" >&2
    exit 1
  fi
  echo "WARN: no rollout success data; falling back to SFT1 best (val_loss) — set SFT1_EPOCH to override" >&2
  SFT1_ADAPTER="${SFT1_RUN}/best"
else
  SFT1_ADAPTER="$(python3 -c "import json; print(json.load(open('${SFT1_RUN}/sft2_init_pick.json'))['adapter_dir'])")"
fi

SFT1_BEST="${SFT1_ADAPTER}/hf_merged"
if [ ! -f "${SFT1_BEST}/config.json" ]; then
  if [ ! -f "${SFT1_ADAPTER}/adapter_config.json" ]; then
    echo "ERROR missing SFT1 adapter at ${SFT1_ADAPTER}" >&2
    exit 1
  fi
  echo "=== Merge SFT1 adapter ${SFT1_ADAPTER} -> hf_merged ===" >&2
  python3 "${NB}/merge_sft1_lora_ckpt.py" \
    --base-model "${BASE_HF}" \
    --adapter-dir "${SFT1_ADAPTER}" \
    --out-dir "${SFT1_BEST}"
fi

cat >"${ENV_OUT}" <<EOF
SFT1_ADAPTER=${SFT1_ADAPTER}
SFT1_BEST=${SFT1_BEST}
EOF
cat "${ENV_OUT}"
