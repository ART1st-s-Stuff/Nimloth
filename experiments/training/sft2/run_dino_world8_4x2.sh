#!/usr/bin/env bash
set -euo pipefail

module load slurm >/dev/null 2>&1

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-query-align}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
OUT_ROOT=${OUT_ROOT:-${ROOT}/sft2/${RUN_NAME}}
TRAIN_OUT=${TRAIN_OUT:-${OUT_ROOT}/train}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/latent_wm_value_k8_dinov2.yaml}
MODEL=${MODEL:-${ROOT}/sft1/epoch_005/hf_merged}
RECORDS=${RECORDS:-${ROOT}/converted_strict_k8_b6c811c}
CACHE=${CACHE:-${ROOT}/sft2/1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm/preprocess_cache}
WORLD_SIZE=8
LOCAL_WORLD_SIZE=2

cd "${REPO}"
set -a
# shellcheck disable=SC1091
source /project/peilab/atst/flower/.env
set +a
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm:${REPO}/external/VAGEN"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=nimloth-sft2 WANDB_MODE=online
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export WORLD_SIZE NIMLOTH_DDP_GPU_STRIDE=1
mkdir -p "${TRAIN_OUT}" "${OUT_ROOT}/logs"

node_for_group() {
  local group=$1 var value component
  var="SLURM_JOB_NODELIST_HET_GROUP_${group}"
  value=${!var:-}
  if [[ -z "${value}" ]]; then
    component="${SLURM_JOB_ID%%+*}+${group}"
    value=$(scontrol show job "${component}" -o | tr " " "\n" | awk -F= '$1=="NodeList" {print $2; exit}')
  fi
  [[ -n "${value}" && "${value}" != "(null)" ]] || {
    echo "cannot resolve node for heterogeneous group ${group}" >&2
    return 1
  }
  scontrol show hostnames "${value}" | head -n1
}

nodes=()
for group in 0 1 2 3; do
  nodes+=("$(node_for_group "${group}")")
done
export DINO_HET_NODES
DINO_HET_NODES=$(IFS=,; echo "${nodes[*]}")
export MASTER_ADDR=${MASTER_ADDR:-${nodes[0]}}
export MASTER_PORT=${MASTER_PORT:-$((21001 + ${SLURM_JOB_ID%%+*} % 10000))}

checkpoint_complete() {
  "${PY}" - "${TRAIN_OUT}/final/training_state.pt" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(state.get("epoch", 0)) >= 10 and bool(state.get("epoch_complete", False)), state
PY
}

if [[ -f "${TRAIN_OUT}/sft2_done.flag" ]]; then
  checkpoint_complete || { echo "invalid SFT2 done flag" >&2; exit 3; }
  echo "SFT2 already complete: ${TRAIN_OUT}"
  exit 0
fi

RESUME=0
[[ -f "${TRAIN_OUT}/latest/training_state.pt" ]] && RESUME=1
export REPO ROOT RUN_NAME OUT_ROOT TRAIN_OUT PY CONFIG MODEL RECORDS CACHE RESUME
export WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

printf 'dino_world8 nodes=%s master=%s:%s resume=%s commit=%s\n' \
  "${DINO_HET_NODES}" "${MASTER_ADDR}" "${MASTER_PORT}" "${RESUME}" "$(git rev-parse HEAD)"

srun --het-group=0-3 --kill-on-bad-exit=1 bash -lc '
  IFS=, read -r -a nodes <<< "${DINO_HET_NODES}"
  host=$(hostname -s)
  group=-1
  for i in "${!nodes[@]}"; do
    [[ "${nodes[$i]}" == "${host}" ]] && group=$i && break
  done
  [[ ${group} -ge 0 ]] || { echo "unmapped host ${host}: ${DINO_HET_NODES}" >&2; exit 4; }
  base_rank=$((group * LOCAL_WORLD_SIZE))
  echo "launcher host=${host} visible=${CUDA_VISIBLE_DEVICES:-unset} base_rank=${base_rank} local_world=${LOCAL_WORLD_SIZE} resume=${RESUME}" >&2
  pids=()
  for ((local=0; local<LOCAL_WORLD_SIZE; local++)); do
    rank=$((base_rank + local))
    resume_args=()
    [[ "${RESUME}" == 1 ]] && resume_args=(--resume)
    RANK=${rank} LOCAL_RANK=${local} WORLD_SIZE=${WORLD_SIZE} \
      MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
      "${PY}" experiments/training/sft2/train.py \
        --config "${CONFIG}" --model "${MODEL}" \
        --train-jsonl "${RECORDS}/train_all.jsonl" \
        --val-jsonl "${RECORDS}/val_all.jsonl" \
        --output-dir "${TRAIN_OUT}" \
        --preprocess-cache-dir "${CACHE}" --require-prebuilt-cache \
        --epochs 10 --batch-size 2 --grad-accum 4 \
        --checkpoint-interval-minutes 20 \
        --latent-token-count 8 --latent-query-mode inject --query-tune adapter \
        --llm-tune freeze --vision-tune full \
        --wandb-run-name "${RUN_NAME}" --gradient-checkpointing --seed 42 \
        "${resume_args[@]}" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "${pid}" || rc=$?; done
  exit "${rc}"
'

checkpoint_complete
touch "${TRAIN_OUT}/sft2_done.flag"
