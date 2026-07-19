#!/usr/bin/env bash
set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:${PATH}

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/dino-profile}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
PROFILE_OUT=${PROFILE_OUT:?PROFILE_OUT is required}
DINO_CACHE_ROOT=${DINO_CACHE_ROOT:?DINO_CACHE_ROOT is required}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
MODEL=${MODEL:-${ROOT}/sft1/epoch_005/hf_merged}
RECORDS=${RECORDS:-${ROOT}/converted_strict_k8_b6c811c}
CACHE=${CACHE:-${ROOT}/sft2/1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm/preprocess_cache}
ONLINE_CONFIG=${ONLINE_CONFIG:-${REPO}/configs/training/sft2/latent_wm_value_k8_dinov2.yaml}
CACHED_CONFIG=${CACHED_CONFIG:-${REPO}/configs/training/sft2/latent_wm_value_k8_dinov2_cached.yaml}
WORLD_SIZE=8
LOCAL_WORLD_SIZE=2

cd "${REPO}"
set -a
# shellcheck disable=SC1091
source /project/peilab/atst/flower/.env
set +a
export PYTHONPATH="${REPO}/src:${REPO}/external/le-wm"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
export WORLD_SIZE NIMLOTH_DDP_GPU_STRIDE=1
mkdir -p "${PROFILE_OUT}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#nodes[@]} -eq 4 ]] || { echo "expected four nodes, got ${nodes[*]}" >&2; exit 2; }
export PROFILE_NODES
PROFILE_NODES=$(IFS=,; echo "${nodes[*]}")
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=${MASTER_PORT:-$((23001 + SLURM_JOB_ID % 10000))}
export REPO ROOT PROFILE_OUT DINO_CACHE_ROOT PY MODEL RECORDS CACHE ONLINE_CONFIG CACHED_CONFIG
export WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

run_variant() {
  local name=$1 config=$2 cache_mode=$3 checkpointing=$4
  export PROFILE_VARIANT=${name} PROFILE_CONFIG=${config} PROFILE_CACHE_MODE=${cache_mode}
  export PROFILE_CHECKPOINTING=${checkpointing}
  rm -rf "${PROFILE_OUT:?}/${name}"
  mkdir -p "${PROFILE_OUT}/${name}"
  echo "profile_variant=${name} nodes=${PROFILE_NODES} commit=$(git rev-parse HEAD)" | tee "${PROFILE_OUT}/${name}/launcher.log"
  set +e
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -lc '
    IFS=, read -r -a nodes <<< "${PROFILE_NODES}"
    host=$(hostname -s); group=-1
    for i in "${!nodes[@]}"; do [[ "${nodes[$i]}" == "${host}" ]] && group=$i && break; done
    [[ ${group} -ge 0 ]] || exit 4
    base_rank=$((group * LOCAL_WORLD_SIZE)); pids=()
    for ((local=0; local<LOCAL_WORLD_SIZE; local++)); do
      rank=$((base_rank + local)); cache_args=(); checkpoint_args=(--gradient-checkpointing)
      [[ "${PROFILE_CACHE_MODE}" == cached ]] && cache_args=(--dino-cache-dir "${DINO_CACHE_ROOT}" --require-dino-cache)
      [[ "${PROFILE_CHECKPOINTING}" == off ]] && checkpoint_args=(--no-gradient-checkpointing)
      RANK=${rank} LOCAL_RANK=${local} WORLD_SIZE=${WORLD_SIZE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
        "${PY}" experiments/training/sft2/train.py \
          --config "${PROFILE_CONFIG}" --model "${MODEL}" \
          --train-jsonl "${RECORDS}/train_all.jsonl" --val-jsonl "${RECORDS}/val_all.jsonl" \
          --output-dir "${PROFILE_OUT}/${PROFILE_VARIANT}" \
          --preprocess-cache-dir "${CACHE}" --require-prebuilt-cache \
          --epochs 10 --batch-size 2 --grad-accum 4 \
          --latent-token-count 8 --latent-query-mode inject --query-tune adapter \
          --llm-tune freeze --vision-tune full --no-wandb \
          --checkpoint-interval-minutes 0 --step-timing --step-timing-interval 10 \
          --profile-optimizer-steps 50 --seed 42 \
          "${cache_args[@]}" "${checkpoint_args[@]}" &
      pids+=("$!")
    done
    rc=0; for pid in "${pids[@]}"; do wait "${pid}" || rc=$?; done; exit "${rc}"
  ' >"${PROFILE_OUT}/${name}/stdout.log" 2>"${PROFILE_OUT}/${name}/stderr.log"
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${PROFILE_OUT}/${name}/exit_code.txt"
  return "${rc}"
}

run_variant online_gc "${ONLINE_CONFIG}" online on
run_variant cached_gc "${CACHED_CONFIG}" cached on
# Diagnostic only: OOM is an acceptable measured result and must not invalidate the first two profiles.
if ! run_variant cached_no_gc "${CACHED_CONFIG}" cached off; then
  echo "cached_no_gc failed as a diagnostic; see its stderr/exit_code" >&2
fi
printf '%s\n' "$(date -Is)" > "${PROFILE_OUT}/done.flag"
