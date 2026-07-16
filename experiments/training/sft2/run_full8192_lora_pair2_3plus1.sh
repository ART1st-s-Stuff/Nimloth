#!/bin/bash
set -euo pipefail
module load slurm >/dev/null 2>&1

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
OUT_ROOT=${OUT_ROOT:-${ROOT}/sft2/18_state8192_fullwm8192_llmlora_vislora_pair2_ws4_ga8_ep5}
TRAIN_OUT=${TRAIN_OUT:-${OUT_ROOT}/train}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/latent_wm_value_k8_state8192_ep5.yaml}
MODEL=${MODEL:-${ROOT}/sft1/epoch_005/hf_merged}
RECORDS=${RECORDS:-${ROOT}/converted_strict_k8_b6c811c}
CACHE=${CACHE:-${ROOT}/sft2/1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm/preprocess_cache}
RUN_NAME=${RUN_NAME:-18_state8192_fullwm8192_llmlora64a128_vislora64a128_pair2_ws4_ga8_ep5}
PAIR_LAYOUT=${PAIR_LAYOUT:-hetero_3plus1}
LAYOUT_HELPER=$REPO/experiments/training/sft2/pair_launcher_layout.sh

cd "$REPO"
set -a
source /project/peilab/atst/flower/.env
set +a
export PYTHONPATH="$REPO/src:$REPO/external/le-wm:$REPO/external/VAGEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=nimloth-sft2 WANDB_MODE=online
export NCCL_DEBUG=INFO NCCL_IB_DISABLE=0
source "$LAYOUT_HELPER"
read -r NCCL_IF GLOO_IF < <(pair_network_values "$PAIR_LAYOUT")
if [[ "$NCCL_IF" == auto ]]; then
  unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
else
  export NCCL_SOCKET_IFNAME=$NCCL_IF GLOO_SOCKET_IFNAME=$GLOO_IF
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_ASYNC_ERROR_HANDLING=1
if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ "$PAIR_LAYOUT" == one_rank_per_node ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  else
    MASTER_ADDR=dgx-27
  fi
fi
export MASTER_ADDR WORLD_SIZE=4 NIMLOTH_DDP_GPU_STRIDE=2
export MASTER_PORT=${MASTER_PORT:-$((20001 + ${HOLD_JOB:-${SLURM_JOB_ID%%+*}} % 10000))}
mkdir -p "$TRAIN_OUT" "$OUT_ROOT/logs"

checkpoint_complete() {
  "$PY" - "$TRAIN_OUT/final/training_state.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert int(state.get("epoch", 0)) >= 5 and bool(state.get("epoch_complete", False)), state
PY
}
if [[ -f "$TRAIN_OUT/sft2_done.flag" ]]; then
  checkpoint_complete || { echo "invalid SFT2 done flag" >&2; exit 3; }
  echo "SFT2 already complete: $TRAIN_OUT"
  exit 0
fi
RESUME=0
[[ -f "$TRAIN_OUT/latest/training_state.pt" ]] && RESUME=1
export REPO ROOT OUT_ROOT TRAIN_OUT PY CONFIG MODEL RECORDS CACHE RUN_NAME RESUME
export PAIR_LAYOUT LAYOUT_HELPER

if [[ "$PAIR_LAYOUT" == one_rank_per_node ]]; then
  SRUN_ARGS=(--nodes=4 --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1)
else
  SRUN_ARGS=(--het-group=0-1 --kill-on-bad-exit=1)
fi
if [[ -n "${HOLD_JOB:-}" ]]; then
  SRUN_ARGS=(--jobid="$HOLD_JOB" --overlap "${SRUN_ARGS[@]}")
fi
srun "${SRUN_ARGS[@]}" bash -lc '
  source "$LAYOUT_HELPER"
  read -r BASE_RANK LOCAL_WORLD < <(
    pair_layout_values "$PAIR_LAYOUT" "$(hostname)" "${SLURM_PROCID:-0}"
  )
  echo "launcher host=$(hostname) visible=${CUDA_VISIBLE_DEVICES:-unset} base=$BASE_RANK local_world=$LOCAL_WORLD resume=$RESUME" >&2
  pids=()
  for ((local=0; local<LOCAL_WORLD; local++)); do
    rank=$((BASE_RANK + local))
    resume_args=()
    [[ "$RESUME" == 1 ]] && resume_args=(--resume)
    RANK=$rank LOCAL_RANK=$local WORLD_SIZE=4 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
      "$PY" experiments/training/sft2/train.py \
        --config "$CONFIG" --model "$MODEL" \
        --train-jsonl "$RECORDS/train_all.jsonl" --val-jsonl "$RECORDS/val_all.jsonl" \
        --output-dir "$TRAIN_OUT" --preprocess-cache-dir "$CACHE" --require-prebuilt-cache \
        --epochs 5 --batch-size 2 --grad-accum 8 --checkpoint-interval-minutes 20 \
        --latent-token-count 8 --latent-query-mode inject --query-tune freeze \
        --llm-tune lora --vision-tune lora --lora-r 64 --lora-alpha 128 \
        --wandb-run-name "$RUN_NAME" --gradient-checkpointing --seed 42 \
        "${resume_args[@]}" &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  exit "$rc"
'
checkpoint_complete
touch "$TRAIN_OUT/sft2_done.flag"
