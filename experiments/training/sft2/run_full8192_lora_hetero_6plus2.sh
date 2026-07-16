#!/bin/bash
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/k8-preprojection-recon}
ROOT=${ROOT:-/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97}
OUT_ROOT=${OUT_ROOT:-${ROOT}/sft2/17_state8192_fullwm8192_ws8_ga4_ep5_dgx27x6_dgx54x2}
TRAIN_OUT=${TRAIN_OUT:-${OUT_ROOT}/train}
PY=${PY:-/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3}
CONFIG=${CONFIG:-${REPO}/configs/training/sft2/latent_wm_value_k8_state8192_ep5.yaml}
MODEL=${MODEL:-${ROOT}/sft1/epoch_005/hf_merged}
RECORDS=${RECORDS:-${ROOT}/converted_strict_k8_b6c811c}
CACHE=${CACHE:-${ROOT}/sft2/1_k8inject_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_px100352_img12_bestwm/preprocess_cache}
RUN_NAME=${RUN_NAME:-17_state8192_fullwm8192_llmlora64a128_vislora64a128_ws8_ga4_ep5}

cd "$REPO"
set -a
source /project/peilab/atst/flower/.env
set +a
export PYTHONPATH="$REPO/src:$REPO/external/le-wm:$REPO/external/VAGEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=nimloth-sft2 WANDB_MODE=online
export NCCL_DEBUG=INFO NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ibp41s0f0 GLOO_SOCKET_IFNAME=ibp41s0f0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=dgx-27 WORLD_SIZE=8 NIMLOTH_DDP_GPU_STRIDE=1
export MASTER_PORT=${MASTER_PORT:-$((20000 + ${HOLD_JOB:-${SLURM_JOB_ID%%+*}} % 10000))}
mkdir -p "$TRAIN_OUT" "$OUT_ROOT/logs"

if [[ -f "$TRAIN_OUT/sft2_done.flag" ]]; then
  echo "SFT2 already complete: $TRAIN_OUT"
  exit 0
fi
RESUME=0
[[ -f "$TRAIN_OUT/latest/training_state.pt" ]] && RESUME=1
export REPO ROOT OUT_ROOT TRAIN_OUT PY CONFIG MODEL RECORDS CACHE RUN_NAME RESUME

SRUN_ARGS=(--het-group=0-1 --kill-on-bad-exit=1)
if [[ -n "${HOLD_JOB:-}" ]]; then
  SRUN_ARGS=(--jobid="$HOLD_JOB" --overlap "${SRUN_ARGS[@]}")
fi
srun "${SRUN_ARGS[@]}" bash -lc '
  case "$(hostname)" in
    dgx-27*) BASE_RANK=0; LOCAL_WORLD=6 ;;
    dgx-54*) BASE_RANK=6; LOCAL_WORLD=2 ;;
    *) echo "unexpected host $(hostname)" >&2; exit 2 ;;
  esac
  echo "launcher host=$(hostname) visible=${CUDA_VISIBLE_DEVICES:-unset} base=$BASE_RANK local_world=$LOCAL_WORLD resume=$RESUME" >&2
  pids=()
  for ((local=0; local<LOCAL_WORLD; local++)); do
    rank=$((BASE_RANK + local))
    resume_args=()
    [[ "$RESUME" == 1 ]] && resume_args=(--resume)
    RANK=$rank LOCAL_RANK=$local WORLD_SIZE=8 MASTER_ADDR=dgx-27 MASTER_PORT="$MASTER_PORT" \
      "$PY" experiments/training/sft2/train.py \
        --config "$CONFIG" --model "$MODEL" \
        --train-jsonl "$RECORDS/train_all.jsonl" --val-jsonl "$RECORDS/val_all.jsonl" \
        --output-dir "$TRAIN_OUT" --preprocess-cache-dir "$CACHE" --require-prebuilt-cache \
        --epochs 5 --batch-size 2 --grad-accum 4 --checkpoint-interval-minutes 20 \
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
touch "$TRAIN_OUT/sft2_done.flag"
