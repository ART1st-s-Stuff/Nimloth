#!/usr/bin/env bash
# Launch one Slurm task inside a heterogeneous FSDP trainer fragment.

set -euo pipefail
: "${RANK_OFFSET:?must set rank offset for this trainer fragment}"
: "${REPO:?must set repo}"
: "${CONFIG:?must set config}"
: "${MODEL:?must set model}"
: "${SFT2_SNAPSHOT:?must set SFT2 snapshot}"
: "${ENV_URL:?must set env URL}"
: "${TRAIN_OUT:?must set trainer output}"
: "${VISION_TUNE:?must set vision tune mode}"
: "${WANDB_RUN_NAME:?must set W&B name}"
: "${MASTER_ADDR:?must set master address}"
: "${MASTER_PORT:?must set master port}"

export RANK=$((RANK_OFFSET + SLURM_PROCID))
export WORLD_SIZE=8
if [[ "${SINGLE_VISIBLE_GPU:-0}" == 1 ]]; then
  export LOCAL_RANK=0
else
  export LOCAL_RANK=${SLURM_LOCALID}
fi

ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  ARGS=(--resume --resume-checkpoint "${RESUME_CHECKPOINT}")
fi

cd "${REPO}"
exec /project/peilab/atst/nimloth/.venv-vagen-main/bin/python3 \
  -m nimloth.training.rl.cli \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --llm-tune lora \
  --vision-tune "${VISION_TUNE}" \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.0 \
  --gradient-checkpointing \
  --wm-checkpoint "${SFT2_SNAPSHOT}/wm_predictor" \
  --state-proj-checkpoint "${SFT2_SNAPSHOT}/state_proj.pt" \
  --value-head-checkpoint "${SFT2_SNAPSHOT}/value_head" \
  --env-url "${ENV_URL}" \
  --attn-implementation flash_attention_2 \
  --max-pixels 100352 \
  --experiment-name "${WANDB_RUN_NAME}" \
  --output-dir "${TRAIN_OUT}" \
  "${ARGS[@]}"
