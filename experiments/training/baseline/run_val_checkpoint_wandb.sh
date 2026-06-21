#!/usr/bin/env bash
# Run val_only on the latest checkpoint under TRAIN_RUN_DIR and log to wandb.
# Requires external env URLs at ENV_URL_FILE and a local Ray cluster (8 GPUs).
set -euo pipefail

REPO="${REPO:-/project/peilab/atst/nimloth}"
SCRIPTDIR="${REPO}/experiments/training/baseline"
CONFIG_DIR="${REPO}/configs/training/baseline"
BASEDIR="${REPO}/external/VAGEN"

: "${TRAIN_RUN_DIR:?TRAIN_RUN_DIR required}"
: "${VAL_RUN_DIR:?VAL_RUN_DIR required}"
: "${ENV_URL_FILE:?ENV_URL_FILE required}"
: "${CHECKPOINT_STEP:?CHECKPOINT_STEP required}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vagen_nav_baseline}"
PROJECT_NAME="${PROJECT_NAME:-nimloth_navigation}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_NAME}_val_curve}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-24}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-8}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
AGENT_MAX_CONCURRENT_TRAJECTORIES="${AGENT_MAX_CONCURRENT_TRAJECTORIES:-6}"
RAY_PORT="${RAY_PORT:-6379}"
SAVE_CHECKPOINT_DIR="${TRAIN_RUN_DIR}/checkpoints"

mkdir -p "${VAL_RUN_DIR}" "${VAL_RUN_DIR}/validation" "${VAL_RUN_DIR}/logs"

# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/vagen_paper_ppo_cli.inc.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/vagen_rollout_vllm_cli.inc.sh"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/vagen_env_repro_cli.inc.sh"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY not set (source flower/.env)" >&2
  exit 2
fi

WANDB_ID_FILE="${VAL_RUN_DIR}/wandb_run_id.txt"
if [ -f "${WANDB_ID_FILE}" ]; then
  export WANDB_RUN_ID
  WANDB_RUN_ID="$(tr -d '[:space:]' < "${WANDB_ID_FILE}")"
  export WANDB_RESUME=allow
else
  WANDB_RUN_ID="$(python3 - <<'PY'
import uuid
print(uuid.uuid4().hex[:16])
PY
)"
  printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_ID_FILE}"
  export WANDB_RUN_ID
  export WANDB_RESUME=allow
fi

TMP_CONFIG_DIR=$(mktemp -d -p "${VAL_RUN_DIR}" tmpcfg.XXXXXX)
cp "${CONFIG_DIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${CONFIG_DIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

VAL_LOG="${VAL_RUN_DIR}/logs/val_step_${CHECKPOINT_STEP}.log"
{
  echo "=== val_only checkpoint step ${CHECKPOINT_STEP} at $(date) ==="
  echo "TRAIN_RUN_DIR=${TRAIN_RUN_DIR}"
  echo "CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}"
  echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
} | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"

ray stop --force >/dev/null 2>&1 || true
pkill -u "$USER" -f 'vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
sleep 5

HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${HEAD_IP}" ]; then HEAD_IP=$(hostname -I | awk '{print $1}'); fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head \
  --port="${RAY_PORT}" \
  --num-cpus=112 \
  --num-gpus="${TRAIN_GPUS_PER_NODE}" \
  --node-ip-address="${HEAD_IP}" \
  --include-dashboard=false >/dev/null
sleep 15

# Hydra searchpath in vagen_multiturn.yaml (file:../../verl/verl/trainer/config) resolves
# relative to cwd; must match train_resume.slurm / run_preempt_training.sh.
cd "${BASEDIR}"

set +e
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
  --config-path="${BASEDIR}/vagen/configs" \
  --config-name='vagen_multiturn' \
  data.train_files="${TMP_CONFIG_DIR}/train.yaml" \
  data.val_files="${TMP_CONFIG_DIR}/val.yaml" \
  data.custom_cls.path="${REPO}/external/VAGEN/vagen/gym_agent_dataset.py" \
  data.train_batch_size="${VAL_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.seed="${VAGEN_DATA_SEED}" \
  "+data.base_seed=${VAGEN_BASE_SEED}" \
  data.validation_shuffle=False \
  "+trainer.assert_val_env_composition=True" \
  '+trainer.val_env_composition.navigation_base={count:60,eval_set:base}' \
  '+trainer.val_env_composition.navigation_common={count:60,eval_set:common_sense}' \
  "${VAGEN_PAPER_PPO_ARGS[@]}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${VAL_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  "${VAGEN_ROLLOUT_VLLM_ARGS[@]}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.max_concurrent_trajectories="${AGENT_MAX_CONCURRENT_TRAJECTORIES}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node="${TRAIN_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=0 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${WANDB_RUN_NAME}" \
  trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
  trainer.validation_data_dir="${VAL_RUN_DIR}/validation" \
  trainer.rollout_data_dir=null \
  trainer.log_val_generations=0 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${SAVE_CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}" \
  critic.model.use_remove_padding=True \
  critic.model.path="${MODEL_PATH}" \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=32768 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  huggingface_hub.hf_save_freq=null \
  "+ray_kwargs.ray_init.address=auto" \
  2>&1 | tee -a "${VAL_LOG}"
RC=${PIPESTATUS[0]}
set -e

ray stop --force >/dev/null 2>&1 || true
pkill -u "$USER" -f 'vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
rm -rf "${TMP_CONFIG_DIR}"

python3 "${SCRIPTDIR}/upload_val_curve_wandb.py" \
  --log "${VAL_LOG}" \
  --val-run-dir "${VAL_RUN_DIR}" \
  --checkpoint-step "${CHECKPOINT_STEP}" \
  --project "${PROJECT_NAME}" \
  --name "${WANDB_RUN_NAME}" \
  --run-id "${WANDB_RUN_ID}" || true

echo "=== val step ${CHECKPOINT_STEP} finished rc=${RC} at $(date) ===" | tee -a "${VAL_RUN_DIR}/val_wandb_watcher.log"
exit "${RC}"
