#!/usr/bin/env bash
# Run VAGEN legacy navigation reproduction on an existing Ray/env allocation.
# Expected usage: called from legacy_preempt_reproduction.slurm after Ray and
# the legacy BatchEnvServer are ready.
set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/baseline
CONFIG_DIR=${REPO}/configs/training/baseline
BASEDIR=${REPO}/external/VAGEN

RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_legacy_nav_wm_bilevel}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
DATA_DIR=${DATA_DIR:-${RUN_DIR}/data}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
LOG_FILE=${LOG_FILE:-${RUN_DIR}/${EXPERIMENT_NAME}.log}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-${MODEL_PATH}}
TRAIN_CONFIG=${TRAIN_CONFIG:-${CONFIG_DIR}/legacy_train.yaml}
VAL_CONFIG=${VAL_CONFIG:-${CONFIG_DIR}/legacy_val.yaml}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_DIR}/val.parquet}
VAGEN_DATA_SEED=${VAGEN_DATA_SEED:-42}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-24}
TOTAL_STEPS=${TOTAL_STEPS:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-1}
ALGORITHM_GAMMA=${ALGORITHM_GAMMA:-1.0}
HIGH_LEVEL_GAMMA=${HIGH_LEVEL_GAMMA:-0.95}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
TRAIN_NODES=${TRAIN_NODES:-2}
SERVICE_BASE_URL=${SERVICE_BASE_URL:-http://127.0.0.1:5000}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
RESTORE_DATALOADER_STATE=${RESTORE_DATALOADER_STATE:-true}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
VAGEN_MAX_PROMPT_LENGTH=${VAGEN_MAX_PROMPT_LENGTH:-3000}
VAGEN_MAX_RESPONSE_LENGTH=${VAGEN_MAX_RESPONSE_LENGTH:-20000}
ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-32000}
ROLLOUT_MAX_TURNS=${ROLLOUT_MAX_TURNS:-20}
VLLM_MAX_RESPONSE_PER_TURN=${VLLM_MAX_RESPONSE_PER_TURN:-1024}
VAGEN_NON_RESPONSE_TOKEN_RESERVE=${VAGEN_NON_RESPONSE_TOKEN_RESERVE:-11000}
VAGEN_TRAJECTORY_TRUNCATION=${VAGEN_TRAJECTORY_TRUNCATION:-error}
VLLM_TENSOR_MODEL_PARALLEL_SIZE=${VLLM_TENSOR_MODEL_PARALLEL_SIZE:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.6}
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv}
ENABLE_WANDB=${ENABLE_WANDB:-0}
if [ "${ENABLE_WANDB}" = "1" ]; then
  LOGGER_HYDRA="['console','wandb']"
else
  LOGGER_HYDRA="['console']"
fi

MAX_GENERATED_TOKEN_BUDGET=$((ROLLOUT_MAX_TURNS * VLLM_MAX_RESPONSE_PER_TURN))
if [ $((MAX_GENERATED_TOKEN_BUDGET + VAGEN_NON_RESPONSE_TOKEN_RESERVE)) -gt "${ROLLOUT_MAX_TRAJECTORY_LENGTH}" ]; then
  echo "ERROR trajectory token budget exceeds max: generated=${MAX_GENERATED_TOKEN_BUDGET} reserve=${VAGEN_NON_RESPONSE_TOKEN_RESERVE} max=${ROLLOUT_MAX_TRAJECTORY_LENGTH}" >&2
  exit 2
fi

RESUME_ARGS=("trainer.resume_mode=${RESUME_MODE}")
if [ -n "${RESUME_FROM_PATH}" ]; then
  RESUME_ARGS+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${SAVE_CHECKPOINT_DIR}"

# shellcheck disable=SC1091
source "${PYTHON_ENV}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
cd "${BASEDIR}"

TRAIN_CONFIG_SUMMARY=$(python - <<PY
import yaml
with open(${TRAIN_CONFIG@Q}, 'r') as f:
    cfg = yaml.safe_load(f)
prompt_formats = sorted({value.get('env_config', {}).get('prompt_format', '<missing>') for value in cfg.values()})
use_state_reward = sorted({str(value.get('env_config', {}).get('use_state_reward', '<missing>')).lower() for value in cfg.values()})
print(f"prompt_format={','.join(prompt_formats)} use_state_reward={','.join(use_state_reward)}")
PY
)

{
  echo "=== VAGEN legacy reproduction launch $(date) ==="
  echo "repo=${REPO}"
  echo "vagen_commit=$(git rev-parse HEAD)"
  echo "root_commit=$(cd "${REPO}" && git rev-parse HEAD)"
  echo "run_dir=${RUN_DIR}"
  echo "actor_model=${MODEL_PATH}"
  echo "critic_model=${CRITIC_MODEL_PATH}"
  echo "train_config=${TRAIN_CONFIG}"
  echo "val_config=${VAL_CONFIG}"
  echo "service_base_url=${SERVICE_BASE_URL}"
  echo "adv_estimator=bi_level_gae ${TRAIN_CONFIG_SUMMARY} reward_model.enable=false"
  echo "gamma=${ALGORITHM_GAMMA} high_level_gamma=${HIGH_LEVEL_GAMMA} total_steps=${TOTAL_STEPS} total_epochs=${TOTAL_EPOCHS} save_freq=${SAVE_FREQ} test_freq=${TEST_FREQ}"
  echo "resume_mode=${RESUME_MODE} resume_from_path=${RESUME_FROM_PATH:-<auto-from-default_local_dir>} restore_dataloader_state=${RESTORE_DATALOADER_STATE}"
  echo "trajectory max_turns=${ROLLOUT_MAX_TURNS} max_response_per_turn=${VLLM_MAX_RESPONSE_PER_TURN} max_trajectory_length=${ROLLOUT_MAX_TRAJECTORY_LENGTH} truncation=${VAGEN_TRAJECTORY_TRUNCATION} max_generated_budget=${MAX_GENERATED_TOKEN_BUDGET} non_response_reserve=${VAGEN_NON_RESPONSE_TOKEN_RESERVE}"
  echo "resources train_nodes=${TRAIN_NODES} train_gpus_per_node=${TRAIN_GPUS_PER_NODE}"
} | tee -a "${LOG_FILE}"

PYTHONUNBUFFERED=1 python -m vagen.env.create_dataset \
  --yaml_path "${TRAIN_CONFIG}" \
  --force_gen \
  --seed "${VAGEN_DATA_SEED}" \
  --train_path "${TRAIN_PARQUET}" \
  --test_path "${DATA_DIR}/train_unused_test.parquet" \
  2>&1 | tee -a "${LOG_FILE}"

PYTHONUNBUFFERED=1 python -m vagen.env.create_dataset \
  --yaml_path "${VAL_CONFIG}" \
  --force_gen \
  --seed "${VAGEN_DATA_SEED}" \
  --train_path "${DATA_DIR}/val_unused_train.parquet" \
  --test_path "${VAL_PARQUET}" \
  2>&1 | tee -a "${LOG_FILE}"

set +e
PYTHONUNBUFFERED=1 python -m vagen.trainer.main_ppo \
  algorithm.adv_estimator=bi_level_gae \
  algorithm.high_level_gamma="${HIGH_LEVEL_GAMMA}" \
  algorithm.gamma="${ALGORITHM_GAMMA}" \
  algorithm.lam=1.0 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${VAGEN_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${VAGEN_MAX_RESPONSE_LENGTH}" \
  data.max_trajectory_length="${ROLLOUT_MAX_TRAJECTORY_LENGTH}" \
  data.image_key=images \
  data.truncation="${VAGEN_TRAJECTORY_TRUNCATION}" \
  +data.seed="${VAGEN_DATA_SEED}" \
  data.shuffle=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${VLLM_TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.limit_mm_per_prompt="${ROLLOUT_MAX_TURNS}" \
  actor_rollout_ref.rollout.max_response_per_turn="${VLLM_MAX_RESPONSE_PER_TURN}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=1e-5 \
  critic.model.path="${CRITIC_MODEL_PATH}" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  reward_model.enable=False \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER_HYDRA}" \
  trainer.project_name=nimloth_navigation \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${TRAIN_GPUS_PER_NODE}" \
  trainer.nnodes="${TRAIN_NODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
  trainer.restore_dataloader_state="${RESTORE_DATALOADER_STATE}" \
  "${RESUME_ARGS[@]}" \
  trainer.val_before_train=True \
  trainer.val_generations_to_log_to_wandb=8 \
  trainer.validation_data_dir="${RUN_DIR}/validation" \
  rollout_manager.max_turns="${ROLLOUT_MAX_TURNS}" \
  rollout_manager.window_size=5 \
  rollout_manager.n_trajectory=1 \
  rollout_manager.use_service=True \
  rollout_manager.base_url="${SERVICE_BASE_URL}" \
  rollout_manager.timeout=1200 \
  rollout_manager.max_workers="${AGENT_NUM_WORKERS}" \
  rollout_manager.use_loss_mask=True \
  rollout_manager.use_gae_mask=True \
  rollout_manager.use_multi_turn_reward=True \
  +rollout_manager.mini_batch_size="${VAL_BATCH_SIZE}" \
  2>&1 | tee -a "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e

echo "=== VAGEN legacy reproduction finished rc=${RC} $(date) ===" | tee -a "${LOG_FILE}"
exit "${RC}"
