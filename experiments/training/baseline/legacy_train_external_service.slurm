#!/bin/bash
#SBATCH --job-name=vagen-legacy-train
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:7
#SBATCH --mem=512G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth}
SCRIPTDIR=${REPO}/experiments/training/baseline
BASEDIR=${REPO}/external/VAGEN
PYTHON_ENV=${PYTHON_ENV:-/project/peilab/atst/nimloth/.venv}
CONVERT_PY=${CONVERT_PY:-${SCRIPTDIR}/convert_vagen_actor_only_to_world_size.py}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_legacy_train_external_service}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
LOG_FILE=${LOG_FILE:-${RUN_DIR}/${EXPERIMENT_NAME}.log}
CONTROL_DIR=${CONTROL_DIR:-${RUN_DIR}/legacy_env_service}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
SERVICE_BASE_URL=${SERVICE_BASE_URL:-}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:?set ACTOR_MODEL_PATH}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:?set CRITIC_MODEL_PATH}
SOURCE_CHECKPOINT_STEP=${SOURCE_CHECKPOINT_STEP:-300}
TOTAL_STEPS=${TOTAL_STEPS:-330}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-1}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-7}
TRAIN_NODES=${TRAIN_NODES:-1}
VAGEN_DATA_SEED=${VAGEN_DATA_SEED:-42}
ENABLE_WANDB=${ENABLE_WANDB:-1}
RAY_PORT=${RAY_PORT:-6379}
CONVERT_DIR=${CONVERT_DIR:-${RUN_DIR}/ws${TRAIN_GPUS_PER_NODE}_convert}

mkdir -p "${RUN_DIR}" "${SAVE_CHECKPOINT_DIR}" "${CONVERT_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

# shellcheck disable=SC1091
source "${PYTHON_ENV}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
cd "${BASEDIR}"

cleanup() {
  set +e
  echo "=== legacy train cleanup $(date) ===" | tee -a "${LOG_FILE}"
  ray stop --force >/dev/null 2>&1 || true
  pkill -u "$USER" -f 'vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  echo "=== legacy train-only normal job $(date) ==="
  echo "job=${SLURM_JOB_ID} host=$(hostname) run_dir=${RUN_DIR}"
  echo "actor_model=${ACTOR_MODEL_PATH}"
  echo "critic_model=${CRITIC_MODEL_PATH}"
  echo "target_steps=${SOURCE_CHECKPOINT_STEP}->${TOTAL_STEPS}"
  echo "world_size=${TRAIN_GPUS_PER_NODE} (non-strict resume: model weights only, fresh ws${TRAIN_GPUS_PER_NODE} optimizer/lr scheduler state)"
} | tee -a "${LOG_FILE}"

if [ -z "${SERVICE_BASE_URL}" ]; then
  for i in $(seq 1 120); do
    if [ -f "${CONTROL_DIR}/failed" ]; then
      echo "ERROR env control dir marked failed" | tee -a "${LOG_FILE}"
      exit 3
    fi
    if [ -f "${CONTROL_DIR}/base_url.txt" ]; then
      SERVICE_BASE_URL=$(tr -d '[:space:]' < "${CONTROL_DIR}/base_url.txt")
      break
    fi
    sleep 5
  done
fi

if [ -z "${SERVICE_BASE_URL}" ]; then
  echo "ERROR missing SERVICE_BASE_URL and ${CONTROL_DIR}/base_url.txt" | tee -a "${LOG_FILE}"
  exit 3
fi

for i in $(seq 1 120); do
  if curl -fsS "${SERVICE_BASE_URL}/health" >/dev/null 2>&1; then
    echo "legacy env health OK: ${SERVICE_BASE_URL}" | tee -a "${LOG_FILE}"
    break
  fi
  if [ -f "${CONTROL_DIR}/failed" ]; then
    echo "ERROR env control dir marked failed while waiting for health" | tee -a "${LOG_FILE}"
    exit 3
  fi
  sleep 5
  if [ "$i" -eq 120 ]; then
    echo "ERROR timed out waiting for legacy env health at ${SERVICE_BASE_URL}" | tee -a "${LOG_FILE}"
    exit 3
  fi
done

HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${HEAD_IP}" ]; then HEAD_IP=$(hostname -I | awk '{print $1}'); fi
TRAIN_CUDA=$(python - <<PY
print(','.join(str(i) for i in range(${TRAIN_GPUS_PER_NODE})))
PY
)

start_ray_head() {
  ray stop --force >/dev/null 2>&1 || true
  pkill -u "$USER" -f 'vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
  sleep 10
  echo "=== Starting Ray head CUDA=${TRAIN_CUDA} ip=${HEAD_IP} port=${RAY_PORT} ===" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" ray start --head \
    --port="${RAY_PORT}" \
    --num-cpus=96 \
    --num-gpus="${TRAIN_GPUS_PER_NODE}" \
    --node-ip-address="${HEAD_IP}" \
    --include-dashboard=false \
    --disable-usage-stats \
    --block > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_$(hostname).log" 2>&1 &
  RAY_PID=$!
  sleep 20

  ray_ready=0
  for i in $(seq 1 60); do
    ray_gpus=$(python3 - <<'PY' 2>/dev/null | tail -1 || true
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR')
    print(int(ray.cluster_resources().get('GPU', 0)))
finally:
    ray.shutdown()
PY
    )
    if [ -n "${ray_gpus}" ]; then
      echo "ray gpu check ${i}: ${ray_gpus}/${TRAIN_GPUS_PER_NODE}" | tee -a "${LOG_FILE}"
      if [ "${ray_gpus:-0}" -ge "${TRAIN_GPUS_PER_NODE}" ]; then
        ray_ready=1
        break
      fi
    fi
    sleep 5
  done
  if [ "${ray_ready}" -ne 1 ]; then
    echo "ERROR Ray did not reach ${TRAIN_GPUS_PER_NODE} GPUs" | tee -a "${LOG_FILE}"
    exit 4
  fi
}

ensure_ws_checkpoint() {
  local actor_rank_file="${SAVE_CHECKPOINT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TRAIN_GPUS_PER_NODE}_rank_$((TRAIN_GPUS_PER_NODE - 1)).pt"
  local critic_rank_file="${SAVE_CHECKPOINT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/critic/model_world_size_${TRAIN_GPUS_PER_NODE}_rank_$((TRAIN_GPUS_PER_NODE - 1)).pt"
  if [ -f "${actor_rank_file}" ] && [ -f "${critic_rank_file}" ]; then
    echo "=== ws${TRAIN_GPUS_PER_NODE} checkpoint already present; skip conversion ===" | tee -a "${LOG_FILE}"
    echo "${SOURCE_CHECKPOINT_STEP}" > "${SAVE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
    return
  fi

  cat > "${CONVERT_DIR}/dummy.yaml" <<YAML
envs:
  - name: RemoteEnv
    n_envs: 12
    data_source: navigation_base_train
    seed: [1, 12, 1]
    max_turns: 1
    response_length_per_turn: 16
    config:
      base_urls: "http://127.0.0.1:9"
      timeout: 1
      retries: 0
      eval_set: base_train
      prompt_format: eval_mode
      max_actions_per_step: 1
YAML

  echo "=== Converting actor/critic HF exports to ws${TRAIN_GPUS_PER_NODE} checkpoint at global_step_${SOURCE_CHECKPOINT_STEP} ===" | tee -a "${LOG_FILE}"
  echo "NOTE: this conversion preserves model weights only; optimizer/lr scheduler are freshly initialized for ws${TRAIN_GPUS_PER_NODE}." | tee -a "${LOG_FILE}"
  # shellcheck disable=SC1091
  source "${SCRIPTDIR}/vagen_rollout_vllm_cli.inc.sh"
  PYTHONUNBUFFERED=1 python3 "${CONVERT_PY}" \
    --config-path='${BASEDIR}/vagen/configs' \
    --config-name='vagen_multiturn' \
    hydra.searchpath=[file://${REPO}/external/VAGEN/verl/verl/trainer/config] \
    data.train_files='${CONVERT_DIR}/dummy.yaml' \
    data.val_files='${CONVERT_DIR}/dummy.yaml' \
    data.train_batch_size=${TRAIN_GPUS_PER_NODE} \
    data.val_batch_size=${TRAIN_GPUS_PER_NODE} \
    data.max_prompt_length=3000 \
    data.max_response_length=20000 \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path='${ACTOR_MODEL_PATH}' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_GPUS_PER_NODE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    +actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    "${VAGEN_ROLLOUT_VLLM_ARGS[@]}" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='${BASEDIR}/vagen/configs/agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    +actor_rollout_ref.ref.fsdp_config.sync_module_states=False \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
    trainer.nnodes=${TRAIN_NODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=0 \
    trainer.max_actor_ckpt_to_keep=50 \
    trainer.max_critic_ckpt_to_keep=50 \
    trainer.project_name=nimloth_navigation \
    trainer.experiment_name=${EXPERIMENT_NAME}_ws${TRAIN_GPUS_PER_NODE}_convert \
    trainer.default_local_dir='${SAVE_CHECKPOINT_DIR}' \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    +trainer.actor_only_convert=True \
    +trainer.convert_global_step=${SOURCE_CHECKPOINT_STEP} \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path='${CRITIC_MODEL_PATH}' \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.use_torch_compile=False \
    +critic.model.fsdp_config.sync_module_states=False \
    reward_model.enable=False \
    huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto \
    2>&1 | tee -a "${LOG_FILE}"

  if [ ! -f "${actor_rank_file}" ]; then
    echo "ERROR conversion finished without actor shard ${actor_rank_file}" | tee -a "${LOG_FILE}"
    exit 5
  fi
  if [ ! -f "${critic_rank_file}" ]; then
    echo "ERROR conversion finished without critic shard ${critic_rank_file}" | tee -a "${LOG_FILE}"
    exit 5
  fi
  echo "${SOURCE_CHECKPOINT_STEP}" > "${SAVE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
  echo "=== ws${TRAIN_GPUS_PER_NODE} conversion finished ===" | tee -a "${LOG_FILE}"
}

start_ray_head
ensure_ws_checkpoint
ray stop --force >/dev/null 2>&1 || true
pkill -u "$USER" -f 'vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
sleep 10
start_ray_head

export REPO RUN_DATE EXPERIMENT_NAME RUN_DIR LOG_FILE SAVE_CHECKPOINT_DIR SERVICE_BASE_URL TRAIN_GPUS_PER_NODE TRAIN_NODES
export MODEL_PATH="${ACTOR_MODEL_PATH}"
export CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH}"
export RESUME_MODE=auto
export TOTAL_STEPS TOTAL_EPOCHS TEST_FREQ SAVE_FREQ VAGEN_DATA_SEED ENABLE_WANDB

echo "=== Launching legacy training from ws${TRAIN_GPUS_PER_NODE} converted checkpoint ===" | tee -a "${LOG_FILE}"
bash "${SCRIPTDIR}/run_legacy_reproduction.sh"
