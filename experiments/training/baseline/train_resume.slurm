#!/bin/bash
#SBATCH --job-name=vagen-train-resume
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --requeue

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${REPO}/experiments/training/baseline
CONFIG_DIR=${REPO}/configs/training/baseline
BASEDIR=${REPO}/external/VAGEN

EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
RUN_DATE=${RUN_DATE:-}
RUN_DIR=${RUN_DIR:-}
if [ -z "${RUN_DIR}" ]; then
  if [ -z "${RUN_DATE}" ]; then
    RUN_DATE=$(date +%Y-%m-%d)
  fi
  RUN_DIR=${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}
fi

SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
CONTROL_DIR=${RUN_DIR}/resume_${SLURM_JOB_ID}_train_control
ENV_CONTROL_DIR=${ENV_CONTROL_DIR:-${RUN_DIR}/external_env_4gpu}
ENV_URL_FILE=${ENV_CONTROL_DIR}/env_urls.txt
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
TOTAL_STEPS=${TOTAL_STEPS:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-24}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-12}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-8}
RAY_PORT=${RAY_PORT:-6379}

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

cleanup() {
  set +e
  echo "=== resume train cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null || true))
  for node in "${NODES[@]:-}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

{
  echo "=== Resume multinode VAGEN training starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Train nodes: ${SLURM_JOB_NODELIST}"
  echo "RUN_DIR: ${RUN_DIR}"
  echo "External env URL file: ${ENV_URL_FILE}"
  echo "Resume: trainer.resume_mode=auto from ${SAVE_CHECKPOINT_DIR}"
  echo "Batch: train=${TRAIN_BATCH_SIZE}, ppo_mini=${PPO_MINI_BATCH_SIZE}, val=${VAL_BATCH_SIZE}, total_steps=${TOTAL_STEPS}"
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

for i in $(seq 1 360); do
  if [ -f "${ENV_CONTROL_DIR}/failed" ]; then
    echo "ERROR external env job marked failed" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    exit 4
  fi
  if [ -f "${ENV_CONTROL_DIR}/ready" ] && [ -s "${ENV_URL_FILE}" ]; then
    break
  fi
  sleep 5
done
if [ ! -f "${ENV_CONTROL_DIR}/ready" ] || [ ! -s "${ENV_URL_FILE}" ]; then
  echo "ERROR timed out waiting for external env URLs" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 4
fi

echo "=== External env URLs ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
cat "${ENV_URL_FILE}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
WORKER_NODE=${NODES[1]}
HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "$HEAD_IP" ]; then HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}'); fi
IP_HEAD=${HEAD_IP}:${RAY_PORT}
echo "Head node: ${HEAD_NODE} ${HEAD_IP}; worker: ${WORKER_NODE}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  while read -r url; do
    [ -z \"\$url\" ] && continue
    echo checking \"\$url/health\"
    curl -fsS \"\$url/health\" >/dev/null
  done < '${ENV_URL_FILE}'
" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

TMP_CONFIG_DIR=$(mktemp -d -p "$CONTROL_DIR" tmpcfg.XXXXXX)
cp "${CONFIG_DIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${CONFIG_DIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

# shellcheck disable=SC1091
source "${SCRIPTDIR}/vagen_env_repro_cli.inc.sh"

for node in "${NODES[@]}"; do
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "
    REPO='${REPO}'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/common_env.sh'
  ray stop --force >/dev/null 2>&1 || true
  pkill -u \$USER -f 'sglang|SGLangHttpServer|sglang_server|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
  " || true
done
sleep 10

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  REPO='${REPO}'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/common_env.sh'
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=160 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' \
    --include-dashboard=false \
    --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 20

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -lc "
  set -euo pipefail
  REPO='${REPO}'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/common_env.sh'
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --address='${IP_HEAD}' \
    --num-cpus=160 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --block
" > "${RUN_DIR}/ray_worker_${SLURM_JOB_ID}_${WORKER_NODE}.log" 2>&1 &
sleep 25

echo "=== Launching resumed multinode VAGEN training ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
set +e
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  REPO='${REPO}'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/common_env.sh'
  PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path='${BASEDIR}/vagen/configs' \
    --config-name='vagen_multiturn' \
    data.train_files='${TMP_CONFIG_DIR}/train.yaml' \
    data.val_files='${TMP_CONFIG_DIR}/val.yaml' \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.seed=${VAGEN_DATA_SEED} \
    +data.base_seed=${VAGEN_BASE_SEED} \
    data.validation_shuffle=False \
    trainer.assert_val_env_composition=True \
    '+trainer.val_env_composition.navigation_base={count:60,eval_set:base}' \
    '+trainer.val_env_composition.navigation_common={count:60,eval_set:common_sense}' \
    data.max_prompt_length=3000 \
    data.max_response_length=20000 \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='${BASEDIR}/vagen/configs/agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
    actor_rollout_ref.rollout.agent.max_concurrent_trajectories=${AGENT_MAX_CONCURRENT_TRAJECTORIES} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.sync_module_states=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
    trainer.nnodes=2 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.max_actor_ckpt_to_keep=50 \
    trainer.max_critic_ckpt_to_keep=50 \
    trainer.project_name=nimloth_navigation \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir='${SAVE_CHECKPOINT_DIR}' \
    trainer.validation_data_dir='${RUN_DIR}/validation' \
    trainer.rollout_data_dir='${RUN_DIR}/rollout_data' \
    trainer.log_val_generations=32 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.resume_mode=auto \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.use_torch_compile=False \
    critic.model.fsdp_config.sync_module_states=False \
    huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto
" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "=== Resumed training finished rc=${TRAIN_EXIT} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
exit ${TRAIN_EXIT}
