#!/bin/bash
#SBATCH --job-name=vagen-resume-dgx32-4train
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodelist=dgx-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:4
#SBATCH --mem=900G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
EXPERIMENT_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
CONTROL_DIR=${RUN_DIR}/resume_${SLURM_JOB_ID}_train_control
ENV_CONTROL_DIR=${RUN_DIR}/external_env_dgx09_2gpu
ENV_URL_FILE=${ENV_CONTROL_DIR}/env_urls.txt
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TOTAL_STEPS=100
TRAIN_BATCH_SIZE=48
PPO_MINI_BATCH_SIZE=12
VAL_BATCH_SIZE=12
AGENT_NUM_WORKERS=4
AGENT_MAX_CONCURRENT_TRAJECTORIES=6
TRAIN_GPUS_PER_NODE=4
TRAIN_NODES=1
TARGET_WORLD_SIZE=4
RAY_PORT=$((27000 + (${SLURM_JOB_ID:-0} % 1000)))

mkdir -p "$RUN_DIR" "$CONTROL_DIR"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0; export HYDRA_FULL_ERROR=1; export TORCHINDUCTOR_DISABLE=1; export TORCH_COMPILE_DISABLE=1; export TORCHDYNAMO_DISABLE=1; export TORCHINDUCTOR_COMPILE_THREADS=1; export VERL_SGLANG_MASTER_PORT_BASE=50000; export VERL_SGLANG_MASTER_PORT_STRIDE=256; export GLOO_SOCKET_IFNAME=ibp24s0; export NCCL_SOCKET_IFNAME=ibp24s0; export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=0'

cleanup() {
  set +e
  echo "=== resume dgx32 train cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  echo "=== Resume ws${TARGET_WORLD_SIZE} VAGEN on dgx-32 with dgx-09 external env at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Train: ${SLURM_JOB_NODELIST} x ${TRAIN_GPUS_PER_NODE} GPU"
  echo "Env: ${ENV_URL_FILE}"
  echo "Batch: train=${TRAIN_BATCH_SIZE}, ppo_mini=${PPO_MINI_BATCH_SIZE}, val=${VAL_BATCH_SIZE}, steps 50->${TOTAL_STEPS}"
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -f "${SAVE_CHECKPOINT_DIR}/global_step_50/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ]; then
  echo "ERROR missing world_size=${TARGET_WORLD_SIZE} actor checkpoint at global_step_50" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 3
fi

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

HEAD_NODE=dgx-32
HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "$HEAD_IP" ]; then HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}'); fi
echo "Head: ${HEAD_NODE} ${HEAD_IP}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  while read -r url; do
    [ -z \"\$url\" ] && continue
    curl -fsS \"\$url/health\" >/dev/null
  done < '${ENV_URL_FILE}'
" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

TMP_CONFIG_DIR=$(mktemp -d -p "$CONTROL_DIR" tmpcfg.XXXXXX)
cp "${SCRIPTDIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${SCRIPTDIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true; pkill -u \$USER -f 'sglang|SGLangHttpServer|sglang_server|torch/_inductor/compile_worker' >/dev/null 2>&1 || true" || true
sleep 10

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=96 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' \
    --include-dashboard=false \
    --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 20

echo "=== Launching resumed VAGEN training ws${TARGET_WORLD_SIZE} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
set +e
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path='${BASEDIR}/vagen/configs' \
    --config-name='vagen_multiturn' \
    data.train_files='${TMP_CONFIG_DIR}/train.yaml' \
    data.val_files='${TMP_CONFIG_DIR}/val.yaml' \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
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
    +actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
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
    +actor_rollout_ref.ref.fsdp_config.sync_module_states=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
    trainer.nnodes=${TRAIN_NODES} \
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
    +critic.model.fsdp_config.sync_module_states=False \
    huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto
" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "=== Resumed dgx32 training finished rc=${TRAIN_EXIT} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
exit ${TRAIN_EXIT}
