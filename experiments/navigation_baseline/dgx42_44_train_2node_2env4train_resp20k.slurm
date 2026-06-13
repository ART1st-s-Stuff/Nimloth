#!/bin/bash
#SBATCH --job-name=vagen-dgx42-44-2node-2env4train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-42,dgx-44
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:6
#SBATCH --mem=512G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
EXPERIMENT_NAME=vagen_nav_dgx42_44_2node_2env4train_original_base_common_sticky_semaphore_resp20k_envstay_fa283_single_action_promptfix
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TOTAL_STEPS=50
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=4
AGENT_MAX_CONCURRENT_TRAJECTORIES=6
ENV_GPUS_PER_NODE=2
TRAIN_GPUS_PER_NODE=4
ENV_PORT_BASE=8000
RAY_PORT=6379

mkdir -p "$RUN_DIR"
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
mkdir -p "$CONTROL_DIR"
: > "${CONTROL_DIR}/env_hosts.txt"

echo "=== Two-node VAGEN training starting at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Topology: 2 nodes; per node ${ENV_GPUS_PER_NODE} env GPUs + ${TRAIN_GPUS_PER_NODE} train GPUs"
echo "Training world: trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE}, trainer.nnodes=${SLURM_NNODES}"
echo "Train batch: ${TRAIN_BATCH_SIZE}; PPO mini batch: ${PPO_MINI_BATCH_SIZE}; max_response_length=20000"

NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$($SL/srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
IP_HEAD=${HEAD_IP}:${RAY_PORT}
echo "Head node: ${HEAD_NODE} (${HEAD_IP})"

cleanup() {
  set +e
  for node in "${NODES[@]}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc '
      if [ -f /tmp/vagen_${SLURM_JOB_ID}_env_pids ]; then
        xargs -r kill < /tmp/vagen_${SLURM_JOB_ID}_env_pids 2>/dev/null || true
      fi
      ray stop --force >/dev/null 2>&1 || true
    ' >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0'

# Per-node AI2-THOR probe, env-server launch, and train CUDA selection.
# These srun steps intentionally stay alive so Slurm does not reap env-server children.
PREP_PIDS=()
for node_idx in "${!NODES[@]}"; do
  node=${NODES[$node_idx]}
  echo "=== Preparing node ${node} ==="
  (
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" \
      bash "${SCRIPTDIR}/prepare_2node_env_ray_node.sh" \
        "$node" "$node_idx" "$CONTROL_DIR" "$RUN_DIR" "$SLURM_JOB_ID" \
        "$ENV_GPUS_PER_NODE" "$TRAIN_GPUS_PER_NODE" "$ENV_PORT_BASE"
  ) 2>&1 | tee "${RUN_DIR}/prepare_${SLURM_JOB_ID}_${node}.log" &
  PREP_PIDS+=("$!")
done

for node in "${NODES[@]}"; do
  ready_file="${CONTROL_DIR}/ready_${node}"
  echo "=== Waiting for env readiness on ${node} ==="
  ready=0
  for i in $(seq 1 180); do
    if [ -f "$ready_file" ]; then
      ready=1
      break
    fi
    for pid in "${PREP_PIDS[@]}"; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        echo "ERROR: env preparation process exited before all nodes became ready"
        exit 4
      fi
    done
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then
    echo "ERROR: timed out waiting for ${ready_file}"
    exit 4
  fi
done

ENV_URL_FILE=${CONTROL_DIR}/env_urls.txt
awk 'NF {print "http://" $0}' "${CONTROL_DIR}/env_hosts.txt" > "$ENV_URL_FILE"
echo "=== Env URLs ==="
cat "$ENV_URL_FILE"
if [ "$(awk 'NF {n++} END {print n+0}' "$ENV_URL_FILE")" -lt 4 ]; then
  echo "ERROR: expected 4 env servers for two-node 2env/node topology"
  exit 4
fi

TMP_CONFIG_DIR=$(mktemp -d -p "$CONTROL_DIR" tmpcfg.XXXXXX)
cp "${SCRIPTDIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${SCRIPTDIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

# Start one Ray cluster across both nodes. Ray sees only the 4 train GPUs per node.
echo "=== Stopping stale Ray on all allocated nodes ==="
for node in "${NODES[@]}"; do
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true" || true
done

HEAD_TRAIN_CUDA=$(cat "${CONTROL_DIR}/train_cuda_${HEAD_NODE}.txt")
echo "=== Starting Ray head on ${HEAD_NODE}, CUDA_VISIBLE_DEVICES=${HEAD_TRAIN_CUDA} ==="
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES='${HEAD_TRAIN_CUDA}' ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=96 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' \
    --include-dashboard=false \
    --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 20

for node in "${NODES[@]:1}"; do
  TRAIN_CUDA=$(cat "${CONTROL_DIR}/train_cuda_${node}.txt")
  echo "=== Starting Ray worker on ${node}, CUDA_VISIBLE_DEVICES=${TRAIN_CUDA} ==="
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    CUDA_VISIBLE_DEVICES='${TRAIN_CUDA}' ray start --address='${IP_HEAD}' \
      --num-cpus=96 \
      --num-gpus=${TRAIN_GPUS_PER_NODE} \
      --block
  " > "${RUN_DIR}/ray_worker_${SLURM_JOB_ID}_${node}.log" 2>&1 &
done
sleep 25

# Launch the single trainer on the Ray head; it consumes 4 train GPUs per node x 2 nodes = 8 train GPUs.
echo "=== Launching single multinode VAGEN training ==="
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
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
    trainer.nnodes=${SLURM_NNODES} \
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
    huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto
" 2>&1 | tee "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "=== Env server log tails ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
for f in "${RUN_DIR}"/env_server_${SLURM_JOB_ID}_*.log; do
  echo "--- $f ---" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  tail -80 "$f" 2>/dev/null | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true
done

echo "=== Two-node training finished rc=${TRAIN_EXIT} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
exit ${TRAIN_EXIT}
