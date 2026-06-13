#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN

TRAIN_NODES=(dgx-31 dgx-49)
ENV_NODE=dgx-36
HEAD_NODE=${TRAIN_NODES[0]}
WORKER_NODE=${TRAIN_NODES[1]}

EXPERIMENT_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
CONTROL_DIR=${RUN_DIR}/held_${SLURM_JOB_ID}_control
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TOTAL_STEPS=50
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=4
AGENT_MAX_CONCURRENT_TRAJECTORIES=6
ENV_GPUS=8
TRAIN_GPUS_PER_NODE=8
ENV_PORT_BASE=8200
RAY_PORT=6379

mkdir -p "$RUN_DIR" "$CONTROL_DIR"
: > "${CONTROL_DIR}/env_hosts.txt"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0'

cleanup() {
  set +e
  echo "=== Cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$ENV_NODE" bash -lc '
    if [ -f /tmp/vagen_3node_${SLURM_JOB_ID}_env_pids ]; then
      xargs -r kill < /tmp/vagen_3node_${SLURM_JOB_ID}_env_pids 2>/dev/null || true
    fi
  ' >/dev/null 2>&1 || true
  for node in "${TRAIN_NODES[@]}"; do
    $SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

{
  echo "=== Held-allocation 3-node VAGEN training starting at $(date) ==="
  echo "Hold Job ID: ${SLURM_JOB_ID}"
  echo "Train nodes: ${TRAIN_NODES[*]} (${TRAIN_GPUS_PER_NODE} GPUs/node; total 16 train GPUs)"
  echo "Env node: ${ENV_NODE} (${ENV_GPUS} env GPUs)"
  echo "Model init: ${MODEL_PATH}; resume_mode=disable via fresh run dir/auto no checkpoint"
  echo "Trainable: actor + critic PPO; frozen/ref: reference model; rollout: SGLang async"
  echo "Batch: train=${TRAIN_BATCH_SIZE}, ppo_mini=${PPO_MINI_BATCH_SIZE}; max_actions_per_step=1; max_turns=20; max_response_length=20000"
} | tee "${RUN_DIR}/${EXPERIMENT_NAME}.log"

# Start env servers only on dgx-36, all 8 GPUs.
echo "=== Starting env servers on ${ENV_NODE} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
$SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$ENV_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  source /project/peilab/atst/nimloth/experiments/navigation_baseline/setup_ai2thor_env.sh
  NODE_IP=\$(hostname -I | tr ' ' '\n' | awk '/^10\\.23\\./ {print; exit}')
  if [ -z \"\${NODE_IP}\" ]; then NODE_IP=\$(hostname -I | awk '{print \$1}'); fi
  echo \"[${ENV_NODE}] NODE_IP=\${NODE_IP}\"
  : > /tmp/vagen_3node_${SLURM_JOB_ID}_env_pids
  for gpu in 0 1 2 3 4 5 6 7; do
    port=\$(( ${ENV_PORT_BASE} + gpu ))
    echo \"\${NODE_IP}:\${port}\" >> '${CONTROL_DIR}/env_hosts.txt'
    echo \"[${ENV_NODE}] starting env gpu=\${gpu} port=\${port}\"
    CUDA_VISIBLE_DEVICES=\${gpu} PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
      --port \"\${port}\" \
      --devices='[0]' \
      --max_envs 48 \
      --max_inflight 48 \
      --thread_pool_size 48 \
      --session_timeout 7200.0 \
      > '${RUN_DIR}/env_server_${SLURM_JOB_ID}_${ENV_NODE}_'\${gpu}'.log' 2>&1 &
    echo \$! >> /tmp/vagen_3node_${SLURM_JOB_ID}_env_pids
  done
  for gpu in 0 1 2 3 4 5 6 7; do
    port=\$(( ${ENV_PORT_BASE} + gpu ))
    ok=0
    for i in \$(seq 1 120); do
      if curl -fsS \"http://127.0.0.1:\${port}/health\" >/dev/null 2>&1; then
        echo \"[${ENV_NODE}] health OK gpu=\${gpu} port=\${port} tries=\${i}\"
        ok=1; break
      fi
      sleep 3
    done
    if [ \"\${ok}\" -ne 1 ]; then
      echo \"ERROR env server failed gpu=\${gpu} port=\${port}\"
      tail -200 '${RUN_DIR}/env_server_${SLURM_JOB_ID}_${ENV_NODE}_'\${gpu}'.log' || true
      exit 4
    fi
  done
  touch '${CONTROL_DIR}/ready_env'
  wait \$(cat /tmp/vagen_3node_${SLURM_JOB_ID}_env_pids)
" > "${RUN_DIR}/env_node_${SLURM_JOB_ID}_${ENV_NODE}.log" 2>&1 &
ENV_STEP_PID=$!

for i in $(seq 1 180); do
  [ -f "${CONTROL_DIR}/ready_env" ] && break
  if ! kill -0 "$ENV_STEP_PID" >/dev/null 2>&1; then
    echo "ERROR: env step exited early" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    tail -200 "${RUN_DIR}/env_node_${SLURM_JOB_ID}_${ENV_NODE}.log" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true
    exit 4
  fi
  sleep 5
done
if [ ! -f "${CONTROL_DIR}/ready_env" ]; then
  echo "ERROR: timed out waiting for env readiness" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  tail -200 "${RUN_DIR}/env_node_${SLURM_JOB_ID}_${ENV_NODE}.log" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true
  exit 4
fi

ENV_URL_FILE=${CONTROL_DIR}/env_urls.txt
awk 'NF {print "http://" $0}' "${CONTROL_DIR}/env_hosts.txt" > "$ENV_URL_FILE"
echo "=== Env URLs ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
cat "$ENV_URL_FILE" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
if [ "$(awk 'NF {n++} END {print n+0}' "$ENV_URL_FILE")" -ne 8 ]; then
  echo "ERROR: expected 8 env servers" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 4
fi

TMP_CONFIG_DIR=$(mktemp -d -p "$CONTROL_DIR" tmpcfg.XXXXXX)
cp "${SCRIPTDIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${SCRIPTDIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

HEAD_IP=$($SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "$HEAD_IP" ]; then
  HEAD_IP=$($SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
fi
IP_HEAD=${HEAD_IP}:${RAY_PORT}
echo "=== Ray head ${HEAD_NODE} ${HEAD_IP} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

for node in "${TRAIN_NODES[@]}"; do
  $SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true" || true
done

$SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=160 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' \
    --include-dashboard=false \
    --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 20

$SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --address='${IP_HEAD}' \
    --num-cpus=160 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --block
" > "${RUN_DIR}/ray_worker_${SLURM_JOB_ID}_${WORKER_NODE}.log" 2>&1 &
sleep 25

# Launch single trainer on Ray head: 8 train GPUs/node x 2 train nodes = 16 train GPUs.
echo "=== Launching single multinode VAGEN training ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
set +e
$SL/srun --jobid=${SLURM_JOB_ID} --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
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
    huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto
" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "=== Env server log tails ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
for f in "${RUN_DIR}"/env_server_${SLURM_JOB_ID}_*.log; do
  echo "--- $f ---" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  tail -80 "$f" 2>/dev/null | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true
done

echo "=== 3-node training finished rc=${TRAIN_EXIT} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
exit ${TRAIN_EXIT}
