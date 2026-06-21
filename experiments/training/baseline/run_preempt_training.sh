#!/usr/bin/env bash
# Co-located env + train on nodes already held by hold_preempt.slurm.
# Usage: HOLD_JOB=<id> EXPERIMENT_NAME=... bash run_preempt_training.sh
set -euo pipefail

REPO=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${REPO}/experiments/training/baseline
NAV_SCRIPTDIR=${REPO}/experiments/navigation_baseline
CONFIG_DIR=${REPO}/configs/training/baseline
BASEDIR=${REPO}/external/VAGEN
PREPARE=${NAV_SCRIPTDIR}/prepare_2node_env_ray_node.sh

HOLD_JOB=${HOLD_JOB:?set HOLD_JOB}
: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"

mapfile -t NODES < <($SL/scontrol show hostnames "$($SL/squeue -j "${HOLD_JOB}" -h -o "%N")")
if [ "${#NODES[@]}" -lt 2 ]; then
  echo "ERROR: need 2 nodes in hold ${HOLD_JOB}, got ${#NODES[@]}"
  exit 1
fi
HEAD_NODE=${NODES[0]}

RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
CONTROL_DIR=${RUN_DIR}/held_${HOLD_JOB}_control

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
TOTAL_STEPS=${TOTAL_STEPS:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-24}
TEST_FREQ=${TEST_FREQ:-10}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-6}
ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-2}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
ENV_PORT_BASE=${ENV_PORT_BASE:-8400}
RAY_PORT=${RAY_PORT:-6379}
RESUME_FROM_STEP=${RESUME_FROM_STEP:-}
PRUNE_CHECKPOINTS=${PRUNE_CHECKPOINTS:-0}
ENABLE_WANDB=${ENABLE_WANDB:-0}
if [ "${ENABLE_WANDB}" = "1" ]; then
  LOGGER_HYDRA="['console','wandb']"
else
  LOGGER_HYDRA="['console']"
fi

RESUME_MODE=disable
RESUME_PATH_ARG=""
if [ -n "${RESUME_FROM_STEP}" ]; then
  RESUME_MODE=resume_path
  RESUME_PATH_ARG="trainer.resume_from_path='${SAVE_CHECKPOINT_DIR}/global_step_${RESUME_FROM_STEP}'"
fi

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"
: > "${CONTROL_DIR}/env_hosts.txt"

# shellcheck disable=SC1091
source "${SCRIPTDIR}/vagen_env_repro_cli.inc.sh"

cleanup() {
  set +e
  echo "=== preempt training cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  for node in "${NODES[@]}"; do
    $SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc '
      if [ -f /tmp/vagen_'"${HOLD_JOB}"'_env_pids ]; then
        xargs -r kill < /tmp/vagen_'"${HOLD_JOB}"'_env_pids 2>/dev/null || true
      fi
      REPO='"${REPO}"'
      # shellcheck disable=SC1091
      source '"${SCRIPTDIR}"'/common_env.sh
      ray stop --force >/dev/null 2>&1 || true
      pkill -u \$USER -f 'vllm|VLLM|torch/_inductor/compile_worker' >/dev/null 2>&1 || true
    ' >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

{
  echo "=== preempt co-located training starting at $(date) ==="
  echo "Hold Job ID: ${HOLD_JOB}"
  echo "Nodes: ${NODES[*]}"
  echo "Per node: ${ENV_GPUS_PER_NODE} env + ${TRAIN_GPUS_PER_NODE} train GPUs"
  echo "RUN_DIR: ${RUN_DIR}"
  if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "Resume: global_step_${RESUME_FROM_STEP} (mode=${RESUME_MODE})"
  fi
} | tee "${RUN_DIR}/${EXPERIMENT_NAME}.log"

PREP_PIDS=()
for node_idx in "${!NODES[@]}"; do
  node=${NODES[$node_idx]}
  echo "=== Preparing ${node} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  (
    $SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$node" \
      bash "${PREPARE}" \
        "$node" "$node_idx" "$CONTROL_DIR" "$RUN_DIR" "$HOLD_JOB" \
        "$ENV_GPUS_PER_NODE" "$TRAIN_GPUS_PER_NODE" "$ENV_PORT_BASE"
  ) > "${RUN_DIR}/prepare_${HOLD_JOB}_${node}.log" 2>&1 &
  PREP_PIDS+=("$!")
done

for node in "${NODES[@]}"; do
  ready_file="${CONTROL_DIR}/ready_${node}"
  for i in $(seq 1 180); do
    [ -f "$ready_file" ] && break
    for pid in "${PREP_PIDS[@]}"; do
      kill -0 "$pid" >/dev/null 2>&1 || { echo "ERROR prep exited early on ${node}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 4; }
    done
    sleep 5
  done
  [ -f "$ready_file" ] || { echo "ERROR timeout ready_${node}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 4; }
done

ENV_URL_FILE=${CONTROL_DIR}/env_urls.txt
awk 'NF {print "http://" $0}' "${CONTROL_DIR}/env_hosts.txt" > "$ENV_URL_FILE"
echo "=== Env URLs ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
cat "$ENV_URL_FILE" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

TMP_CONFIG_DIR=$(mktemp -d -p "$CONTROL_DIR" tmpcfg.XXXXXX)
cp "${CONFIG_DIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${CONFIG_DIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

HEAD_IP=$($SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[ -n "$HEAD_IP" ] || HEAD_IP=$($SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
IP_HEAD=${HEAD_IP}:${RAY_PORT}

for node in "${NODES[@]}"; do
  $SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "
    REPO='${REPO}'; source '${SCRIPTDIR}/common_env.sh'; ray stop --force >/dev/null 2>&1 || true
  " || true
done
sleep 10

HEAD_TRAIN_CUDA=$(cat "${CONTROL_DIR}/train_cuda_${HEAD_NODE}.txt")
echo "=== Ray head ${HEAD_NODE} CUDA=${HEAD_TRAIN_CUDA} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
$SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  REPO='${REPO}'; source '${SCRIPTDIR}/common_env.sh'
  export RAY_raylet_start_wait_time_s=120
  CUDA_VISIBLE_DEVICES='${HEAD_TRAIN_CUDA}' ray start --head \
    --port=${RAY_PORT} --num-cpus=96 --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' --include-dashboard=false --disable-usage-stats --block
" > "${RUN_DIR}/ray_head_${HOLD_JOB}_${HEAD_NODE}.log" 2>&1 &
HEAD_RAY_PID=$!

echo "=== Waiting for Ray head ${HEAD_IP}:${RAY_PORT} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
head_ready=0
for i in $(seq 1 60); do
  if $SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc \
    "python3 - <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('${HEAD_IP}', ${RAY_PORT}))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY" >/dev/null 2>&1; then
    head_ready=1
    echo "ray head port ready after ${i} checks" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    break
  fi
  kill -0 "${HEAD_RAY_PID}" >/dev/null 2>&1 || { echo "ERROR: Ray head process died" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 4; }
  sleep 5
done
[ "${head_ready}" -eq 1 ] || { echo "ERROR: Ray head port not ready" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 4; }
sleep 10

for node in "${NODES[@]:1}"; do
  TRAIN_CUDA=$(cat "${CONTROL_DIR}/train_cuda_${node}.txt")
  echo "=== Ray worker ${node} CUDA=${TRAIN_CUDA} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "
    set -euo pipefail
    REPO='${REPO}'; source '${SCRIPTDIR}/common_env.sh'
    CUDA_VISIBLE_DEVICES='${TRAIN_CUDA}' ray start --address='${IP_HEAD}' \
      --num-cpus=96 --num-gpus=${TRAIN_GPUS_PER_NODE} --block
  " > "${RUN_DIR}/ray_worker_${HOLD_JOB}_${node}.log" 2>&1 &
done
sleep 45

NEED_GPUS=$((TRAIN_GPUS_PER_NODE * ${#NODES[@]}))
ray_ready=0
for i in $(seq 1 48); do
  ray_gpus=$($SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
    REPO='${REPO}'; source '${SCRIPTDIR}/common_env.sh'
    python3 - <<'PY'
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR')
    print(int(ray.cluster_resources().get('GPU', 0)))
finally:
    ray.shutdown()
PY
  " 2>/dev/null | tail -1 || true)
  [ -n "${ray_gpus}" ] && echo "ray gpu check ${i}: ${ray_gpus} (need ${NEED_GPUS})" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  if [ "${ray_gpus:-0}" -ge "${NEED_GPUS}" ]; then ray_ready=1; break; fi
  sleep 5
done
[ "${ray_ready}" -eq 1 ] || { echo "ERROR: Ray not ready" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 4; }

echo "=== Launching VAGEN PPO training ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
set +e
$SL/srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  REPO='${REPO}'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/common_env.sh'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/vagen_paper_ppo_cli.inc.sh'
  # shellcheck disable=SC1091
  source '${SCRIPTDIR}/vagen_rollout_vllm_cli.inc.sh'
  cd '${BASEDIR}'
  PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path='${BASEDIR}/vagen/configs' --config-name='vagen_multiturn' \
    data.train_files='${TMP_CONFIG_DIR}/train.yaml' \
    data.val_files='${TMP_CONFIG_DIR}/val.yaml' \
    data.custom_cls.path='${REPO}/external/VAGEN/vagen/gym_agent_dataset.py' \
    data.train_batch_size=${TRAIN_BATCH_SIZE} data.val_batch_size=${VAL_BATCH_SIZE} \
    data.seed=${VAGEN_DATA_SEED} +data.base_seed=${VAGEN_BASE_SEED} \
    data.validation_shuffle=False \
    +trainer.assert_val_env_composition=True \
    '+trainer.val_env_composition.navigation_base={count:60,eval_set:base}' \
    '+trainer.val_env_composition.navigation_common={count:60,eval_set:common_sense}' \
    \"\${VAGEN_PAPER_PPO_ARGS[@]}\" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \"\${VAGEN_ROLLOUT_VLLM_ARGS[@]}\" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='${BASEDIR}/vagen/configs/agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
    actor_rollout_ref.rollout.agent.max_concurrent_trajectories=${AGENT_MAX_CONCURRENT_TRAJECTORIES} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 trainer.logger=${LOGGER_HYDRA} trainer.val_before_train=True \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} trainer.nnodes=${#NODES[@]} \
    trainer.save_freq=1 trainer.test_freq=${TEST_FREQ} \
    trainer.max_actor_ckpt_to_keep=null trainer.max_critic_ckpt_to_keep=null \
    trainer.project_name=nimloth_navigation trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir='${SAVE_CHECKPOINT_DIR}' \
    trainer.validation_data_dir='${RUN_DIR}/validation' \
    trainer.rollout_data_dir='${RUN_DIR}/rollout_data' \
    trainer.log_val_generations=32 trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.resume_mode=${RESUME_MODE} \
    ${RESUME_PATH_ARG} \
    critic.model.use_remove_padding=True critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True \
    huggingface_hub.hf_save_freq=null +ray_kwargs.ray_init.address=auto
" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "=== preempt training finished rc=${TRAIN_EXIT} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
exit ${TRAIN_EXIT}
