#!/bin/bash
#SBATCH --job-name=vagen-dgx31-2env4train
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
CONVERT_PY=${ROOT}/experiments/navigation_baseline/convert_vagen_actor_only_to_world_size.py
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
EXPERIMENT_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
SOURCE_CHECKPOINT_STEP=50
SOURCE_HF_PATH=${SAVE_CHECKPOINT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/huggingface
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TOTAL_STEPS=100
TRAIN_BATCH_SIZE=48
PPO_MINI_BATCH_SIZE=12
VAL_BATCH_SIZE=12
AGENT_NUM_WORKERS=4
AGENT_MAX_CONCURRENT_TRAJECTORIES=6
ENV_GPUS=2
TRAIN_GPUS=4
TARGET_WORLD_SIZE=4
ENV_PORT=8400
RAY_PORT=$((26000 + (${SLURM_JOB_ID:-0} % 1000)))

mkdir -p "$RUN_DIR"

export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export HOME=${ROOT}/.home
export FLASHINFER_WORKSPACE_DIR=${ROOT}/.cache/flashinfer
mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"
export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/vagen-dgx31-train-${USER}-${SLURM_JOB_ID:-local}
mkdir -p "$TMPDIR"

ENV_PIDS=()
cleanup() {
  set +e
  if ((${#ENV_PIDS[@]} > 0)); then
    for pid in "${ENV_PIDS[@]}"; do kill "$pid" >/dev/null 2>&1 || true; done
    for pid in "${ENV_PIDS[@]}"; do wait "$pid" >/dev/null 2>&1 || true; done
  fi
  ray stop --force >/dev/null 2>&1 || true
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi
source ${ROOT}/.venv/bin/activate
source ${SCRIPTDIR}/setup_ai2thor_env.sh
cd "$BASEDIR"

export TOKENIZERS_PARALLELISM=true RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1
export TORCHINDUCTOR_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export VERL_SGLANG_MASTER_PORT_BASE=50000 VERL_SGLANG_MASTER_PORT_STRIDE=256
export GLOO_SOCKET_IFNAME=ibp24s0 NCCL_SOCKET_IFNAME=ibp24s0 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0
export PYTHONPATH=${ROOT}:${ROOT}/external/VAGEN:${ROOT}/external/VAGEN/verl:${PYTHONPATH:-}

{
  echo "=== DGX-31 resume 2env4train starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID:-srun}"
  echo "Host: $(hostname)"
  echo "Resume step ${SOURCE_CHECKPOINT_STEP} ws=${TARGET_WORLD_SIZE} -> total ${TOTAL_STEPS}"
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -d "${SOURCE_HF_PATH}" ]; then echo "ERROR missing HF ${SOURCE_HF_PATH}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 3; fi

if [ ! -f "${SAVE_CHECKPOINT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ]; then
  echo "=== Converting to ws${TARGET_WORLD_SIZE} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  CONVERT_CONTROL=$(mktemp -d -p "${RUN_DIR}" convert_ctrl.XXXXXX)
  cat > "${CONVERT_CONTROL}/dummy.yaml" <<'YAML'
envs:
  - name: RemoteEnv
    n_envs: 4
    data_source: navigation_base_train
    seed: [1, 4, 1]
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
  ray stop --force >/dev/null 2>&1 || true
  HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
  [ -z "${HEAD_IP}" ] && HEAD_IP=$(hostname -I | awk '{print $1}')
  CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=${RAY_PORT} --num-cpus=96 --num-gpus=${TARGET_WORLD_SIZE} --node-ip-address="${HEAD_IP}" --include-dashboard=false
  sleep 15
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 python3 "${CONVERT_PY}" \
    --config-path="${BASEDIR}/vagen/configs" --config-name=vagen_multiturn \
    hydra.searchpath=[file:///project/peilab/atst/nimloth/external/VAGEN/verl/verl/trainer/config] \
    data.train_files="${CONVERT_CONTROL}/dummy.yaml" data.val_files="${CONVERT_CONTROL}/dummy.yaml" \
    data.train_batch_size=${TARGET_WORLD_SIZE} data.val_batch_size=${TARGET_WORLD_SIZE} \
    data.max_prompt_length=3000 data.max_response_length=20000 algorithm.adv_estimator=gae algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${SOURCE_HF_PATH}" actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=${TARGET_WORLD_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.entropy_coeff=0.0 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
    'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False +actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=1 actor_rollout_ref.rollout.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False +actor_rollout_ref.ref.fsdp_config.sync_module_states=False \
    'trainer.logger=[console]' trainer.val_before_train=False trainer.val_only=False \
    trainer.n_gpus_per_node=${TARGET_WORLD_SIZE} trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=0 \
    trainer.max_actor_ckpt_to_keep=50 trainer.max_critic_ckpt_to_keep=50 trainer.project_name=nimloth_navigation \
    trainer.experiment_name=vagen50_world_size4_convert trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
    trainer.total_training_steps=1 trainer.resume_mode=disable +trainer.actor_only_convert=True \
    +trainer.convert_global_step=${SOURCE_CHECKPOINT_STEP} critic.optim.lr=1e-5 critic.model.use_remove_padding=True \
    critic.model.path="${SOURCE_HF_PATH}" critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 critic.ppo_max_token_len_per_gpu=32768 \
    critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.use_torch_compile=False +critic.model.fsdp_config.sync_module_states=False \
    huggingface_hub.hf_save_freq=null +ray_kwargs.ray_init.address=auto \
    2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  CONVERT_EXIT=${PIPESTATUS[0]}; set -e
  ray stop --force >/dev/null 2>&1 || true; sleep 10
  [ "${CONVERT_EXIT}" -ne 0 ] && { echo "ERROR convert rc=${CONVERT_EXIT}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 5; }
  [ ! -f "${SAVE_CHECKPOINT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ] && exit 5
  [ ! -f "${SAVE_CHECKPOINT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/critic/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ] && exit 5
  echo "=== ws${TARGET_WORLD_SIZE} conversion OK ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
else
  echo "=== ws${TARGET_WORLD_SIZE} checkpoint present; skip convert ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
fi

IFS=',' read -r -a ALLOC_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GOOD_ORD=()
for idx in "${!ALLOC_GPU_IDS[@]}"; do
  phys="${ALLOC_GPU_IDS[$idx]}"; phys=$(echo "$phys" | xargs); [ -z "$phys" ] && continue
  echo "=== AI2-THOR probe ordinal=${idx} phys=${phys} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$phys" timeout 150s python - <<'PY'
import ai2thor.controller
from ai2thor.platform import CloudRendering
c=ai2thor.controller.Controller(agentMode='default', gridSize=0.1, visibilityDistance=10,
    renderDepthImage=False, renderInstanceSegmentation=False, width=255, height=255, fieldOfView=100,
    platform=CloudRendering, gpu_device=0, server_timeout=60, server_start_timeout=60)
try:
    c.reset(scene='FloorPlan1')
finally:
    c.stop()
print('OK')
PY
  rc=$?; set -e
  [ "$rc" -eq 0 ] && GOOD_ORD+=("$idx")
done
[ "${#GOOD_ORD[@]}" -lt "$ENV_GPUS" ] && { echo "ERROR need ${ENV_GPUS} env GPUs" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"; exit 2; }

ENV_ORD=("${GOOD_ORD[@]:0:${ENV_GPUS}}")
TRAIN_ORD=()
for idx in "${!ALLOC_GPU_IDS[@]}"; do
  skip=0; for e in "${ENV_ORD[@]}"; do [ "$idx" = "$e" ] && skip=1; done
  [ "$skip" -eq 0 ] && TRAIN_ORD+=("$idx")
done
TRAIN_ORD=("${TRAIN_ORD[@]:0:${TRAIN_GPUS}}")
TRAIN_CUDA=$(python - <<PY
alloc="${ALLOC_GPU_IDS[*]}".split(); ords=[int(x) for x in "${TRAIN_ORD[*]}".split()]
print(','.join(alloc[i] for i in ords))
PY
)
echo "env ord=${ENV_ORD[*]} train ord=${TRAIN_ORD[*]} cuda=${TRAIN_CUDA}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

ENV_URLS=()
for env_i in "${!ENV_ORD[@]}"; do
  ord="${ENV_ORD[$env_i]}"; phys="${ALLOC_GPU_IDS[$ord]}"; port=$((ENV_PORT + env_i))
  ip=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}'); [ -z "$ip" ] && ip=$(hostname -I | awk '{print $1}')
  url="http://${ip}:${port}"; ENV_URLS+=("$url")
  CUDA_VISIBLE_DEVICES="$phys" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve --port ${port} --devices='[0]' \
    --max_envs 48 --max_inflight 48 --thread_pool_size 48 --session_timeout 7200.0 \
    > "${RUN_DIR}/env_server_${SLURM_JOB_ID:-srun}_${env_i}.log" 2>&1 &
  ENV_PIDS+=("$!")
done
for env_i in "${!ENV_URLS[@]}"; do
  port=$((ENV_PORT + env_i)); pid="${ENV_PIDS[$env_i]}"
  for i in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
    kill -0 "$pid" >/dev/null 2>&1 || { tail -200 "${RUN_DIR}/env_server_${SLURM_JOB_ID:-srun}_${env_i}.log"; exit 4; }
    sleep 5
  done
done

TMP_CONFIG_DIR=$(mktemp -d); ENV_URL_FILE=${TMP_CONFIG_DIR}/env_urls.txt
printf "%s\n" "${ENV_URLS[@]}" > "$ENV_URL_FILE"
cp "${SCRIPTDIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"; cp "${SCRIPTDIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml" "${TMP_CONFIG_DIR}/val.yaml"
cat "${ENV_URL_FILE}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

ray stop --force >/dev/null 2>&1 || true
HEAD_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}'); [ -z "$HEAD_IP" ] && HEAD_IP=$(hostname -I | awk '{print $1}')
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" ray start --head --port=${RAY_PORT} --num-cpus=96 --num-gpus=${TRAIN_GPUS} \
  --node-ip-address="${HEAD_IP}" --include-dashboard=false
sleep 15

set +e
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
  --config-path="${BASEDIR}/vagen/configs" --config-name=vagen_multiturn \
  data.train_files="${TMP_CONFIG_DIR}/train.yaml" data.val_files="${TMP_CONFIG_DIR}/val.yaml" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} data.val_batch_size=${VAL_BATCH_SIZE} \
  data.max_prompt_length=3000 data.max_response_length=20000 \
  algorithm.adv_estimator=gae algorithm.use_kl_in_reward=True algorithm.kl_ctrl.kl_coef=0.001 algorithm.gamma=1.0 algorithm.lam=1.0 \
  actor_rollout_ref.model.path=${MODEL_PATH} actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  +actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=24000 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
  actor_rollout_ref.rollout.agent.max_concurrent_trajectories=${AGENT_MAX_CONCURRENT_TRAJECTORIES} \
  actor_rollout_ref.rollout.disable_log_stats=False actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  +actor_rollout_ref.ref.fsdp_config.sync_module_states=False trainer.critic_warmup=0 'trainer.logger=[console]' \
  trainer.val_before_train=True trainer.n_gpus_per_node=${TRAIN_GPUS} trainer.nnodes=1 trainer.save_freq=1 trainer.test_freq=1 \
  trainer.max_actor_ckpt_to_keep=50 trainer.max_critic_ckpt_to_keep=50 trainer.project_name=nimloth_navigation \
  trainer.experiment_name=${EXPERIMENT_NAME} trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
  trainer.validation_data_dir=${RUN_DIR}/validation trainer.rollout_data_dir=${RUN_DIR}/rollout_data \
  trainer.log_val_generations=32 trainer.total_training_steps=${TOTAL_STEPS} trainer.resume_mode=auto \
  critic.optim.lr=1e-5 critic.model.use_remove_padding=True critic.model.path=${MODEL_PATH} \
  critic.model.enable_gradient_checkpointing=True critic.ppo_micro_batch_size_per_gpu=1 critic.ppo_max_token_len_per_gpu=32768 \
  critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True \
  critic.model.fsdp_config.use_torch_compile=False +critic.model.fsdp_config.sync_module_states=False \
  huggingface_hub.hf_save_freq=null +ray_kwargs.ray_init.address=auto \
  2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}; set -e
echo "=== finished rc=${TRAIN_EXIT} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
exit ${TRAIN_EXIT}
