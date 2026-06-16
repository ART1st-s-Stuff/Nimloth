#!/bin/bash
#SBATCH --job-name=sft1-ws2-dgx26
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodelist=dgx-26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=168
#SBATCH --gres=gpu:3
#SBATCH --mem=1400G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
CHECKPOINT_DIR=${SCRIPTDIR}/runs/vagen50_world_size2_from_hf/checkpoints
CHECKPOINT_STEP=50
EXPERIMENT_NAME=sft1_rollouts_vagen50_ws2_dgx26_1env2qwen
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
ENV_CONTROL_DIR=${CONTROL_DIR}/local_env
ENV_URL_FILE=${ENV_CONTROL_DIR}/env_urls.txt
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TRAIN_GPUS_PER_NODE=2
RAY_PORT=6379
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=8
AGENT_MAX_CONCURRENT_TRAJECTORIES=8
ENV_PORT=8400
ENV_MAX_ENVS=32
ENV_MAX_INFLIGHT=32

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${RUN_DIR}/validation" "${ENV_CONTROL_DIR}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0'

cleanup() {
  set +e
  echo "=== cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  if [ -f "${ENV_CONTROL_DIR}/env_server.pid" ]; then
    kill "$(cat ${ENV_CONTROL_DIR}/env_server.pid)" >/dev/null 2>&1 || true
  fi
  NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null || true))
  for node in "${NODES[@]:-}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
  done
  rm -f "${ENV_CONTROL_DIR}/ready"
}
trap cleanup EXIT

start_local_env_server() {
  local env_gpu=$1
  local node_ip=$2
  : > "${ENV_URL_FILE}"
  rm -f "${ENV_CONTROL_DIR}/ready" "${ENV_CONTROL_DIR}/failed" "${ENV_CONTROL_DIR}/env_server.pid"
  echo "http://${node_ip}:${ENV_PORT}" > "${ENV_URL_FILE}"

  echo "=== Starting local AI2-THOR env server on GPU ${env_gpu}, port ${ENV_PORT} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    source '${SCRIPTDIR}/setup_ai2thor_env.sh'
    CUDA_VISIBLE_DEVICES='${env_gpu}' PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
      --port '${ENV_PORT}' \
      --devices='[0]' \
      --max_envs '${ENV_MAX_ENVS}' \
      --max_inflight '${ENV_MAX_INFLIGHT}' \
      --thread_pool_size '${ENV_MAX_INFLIGHT}' \
      --session_timeout 7200.0
  " > "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
  echo $! > "${ENV_CONTROL_DIR}/env_server.pid"

  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${ENV_PORT}/health" >/dev/null 2>&1; then
      touch "${ENV_CONTROL_DIR}/ready"
      echo "local env health OK port=${ENV_PORT} tries=${i}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
      return 0
    fi
    sleep 3
  done
  echo "ERROR local env health failed port=${ENV_PORT}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  tail -200 "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${HEAD_NODE}.log" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true
  touch "${ENV_CONTROL_DIR}/failed"
  return 4
}

write_split_yaml() {
  local split_name=$1
  local out_yaml=$2
  local seed_start=$3
  local seed_end=$4
  local n_envs=$((seed_end - seed_start + 1))
  case "${split_name}" in
    train)
      cat > "${out_yaml}" <<YAML
# SFT1 train split shard from VAGEN train scenes.
envs:
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_base_train
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: base_train
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_common_train
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: common_sense_train
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_long_horizon_train
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: long_horizon_train
      prompt_format: eval_mode
      max_actions_per_step: 1
YAML
      ;;
    val)
      cat > "${out_yaml}" <<YAML
# SFT1 validation split shard held out from VAGEN train scenes.
envs:
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_base_val_from_train
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: base_train
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_common_val_from_train
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: common_sense_train
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_long_horizon_val_from_train
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: long_horizon_train
      prompt_format: eval_mode
      max_actions_per_step: 1
YAML
      ;;
    test)
      cat > "${out_yaml}" <<YAML
# SFT1 test split shard from VAGEN eval scenes.
envs:
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_base_test
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: base
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_common_test
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: common_sense
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_complex_instruction_test
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: complex_instruction
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_visual_appearance_test
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: visual_appearance
      prompt_format: eval_mode
      max_actions_per_step: 1
  - name: RemoteEnv
    n_envs: ${n_envs}
    data_source: navigation_long_horizon_test
    seed: [${seed_start}, ${seed_end}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: long_horizon
      prompt_format: eval_mode
      max_actions_per_step: 1
YAML
      ;;
    *)
      echo "unknown split ${split_name}" >&2
      exit 2
      ;;
  esac
}

run_split_shard() {
  local split_name=$1
  local shard_name=$2
  local seed_start=$3
  local seed_end=$4
  local n_samples=$5
  local split_dir=${RUN_DIR}/validation/${split_name}/${shard_name}
  local split_yaml=${CONTROL_DIR}/${split_name}_${shard_name}.yaml
  local train_dummy=${CONTROL_DIR}/${split_name}_${shard_name}_dummy_train.yaml

  mkdir -p "${split_dir}"
  write_split_yaml "${split_name}" "${split_yaml}" "${seed_start}" "${seed_end}"
  cp "${split_yaml}" "${train_dummy}"

  if [ -s "${split_dir}/${CHECKPOINT_STEP}.jsonl" ]; then
    echo "=== Shard ${split_name}/${shard_name} already has ${split_dir}/${CHECKPOINT_STEP}.jsonl; skipping for resume ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    return 0
  fi

  echo "=== Running split=${split_name} shard=${shard_name} seeds=${seed_start}-${seed_end} val_n=${n_samples} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    CUDA_VISIBLE_DEVICES='${MODEL_GPU_LIST}' PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
      --config-path='${BASEDIR}/vagen/configs' \
      --config-name='vagen_multiturn' \
      data.train_files='${train_dummy}' \
      data.val_files='${split_yaml}' \
      data.train_batch_size=24 \
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
      actor_rollout_ref.actor.ppo_mini_batch_size=24 \
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
      actor_rollout_ref.rollout.val_kwargs.n=${n_samples} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
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
      trainer.val_only=True \
      trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=0 \
      trainer.max_actor_ckpt_to_keep=50 \
      trainer.max_critic_ckpt_to_keep=50 \
      trainer.project_name=nimloth_navigation \
      trainer.experiment_name=${EXPERIMENT_NAME}_${split_name}_${shard_name} \
      trainer.default_local_dir='${CHECKPOINT_DIR}' \
      trainer.validation_data_dir='${split_dir}' \
      trainer.rollout_data_dir=null \
      trainer.log_val_generations=0 \
      trainer.total_training_steps=${CHECKPOINT_STEP} \
      trainer.resume_mode=auto \
      trainer.log_image.enable=True \
      trainer.log_image.max_pending=1 \
      trainer.log_image.png_compress_level=0 \
      trainer.replace_image_tokens_for_logging=True \
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
}

run_split() {
  local split_name=$1
  case "${split_name}" in
    train)
      run_split_shard train shard_001_180 1 180 1
      run_split_shard train shard_181_360 181 360 1
      run_split_shard train shard_361_540 361 540 1
      run_split_shard train shard_541_720 541 720 1
      run_split_shard train shard_721_900 721 900 1
      run_split_shard train shard_901_1080 901 1080 1
      ;;
    val)
      run_split_shard val shard_1081_1200 1081 1200 2
      ;;
    test)
      run_split_shard test shard_001_060 1 60 7
      ;;
    *)
      echo "unknown split ${split_name}" >&2
      exit 2
      ;;
  esac
}

{
  echo "=== SFT1 VAGEN baseline rollout collection starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Purpose: rollout-only data collection for ai_tasks/sft1_exp.md; no actor/critic update."
  echo "Checkpoint source run: ${BASE_RUN_NAME}"
  echo "Original best checkpoint source: ${BASE_RUN_DIR}/checkpoints/global_step_${CHECKPOINT_STEP}/actor/huggingface"
  echo "Converted world_size=2 checkpoint path: ${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor"
  echo "Converted checkpoint latest file: $(cat ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || echo missing)"
  echo "External env URL file: ${ENV_URL_FILE}"
  echo "Output dir: ${RUN_DIR}/validation/{train,val,test}/shard_*/${CHECKPOINT_STEP}.jsonl"
  echo "Split semantics: VAGEN train scenes are split by seed into train/val; VAGEN eval scenes are used as test."
  echo "Resume behavior: skip each seed shard when ${CHECKPOINT_STEP}.jsonl already exists; Slurm uses --requeue."
  echo "Shard plan: train=6 shards x 540 rollouts; val=1 shard x 720 rollouts; test=1 shard x 2100 rollouts."
  echo "Throughput knobs: data.val_batch_size=${VAL_BATCH_SIZE}; agent workers=${AGENT_NUM_WORKERS}; max concurrent trajectories=${AGENT_MAX_CONCURRENT_TRAJECTORIES}."
  echo "Resource use: dgx-26 normal job with 3 GPUs: 1 local AI2-THOR env GPU and 2 Qwen/Ray GPUs."
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -f "${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor/model_world_size_2_rank_1.pt" ]; then
  echo "ERROR missing converted checkpoint ${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor/model_world_size_2_rank_1.pt" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 3
fi
if [ "$(cat ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || true)" != "${CHECKPOINT_STEP}" ]; then
  echo "ERROR converted latest_checkpointed_iteration.txt is not ${CHECKPOINT_STEP}; refusing to use unexpected checkpoint" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 3
fi

NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${HEAD_IP}" ]; then HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}'); fi
IP_HEAD=${HEAD_IP}:${RAY_PORT}
echo "Head node: ${HEAD_NODE} ${HEAD_IP}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

IFS=',' read -r -a ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [ "${#ALLOC_GPUS[@]}" -lt 3 ]; then
  echo "ERROR expected 3 allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 6
fi
ENV_GPU=$(echo "${ALLOC_GPUS[0]}" | xargs)
MODEL_GPU_A=$(echo "${ALLOC_GPUS[1]}" | xargs)
MODEL_GPU_B=$(echo "${ALLOC_GPUS[2]}" | xargs)
MODEL_GPU_LIST="${MODEL_GPU_A},${MODEL_GPU_B}"
echo "GPU assignment: env=${ENV_GPU}; qwen/ray=${MODEL_GPU_LIST}; original CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

start_local_env_server "${ENV_GPU}" "${HEAD_IP}"

echo "=== Local env URLs ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
cat "${ENV_URL_FILE}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  curl -fsS 'http://127.0.0.1:${ENV_PORT}/health' >/dev/null
" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true" || true
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES='${MODEL_GPU_LIST}' ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=112 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' \
    --include-dashboard=false \
    --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 25

run_split train
run_split val
run_split test

echo "=== SFT1 rollout collection finished at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
