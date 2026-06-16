#!/bin/bash
#SBATCH --job-name=sft1-v48-ws2-vllm
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
CHECKPOINT_DIR=${SCRIPTDIR}/runs/vagen48_world_size8_from_hf/checkpoints
CHECKPOINT_STEP=48
EXPERIMENT_NAME=sft1_rollouts_vagen48_ws2_2node_externalenv_vllm
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
ENV_URL_FILE=${CONTROL_DIR}/env_urls.txt
EXTERNAL_ENV_URLS=(http://10.23.0.101:8400 http://10.23.0.101:8401 http://10.23.0.101:8402 http://10.23.0.101:8403)
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TRAIN_GPUS_PER_NODE=1
RAY_PORT=6379
RAY_CMD=${ROOT}/.venv_vllm128_tmp/bin/ray
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=8
AGENT_MAX_CONCURRENT_TRAJECTORIES=8
mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${RUN_DIR}/validation"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv_vllm128_tmp/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv_vllm128_tmp/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export PYTHONPATH=/project/peilab/atst/nimloth:/project/peilab/atst/nimloth/external/VAGEN:/project/peilab/atst/nimloth/external/VAGEN/verl:${PYTHONPATH:-}; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

cleanup() {
  set +e
  echo "=== cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null || true))
  for node in "${NODES[@]:-}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "${COMMON_ENV}; ${RAY_CMD} stop --force >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

write_external_env_urls() {
  : > "${ENV_URL_FILE}"
  for url in "${EXTERNAL_ENV_URLS[@]}"; do
    echo "${url}" >> "${ENV_URL_FILE}"
  done
}

check_external_envs_from_head() {
  echo "=== Checking external AI2-THOR env health from ${HEAD_NODE} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  for url in "${EXTERNAL_ENV_URLS[@]}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
      set -euo pipefail
      timeout 10 curl -fsS '${url}/health' >/dev/null
    "
    echo "external env health OK: ${url}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  done
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
      echo "Unknown split: ${split_name}" >&2
      exit 1
      ;;
  esac
}

write_dummy_train_yaml() {
  local split_name=$1
  local out_yaml=$2
  cat > "${out_yaml}" <<YAML
# Dummy train file for val_only run on ${split_name} shard.
envs: []
YAML
}

run_shard() {
  local split_name=$1
  local shard_name=$2
  local seed_start=$3
  local seed_end=$4
  local n_samples=$5

  local split_dir="${RUN_DIR}/validation/${split_name}/${shard_name}"
  mkdir -p "${split_dir}"
  local out_jsonl="${split_dir}/${CHECKPOINT_STEP}.jsonl"
  if [ -f "${out_jsonl}" ]; then
    echo "[resume-skip] ${split_name}/${shard_name} already has ${CHECKPOINT_STEP}.jsonl" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    return 0
  fi

  local val_yaml="${CONTROL_DIR}/${split_name}_${shard_name}.yaml"
  local dummy_train_yaml="${CONTROL_DIR}/${split_name}_${shard_name}_dummy_train.yaml"
  write_split_yaml "${split_name}" "${val_yaml}" "${seed_start}" "${seed_end}"
  write_dummy_train_yaml "${split_name}" "${dummy_train_yaml}"

  echo "=== Running split=${split_name} shard=${shard_name} seeds=${seed_start}-${seed_end} n_samples=${n_samples} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
      --config-path='${BASEDIR}/vagen/configs' \
      --config-name='vagen_multiturn' \
      data.train_files='${dummy_train_yaml}' \
      data.val_files='${val_yaml}' \
      data.train_batch_size=${VAL_BATCH_SIZE} \
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
      actor_rollout_ref.rollout.name=vllm \
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
      +ray_kwargs.ray_init.address='auto'
  " 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
}

run_split() {
  local split_name=$1
  case "${split_name}" in
    train)
      run_shard train shard_001_180 1 180 1
      run_shard train shard_181_360 181 360 1
      run_shard train shard_361_540 361 540 1
      run_shard train shard_541_720 541 720 1
      run_shard train shard_721_900 721 900 1
      run_shard train shard_901_1080 901 1080 1
      ;;
    val)
      run_shard val shard_1081_1200 1081 1200 2
      ;;
    test)
      run_shard test shard_001_060 1 60 7
      ;;
    *)
      echo "unknown split ${split_name}" >&2
      exit 2
      ;;
  esac
}

main() {
  echo "=== ${EXPERIMENT_NAME} job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "Output dir: ${RUN_DIR}/validation/{train,val,test}/shard_*/${CHECKPOINT_STEP}.jsonl" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "Resume behavior: skip each seed shard when ${CHECKPOINT_STEP}.jsonl already exists; Slurm uses --requeue." | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "Shard plan: train=6 shards x 540 rollouts; val=1 shard x 720 rollouts; test=1 shard x 2100 rollouts." | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "Total planned rollouts: 6060." | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

  mapfile -t NODES < <($SL/scontrol show hostnames "$SLURM_JOB_NODELIST")
  HEAD_NODE=${NODES[0]}
  HEAD_ADDR=127.0.0.1
  export HEAD_NODE HEAD_ADDR
  NODE_RANK=$(printf '%s
' "${NODES[@]}" | nl -v0 | awk -v me="$(hostname)" '$2==me{print $1}')
  export NODE_RANK

  write_external_env_urls
  check_external_envs_from_head

  $SL/srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "${COMMON_ENV}; local_ip=127.0.0.1; echo ray_head_ip=\$local_ip; ${RAY_CMD} stop --force >/dev/null 2>&1 || true; ${RAY_CMD} start --head --node-ip-address=\$local_ip --port=${RAY_PORT} --num-gpus=${TRAIN_GPUS_PER_NODE} --num-cpus=${SLURM_CPUS_PER_TASK}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  for node in "${NODES[@]:1}"; do
    $SL/srun --nodes=1 --ntasks=1 -w "$node" bash -lc "${COMMON_ENV}; local_ip=\$(hostname -I | tr ' ' '\n' | awk '/^10\\.23\\./ {print; exit}'); if [ -z \"\$local_ip\" ]; then local_ip=\$(hostname -I | awk '{print \$1}'); fi; echo ray_worker_ip=\$local_ip; ${RAY_CMD} stop --force >/dev/null 2>&1 || true; ${RAY_CMD} start --node-ip-address=\$local_ip --address=${HEAD_ADDR}:${RAY_PORT} --num-gpus=${TRAIN_GPUS_PER_NODE} --num-cpus=${SLURM_CPUS_PER_TASK}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  done

  export RAY_ADDRESS="${HEAD_ADDR}:${RAY_PORT}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1

  local_cmd="${COMMON_ENV}; export RAY_ADDRESS=auto; python -c 'import importlib.util; print(\"vllm_import=\", importlib.util.find_spec(\"vllm\") is not None)'"
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "$local_cmd" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

  run_split train
  run_split val
  run_split test
}

main "$@"
