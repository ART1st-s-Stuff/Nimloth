#!/bin/bash
#SBATCH --job-name=sft1-v79-greedy
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%A_%a.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --array=0-3%2
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
CHECKPOINT_DIR=${BASE_RUN_DIR}/checkpoints
CHECKPOINT_STEP=79
EXPERIMENT_NAME=sft1_rollouts_vagen79_greedy_parallel
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
ENV_CONTROL_DIR=${RUN_DIR}/external_env_4gpu
ENV_URL_FILE=${ENV_CONTROL_DIR}/env_urls.txt
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_control
MODEL_PATH=${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor/huggingface
TRAIN_GPUS_PER_NODE=2
RAY_PORT=$((6379 + SLURM_ARRAY_TASK_ID))
RAY_CMD=${ROOT}/.venv/bin/ray
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=12
AGENT_MAX_CONCURRENT_TRAJECTORIES=12
VAL_N=1

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${RUN_DIR}/validation" "${RUN_DIR}/checkpoints_rollout"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_$(hostname); export XDG_CACHE_HOME=/tmp/xdg_cache_${SLURM_JOB_ID}_$(hostname); export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}_$(hostname); mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"; unset VLLM_ATTENTION_BACKEND 2>/dev/null || true; export VLLM_USE_FLASHINFER_SAMPLER=0; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; export PYTHONPATH=/project/peilab/atst/nimloth/external/VAGEN:/project/peilab/atst/nimloth/external/VAGEN/verl:${PYTHONPATH:-}; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0; export HYDRA_FULL_ERROR=1; export TORCHINDUCTOR_DISABLE=1; export TORCH_COMPILE_DISABLE=1; export TORCHDYNAMO_DISABLE=1; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

cleanup() {
  set +e
  bash -lc "${COMMON_ENV}; ${RAY_CMD} stop --force >/dev/null 2>&1 || true"
}
trap cleanup EXIT

write_split_yaml() {
  local split_name=$1
  local out_yaml=$2
  local seed_start=$3
  local seed_end=$4
  local n_envs=$((seed_end - seed_start + 1))
  case "${split_name}" in
    train)
      cat > "${out_yaml}" <<YAML
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

write_dummy_train_yaml() {
  cat > "$1" <<'YAML'
envs: []
YAML
}

run_shard() {
  local split_name=$1
  local shard_name=$2
  local seed_start=$3
  local seed_end=$4
  local split_dir="${RUN_DIR}/validation/${split_name}/${shard_name}"
  local out_jsonl="${split_dir}/${CHECKPOINT_STEP}.jsonl"
  mkdir -p "${split_dir}"
  if [ -s "${out_jsonl}" ]; then
    echo "[skip] ${split_name}/${shard_name} already done" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    return 0
  fi
  local val_yaml="${CONTROL_DIR}/${split_name}_${shard_name}.yaml"
  local dummy_train="${CONTROL_DIR}/${split_name}_${shard_name}_dummy.yaml"
  write_split_yaml "${split_name}" "${val_yaml}" "${seed_start}" "${seed_end}"
  write_dummy_train_yaml "${dummy_train}"
  echo "=== array=${SLURM_ARRAY_TASK_ID} split=${split_name} shard=${shard_name} seeds=${seed_start}-${seed_end} greedy n=${VAL_N} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  bash -lc "${COMMON_ENV}; export RAY_ADDRESS=auto; PYTHONUNBUFFERED=1 python -m vagen.main_ppo \
      --config-path='${BASEDIR}/vagen/configs' \
      --config-name='vagen_multiturn' \
      data.train_files='${val_yaml}' \
      data.val_files='${val_yaml}' \
      data.train_batch_size=${VAL_BATCH_SIZE} \
      data.val_batch_size=${VAL_BATCH_SIZE} \
      data.max_prompt_length=3000 \
      data.max_response_length=20000 \
      algorithm.adv_estimator=reinforce_plus_plus \
      algorithm.use_kl_in_reward=False \
      algorithm.kl_ctrl.kl_coef=0.001 \
      algorithm.gamma=1.0 \
      algorithm.lam=1.0 \
      actor_rollout_ref.model.path='${MODEL_PATH}' \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      actor_rollout_ref.model.use_remove_padding=False \
      actor_rollout_ref.model.use_fused_kernels=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=24 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
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
      actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0 \
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
      trainer.experiment_name=${EXPERIMENT_NAME}_a${SLURM_ARRAY_TASK_ID}_${split_name}_${shard_name} \
      trainer.default_local_dir='${RUN_DIR}/checkpoints_rollout' \
      trainer.validation_data_dir='${split_dir}' \
      trainer.rollout_data_dir=null \
      trainer.log_val_generations=0 \
      trainer.total_training_steps=1 \
      trainer.resume_mode=disable \
      trainer.log_image.enable=True \
      trainer.log_image.max_pending=2 \
      trainer.log_image.png_compress_level=0 \
      trainer.replace_image_tokens_for_logging=True \
      critic.optim.lr=1e-5 \
      critic.model.use_remove_padding=False \
      +critic.model.override_config.attn_implementation=sdpa \
      critic.model.path='${MODEL_PATH}' \
      critic.model.enable_gradient_checkpointing=True \
      critic.ppo_micro_batch_size_per_gpu=1 \
      critic.ppo_max_token_len_per_gpu=24576 \
      critic.model.fsdp_config.param_offload=True \
      critic.model.fsdp_config.optimizer_offload=True \
      huggingface_hub.hf_save_freq=null \
      +ray_kwargs.ray_init.address=auto" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
}

wait_for_env() {
  for i in $(seq 1 360); do
    if [ -f "${ENV_CONTROL_DIR}/failed" ]; then
      echo "ERROR external env failed" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
      exit 4
    fi
    if [ -f "${ENV_CONTROL_DIR}/ready" ] && [ -s "${ENV_URL_FILE}" ]; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR timed out waiting for env URLs" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 4
}

main() {
  echo "=== ${EXPERIMENT_NAME} array_task=${SLURM_ARRAY_TASK_ID} job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=${SLURM_JOB_NODELIST} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "MODEL_PATH=${MODEL_PATH}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "Eval-consistent: greedy do_sample=False temperature=0 n=${VAL_N}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR missing model ${MODEL_PATH}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    exit 3
  fi
  wait_for_env
  while read -r url; do
    [ -z "$url" ] && continue
    curl -fsS "${url}/health" >/dev/null
  done < "${ENV_URL_FILE}"

  local_ip=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
  if [ -z "$local_ip" ]; then local_ip=$(hostname -I | awk '{print $1}'); fi
  bash -lc "${COMMON_ENV}; ${RAY_CMD} stop --force >/dev/null 2>&1 || true; pkill -u \$(whoami) -f 'ray::' >/dev/null 2>&1 || true; sleep 2; ${RAY_CMD} start --head --node-ip-address=${local_ip} --port=${RAY_PORT} --num-gpus=${TRAIN_GPUS_PER_NODE} --num-cpus=${SLURM_CPUS_PER_TASK}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  sleep 20
  bash -lc "${COMMON_ENV}; export RAY_ADDRESS=auto; python -c \"import ray; ray.init(); print(ray.available_resources())\"" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

  case "${SLURM_ARRAY_TASK_ID}" in
    0)
      run_shard train shard_001_180 1 180
      run_shard train shard_181_360 181 360
      ;;
    1)
      run_shard train shard_361_540 361 540
      run_shard train shard_541_720 541 720
      ;;
    2)
      run_shard train shard_721_900 721 900
      run_shard train shard_901_1080 901 1080
      ;;
    3)
      run_shard val shard_1081_1200 1081 1200
      run_shard test shard_001_060 1 60
      ;;
    *)
      echo "unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
      exit 2
      ;;
  esac
}

main "$@"
