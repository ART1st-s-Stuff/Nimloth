#!/bin/bash
#SBATCH --job-name=sft1-eval-v48
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodelist=dgx-26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --requeue

set -euo pipefail

: "${EVAL_TAG:?EVAL_TAG required}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR required}"
: "${CHECKPOINT_STEP:?CHECKPOINT_STEP required}"

MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor/huggingface}"

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
EXPERIMENT_NAME=sft1_eval_vagen48_${EVAL_TAG}
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
ENV_URL_FILE=${CONTROL_DIR}/env_urls.txt
EXTERNAL_ENV_URLS=(http://10.23.0.101:8400 http://10.23.0.101:8401 http://10.23.0.101:8402 http://10.23.0.101:8403)
TRAIN_GPUS_PER_NODE=2
RAY_PORT=6379
RAY_CMD=${ROOT}/.venv_vllm128_tmp/bin/ray
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=8
AGENT_MAX_CONCURRENT_TRAJECTORIES=8
mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${RUN_DIR}/validation"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; module load slurm >/dev/null 2>&1 || true; module load nvhpc-hpcx-cuda12/23.11 >/dev/null 2>&1 || true; export VLLM_ATTENTION_BACKEND=TORCH_SDPA; export VLLM_USE_FLASHINFER_SAMPLER=0; export PATH=/cm/shared/apps/nvhpc/23.11/Linux_x86_64/23.11/compilers/bin:/project/peilab/atst/nimloth/.venv_vllm128_tmp/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; export PYTHONPATH=/project/peilab/atst/nimloth:/project/peilab/atst/nimloth/external/VAGEN:/project/peilab/atst/nimloth/external/VAGEN/verl:${PYTHONPATH:-}; source /project/peilab/atst/nimloth/.venv_vllm128_tmp/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

cleanup() {
  set +e
  echo "=== cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  bash -lc "${COMMON_ENV}; ${RAY_CMD} stop --force >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT

write_external_env_urls() {
  : > "${ENV_URL_FILE}"
  for url in "${EXTERNAL_ENV_URLS[@]}"; do
    echo "${url}" >> "${ENV_URL_FILE}"
  done
}

check_external_envs() {
  echo "=== Checking external AI2-THOR env health ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  for url in "${EXTERNAL_ENV_URLS[@]}"; do
    timeout 10 curl -fsS "${url}/health" >/dev/null
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

run_split_shard() {
  local split_name=$1
  local shard_name=$2
  local seed_start=$3
  local seed_end=$4
  local n_samples=$5
  local ray_addr=$6
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
  bash -lc "${COMMON_ENV}; export RAY_ADDRESS=${ray_addr}; PYTHONUNBUFFERED=1 python -m vagen.main_ppo \
      --config-path='${BASEDIR}/vagen/configs' \
      --config-name='vagen_multiturn' \
      data.train_files='${train_dummy}' \
      data.val_files='${split_yaml}' \
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
      +actor_rollout_ref.model.override_config.attn_implementation=eager \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.use_fused_kernels=False \
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
      critic.model.path='${MODEL_PATH}' \
      critic.model.enable_gradient_checkpointing=True \
      critic.ppo_micro_batch_size_per_gpu=1 \
      critic.ppo_max_token_len_per_gpu=32768 \
      critic.model.fsdp_config.param_offload=True \
      critic.model.fsdp_config.optimizer_offload=True \
      huggingface_hub.hf_save_freq=null \
      +ray_kwargs.ray_init.address=auto" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
}

{
  echo "=== SFT1 val/test eval starting at $(date) ==="
  echo "EVAL_TAG=${EVAL_TAG}"
  echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "CHECKPOINT_STEP=${CHECKPOINT_STEP}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "Engine: vLLM 1-node x2 GPU"
  echo "Output: ${RUN_DIR}/validation/{val,test}/shard_*/${CHECKPOINT_STEP}.jsonl"
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR missing HF model path ${MODEL_PATH}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 3
fi

write_external_env_urls
check_external_envs

local_ip=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${local_ip}" ]; then local_ip=$(hostname -I | awk '{print $1}'); fi
echo "ray_local_ip=${local_ip}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
bash -lc "${COMMON_ENV}; ${RAY_CMD} stop --force >/dev/null 2>&1 || true; ${RAY_CMD} start --head --node-ip-address=${local_ip} --port=${RAY_PORT} --num-gpus=${TRAIN_GPUS_PER_NODE} --num-cpus=${SLURM_CPUS_PER_TASK:-56}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
sleep 15

bash -lc "${COMMON_ENV}; export RAY_ADDRESS=${local_ip}:${RAY_PORT}; python - <<'PY'
import importlib.util, ray, time
print('vllm_import=', importlib.util.find_spec('vllm') is not None)
for i in range(30):
    try:
        ray.init(address='auto', ignore_reinit_error=True)
        print('ray_available_resources=', ray.available_resources())
        break
    except Exception as e:
        print('ray_init_retry', i, e)
        time.sleep(2)
else:
    raise SystemExit('ray init failed')
PY" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

run_split_shard val shard_1081_1200 1081 1200 2 "${local_ip}:${RAY_PORT}"
run_split_shard test shard_001_060 1 60 7 "${local_ip}:${RAY_PORT}"

python3 ${SCRIPTDIR}/summarize_sft1_eval_rollouts.py \
  --root "${RUN_DIR}" \
  --step "${CHECKPOINT_STEP}" \
  | tee "${RUN_DIR}/summary_${CHECKPOINT_STEP}.json"

echo "=== SFT1 val/test eval finished at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
