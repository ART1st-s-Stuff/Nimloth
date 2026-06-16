#!/bin/bash
#SBATCH --job-name=sft1-rollouts-vagen48
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:8
#SBATCH --mem=0
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
ENV_CONTROL_DIR=${BASE_RUN_DIR}/external_env_dgx09_2gpu
ENV_URL_FILE=${ENV_CONTROL_DIR}/env_urls.txt
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
EXPERIMENT_NAME=sft1_rollouts_vagen48_vagen_train_val_test
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
TRAIN_GPUS_PER_NODE=8
RAY_PORT=6379
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=4
AGENT_MAX_CONCURRENT_TRAJECTORIES=6

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${RUN_DIR}/validation"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0'

cleanup() {
  set +e
  echo "=== cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null || true))
  for node in "${NODES[@]:-}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

write_split_yaml() {
  local split_name=$1
  local out_yaml=$2
  case "${split_name}" in
    train)
      cat > "${out_yaml}" <<YAML
# SFT1 train split from VAGEN train scenes. 3 x 1080 prompts x val_kwargs.n=1 = 3240 rollouts.
envs:
  - name: RemoteEnv
    n_envs: 1080
    data_source: navigation_base_train
    seed: [1, 1080, 1]
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
    n_envs: 1080
    data_source: navigation_common_train
    seed: [1, 1080, 1]
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
    n_envs: 1080
    data_source: navigation_long_horizon_train
    seed: [1, 1080, 1]
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
# SFT1 validation split held out from VAGEN train scenes. 3 x 120 prompts x val_kwargs.n=2 = 720 rollouts.
envs:
  - name: RemoteEnv
    n_envs: 120
    data_source: navigation_base_val_from_train
    seed: [1081, 1200, 1]
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
    n_envs: 120
    data_source: navigation_common_val_from_train
    seed: [1081, 1200, 1]
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
    n_envs: 120
    data_source: navigation_long_horizon_val_from_train
    seed: [1081, 1200, 1]
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
# SFT1 test split from VAGEN eval scenes. 5 x 60 prompts x val_kwargs.n=7 = 2100 rollouts.
envs:
  - name: RemoteEnv
    n_envs: 60
    data_source: navigation_base_test
    seed: [1, 60, 1]
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
    n_envs: 60
    data_source: navigation_common_test
    seed: [1, 60, 1]
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
    n_envs: 60
    data_source: navigation_complex_instruction_test
    seed: [1, 60, 1]
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
    n_envs: 60
    data_source: navigation_visual_appearance_test
    seed: [1, 60, 1]
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
    n_envs: 60
    data_source: navigation_long_horizon_test
    seed: [1, 60, 1]
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

run_split() {
  local split_name=$1
  local n_samples=$2
  local split_dir=${RUN_DIR}/validation/${split_name}
  local split_yaml=${CONTROL_DIR}/${split_name}.yaml
  local train_dummy=${CONTROL_DIR}/${split_name}_dummy_train.yaml

  mkdir -p "${split_dir}"
  write_split_yaml "${split_name}" "${split_yaml}"
  cp "${split_yaml}" "${train_dummy}"

  if [ -s "${split_dir}/${CHECKPOINT_STEP}.jsonl" ]; then
    echo "=== Split ${split_name} already has ${split_dir}/${CHECKPOINT_STEP}.jsonl; skipping for resume ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    return 0
  fi

  echo "=== Running split=${split_name} val_n=${n_samples} at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
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
      trainer.experiment_name=${EXPERIMENT_NAME}_${split_name} \
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

{
  echo "=== SFT1 VAGEN baseline rollout collection starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Purpose: rollout-only data collection for ai_tasks/sft1_exp.md; no actor/critic update."
  echo "Checkpoint source run: ${BASE_RUN_NAME}"
  echo "Original best checkpoint source: ${BASE_RUN_DIR}/checkpoints/global_step_${CHECKPOINT_STEP}/actor/huggingface"
  echo "Converted world_size=8 checkpoint path: ${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor"
  echo "Converted checkpoint latest file: $(cat ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || echo missing)"
  echo "External env URL file: ${ENV_URL_FILE}"
  echo "Output dir: ${RUN_DIR}/validation/{train,val,test}"
  echo "Split semantics: VAGEN train scenes are split by seed into train/val; VAGEN eval scenes are used as test."
  echo "Resume behavior: skip each split when ${CHECKPOINT_STEP}.jsonl already exists; Slurm uses --requeue."
  echo "Resource use: one 8-GPU model rollout job; dgx-09 remains external AI2-THOR env server only."
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -f "${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor/model_world_size_8_rank_7.pt" ]; then
  echo "ERROR missing converted checkpoint ${CHECKPOINT_DIR}/global_step_${CHECKPOINT_STEP}/actor/model_world_size_8_rank_7.pt" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 3
fi
if [ "$(cat ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || true)" != "${CHECKPOINT_STEP}" ]; then
  echo "ERROR converted latest_checkpointed_iteration.txt is not ${CHECKPOINT_STEP}; refusing to use unexpected checkpoint" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
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

NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${HEAD_IP}" ]; then HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}'); fi
IP_HEAD=${HEAD_IP}:${RAY_PORT}
echo "Head node: ${HEAD_NODE} ${HEAD_IP}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

echo "=== External env URLs ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
cat "${ENV_URL_FILE}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  while read -r url; do
    [ -z \"\$url\" ] && continue
    echo checking \"\$url/health\"
    curl -fsS \"\$url/health\" >/dev/null
  done < '${ENV_URL_FILE}'
" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true" || true
$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
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
sleep 25

run_split train 1
run_split val 2
run_split test 7

echo "=== SFT1 rollout collection finished at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
