#!/bin/bash
#SBATCH --job-name=sft1-v79-greedy
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%A_%a.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%A_%a.err
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

ROOT=${REPO:-/project/peilab/atst/nimloth}
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/training/sft1
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
BASEDIR=${ROOT}/external/VAGEN
CHECKPOINT_STEP=${INIT_HF_STEP}
ROLLOUT_DUMP_STEP=${ROLLOUT_DUMP_STEP:-0}
EXPERIMENT_NAME=${ROLLOUT_RUN_NAME}
RUN_DIR=${ROLLOUT_RUN_DIR}
ENV_CONTROL_DIR=${RUN_DIR}/external_env_4gpu
ENV_URL_FILE=${ENV_CONTROL_DIR}/env_urls.txt
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_control
MODEL_PATH=${INIT_HF}
TRAIN_GPUS_PER_NODE=${POLICY_GPU_COUNT:-2}
RAY_PORT=$((6379 + SLURM_ARRAY_TASK_ID))
RAY_DASHBOARD_PORT=$((8266 + SLURM_ARRAY_TASK_ID))
RAY_CMD="${PYTHON_ENV}/bin/python3 ${PYTHON_ENV}/bin/ray"
ROLLOUT_SMOKE=${ROLLOUT_SMOKE:-0}
ROLLOUT_GATE_120=${ROLLOUT_GATE_120:-0}
ROLLOUT_SOURCE_PARITY=${ROLLOUT_SOURCE_PARITY:-0}
ROLLOUT_SMOKE_SEED=${ROLLOUT_SMOKE_SEED:-1}
export ROLLOUT_SMOKE_EVAL_SET=${ROLLOUT_SMOKE_EVAL_SET:-base_train}
ROLLOUT_PROTOCOL=${ROLLOUT_PROTOCOL:-nimloth_source_eval}
ROLLOUT_INCLUDE_TEST=${ROLLOUT_INCLUDE_TEST:-1}
case "${ROLLOUT_PROTOCOL}" in
  nimloth_source_eval)
    ROLLOUT_PROMPT_FORMAT=source_eval_mode
    ROLLOUT_ACTION_SEP='|'
    ROLLOUT_MAX_TURNS=20
    ROLLOUT_MAX_TOKENS_PER_TURN=512
    ROLLOUT_DATA_MAX_RESPONSE_LENGTH=20000
    ROLLOUT_MAX_PROMPT_LENGTH=3000
    ROLLOUT_MAX_TRAJECTORY_LENGTH=23000
    ROLLOUT_WINDOW_SIZE=5
    ROLLOUT_DO_SAMPLE=False
    ROLLOUT_TEMPERATURE=0
    ROLLOUT_TOP_P=1.0
    ROLLOUT_TOP_K=-1
    ROLLOUT_STEP_LENGTH=0.3
    ROLLOUT_SUCCESS_THRESHOLD=1.0
    ROLLOUT_FORMAT_REWARD=0.0
    ROLLOUT_PER_TURN_FORMAT_REWARD=0.01
    ROLLOUT_SUCCESS_REWARD=1.0
    ROLLOUT_AGENT_NUM_WORKERS_DEFAULT=4
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
    ;;
  hligb_step10_eval)
    ROLLOUT_PROMPT_FORMAT=hligb_single_action_source
    ROLLOUT_ACTION_SEP=','
    ROLLOUT_MAX_TURNS=25
    ROLLOUT_MAX_TOKENS_PER_TURN=256
    ROLLOUT_DATA_MAX_RESPONSE_LENGTH=256
    ROLLOUT_MAX_PROMPT_LENGTH=1024
    ROLLOUT_MAX_TRAJECTORY_LENGTH=12000
    ROLLOUT_WINDOW_SIZE=5
    ROLLOUT_DO_SAMPLE=True
    ROLLOUT_TEMPERATURE=0.7
    ROLLOUT_TOP_P=0.95
    ROLLOUT_TOP_K=50
    ROLLOUT_STEP_LENGTH=0.5
    ROLLOUT_SUCCESS_THRESHOLD=1.5
    ROLLOUT_FORMAT_REWARD=0.5
    ROLLOUT_PER_TURN_FORMAT_REWARD=0.0
    ROLLOUT_SUCCESS_REWARD=10.0
    ROLLOUT_AGENT_NUM_WORKERS_DEFAULT=8
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.75
    ;;
  *)
    echo "ERROR unknown ROLLOUT_PROTOCOL=${ROLLOUT_PROTOCOL}" >&2
    exit 2
    ;;
esac
export ROLLOUT_PROMPT_FORMAT ROLLOUT_ACTION_SEP ROLLOUT_STEP_LENGTH ROLLOUT_SUCCESS_THRESHOLD
export ROLLOUT_FORMAT_REWARD ROLLOUT_PER_TURN_FORMAT_REWARD ROLLOUT_SUCCESS_REWARD
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-24}
# Match source step60 rollout orchestration values. In this pinned legacy path,
# max_workers controls environment RPC concurrency; policy topology is set by
# POLICY_GPU_COUNT.
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-${ROLLOUT_AGENT_NUM_WORKERS_DEFAULT}}
AGENT_MAX_CONCURRENT_TRAJECTORIES=${AGENT_MAX_CONCURRENT_TRAJECTORIES:-6}
VAL_N=${VAL_N:-1}
if [ "${ROLLOUT_SMOKE}" = "1" ]; then
  TRAIN_BATCH_SIZE=${TRAIN_GPUS_PER_NODE}
  VAL_BATCH_SIZE=1
  AGENT_NUM_WORKERS=1
  AGENT_MAX_CONCURRENT_TRAJECTORIES=1
fi

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${RUN_DIR}/validation" "${RUN_DIR}/checkpoints_rollout"

COMMON_ENV="export REPO='${ROOT}'; source '${SCRIPTDIR}/common_env.sh'; unset PYTORCH_CUDA_ALLOC_CONF; export FLASHINFER_WORKSPACE_DIR='${ROOT}/.cache/flashinfer'; export TRITON_CACHE_DIR='/tmp/triton_cache_${SLURM_JOB_ID}_$(hostname)'; export XDG_CACHE_HOME='/tmp/xdg_cache_${SLURM_JOB_ID}_$(hostname)'; export TORCH_EXTENSIONS_DIR='/tmp/torch_ext_${SLURM_JOB_ID}_$(hostname)'; mkdir -p \"\$FLASHINFER_WORKSPACE_DIR\" \"\$TORCH_EXTENSIONS_DIR\" \"\$TRITON_CACHE_DIR\" \"\$XDG_CACHE_HOME\"; unset VLLM_ATTENTION_BACKEND 2>/dev/null || true; export VLLM_USE_FLASHINFER_SAMPLER=0; cd '${BASEDIR}'; export RAY_DEDUP_LOGS=0; export HYDRA_FULL_ERROR=1; export TORCHINDUCTOR_DISABLE=1; export TORCH_COMPILE_DISABLE=1; export TORCHDYNAMO_DISABLE=1"

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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
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
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
YAML
      ;;
    *)
      echo "unknown split ${split_name}" >&2
      exit 2
      ;;
  esac
}

write_smoke_yaml() {
  local out_yaml=$1
  local seed=$2
  cat > "${out_yaml}" <<YAML
envs:
  - name: RemoteEnv
    n_envs: 1
    data_source: navigation_base_train_smoke
    seed: [${seed}, ${seed}, 1]
    max_turns: 20
    response_length_per_turn: 512
    config:
      url_file: "${ENV_URL_FILE}"
      timeout: 120
      retries: 6
      eval_set: base_train
      prompt_format: source_eval_mode
      max_actions_per_step: 1
      action_sep: "|"
      example_count: 0
      step_length: 0.3
      success_threshold: 1.0
      format_reward: 0.0
      per_turn_format_reward: 0.01
      success_reward: 1.0
YAML
}

write_split_parquet() {
  local split_name=$1
  local out_parquet=$2
  local seed_start=$3
  local seed_end=$4
  "${PYTHON_ENV}/bin/python3" - "${split_name}" "${out_parquet}" "${seed_start}" "${seed_end}" <<'PY'
import sys
from pathlib import Path
from datasets import Dataset

split_name, output, seed_start, seed_end = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
specs = {
    "train": [
        ("navigation_base_train", "base_train"),
        ("navigation_common_train", "common_sense_train"),
        ("navigation_long_horizon_train", "long_horizon_train"),
    ],
    "val": [
        ("navigation_base_val_from_train", "base_train"),
        ("navigation_common_val_from_train", "common_sense_train"),
        ("navigation_long_horizon_val_from_train", "long_horizon_train"),
    ],
    "source_val": [
        ("navigation_base_source_parity", "base"),
        ("navigation_common_source_parity", "common_sense"),
    ],
    "test": [
        ("navigation_base_test", "base"),
        ("navigation_common_test", "common_sense"),
        ("navigation_complex_instruction_test", "complex_instruction"),
        ("navigation_visual_appearance_test", "visual_appearance"),
        ("navigation_long_horizon_test", "long_horizon"),
    ],
}
rows = []
selected_specs = specs[split_name]
if bool(int(__import__("os").environ.get("ROLLOUT_SMOKE", "0"))):
    smoke_eval_set = __import__("os").environ.get("ROLLOUT_SMOKE_EVAL_SET", "base_train")
    selected_specs = [(f"navigation_{smoke_eval_set}_source_parity_smoke", smoke_eval_set)]
for data_source, eval_set in selected_specs:
    for seed in range(seed_start, seed_end + 1):
        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": ""}],
                "extra_info": {
                    "split": split_name,
                    "env_name": "navigation",
                    "env_config": {
                        "eval_set": eval_set,
                        "prompt_format": __import__("os").environ["ROLLOUT_PROMPT_FORMAT"],
                        "max_actions_per_step": 1,
                        "action_sep": __import__("os").environ["ROLLOUT_ACTION_SEP"],
                        "example_count": 0,
                        "step_length": float(__import__("os").environ["ROLLOUT_STEP_LENGTH"]),
                        "success_threshold": float(__import__("os").environ["ROLLOUT_SUCCESS_THRESHOLD"]),
                        "format_reward": float(__import__("os").environ["ROLLOUT_FORMAT_REWARD"]),
                        "per_turn_format_reward": float(__import__("os").environ["ROLLOUT_PER_TURN_FORMAT_REWARD"]),
                        "success_reward": float(__import__("os").environ["ROLLOUT_SUCCESS_REWARD"]),
                    },
                    "seed": seed,
                },
            }
        )
output.parent.mkdir(parents=True, exist_ok=True)
Dataset.from_list(rows).to_parquet(output)
print(f"wrote {len(rows)} deterministic {split_name} rows to {output}")
PY
}

run_shard() {
  local split_name=$1
  local shard_name=$2
  local seed_start=$3
  local seed_end=$4
  local split_dir="${RUN_DIR}/validation/${split_name}/${shard_name}"
  local out_jsonl="${split_dir}/${ROLLOUT_DUMP_STEP}.jsonl"
  mkdir -p "${split_dir}"
  if [ -s "${out_jsonl}" ]; then
    echo "[skip] ${split_name}/${shard_name} already done" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
    return 0
  fi
  local val_parquet="${CONTROL_DIR}/${split_name}_${shard_name}.parquet"
  local train_parquet="${val_parquet}"
  write_split_parquet "${split_name}" "${val_parquet}" "${seed_start}" "${seed_end}"
  if [ "${ROLLOUT_SMOKE}" = "1" ]; then
    # val-only still builds a drop-last train loader. Two deterministic rows
    # satisfy the 2-GPU batch invariant; only val_parquet is rolled out.
    train_parquet="${CONTROL_DIR}/${split_name}_${shard_name}_train_placeholder.parquet"
    write_split_parquet "${split_name}" "${train_parquet}" "${seed_start}" "$((seed_start + TRAIN_GPUS_PER_NODE - 1))"
  fi
  local service_base_url
  service_base_url=$(head -n 1 "${ENV_URL_FILE}")
  echo "=== array=${SLURM_ARRAY_TASK_ID} split=${split_name} shard=${shard_name} seeds=${seed_start}-${seed_end} greedy n=${VAL_N} ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  bash -lc "${COMMON_ENV}; export RAY_ADDRESS='${RAY_ADDRESS}'; PYTHONUNBUFFERED=1 python -m vagen.trainer.main_ppo \
      data.train_files='${train_parquet}' \
      data.val_files='${val_parquet}' \
      data.train_batch_size=${TRAIN_BATCH_SIZE} \
      data.val_batch_size=${VAL_BATCH_SIZE} \
      data.max_prompt_length=${ROLLOUT_MAX_PROMPT_LENGTH} \
      data.max_response_length=${ROLLOUT_DATA_MAX_RESPONSE_LENGTH} \
      data.max_trajectory_length=${ROLLOUT_MAX_TRAJECTORY_LENGTH} \
      data.truncation=left \
      algorithm.adv_estimator=reinforce_plus_plus \
      algorithm.kl_ctrl.kl_coef=0.0 \
      algorithm.gamma=1.0 \
      algorithm.lam=1.0 \
      actor_rollout_ref.model.path='${MODEL_PATH}' \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      actor_rollout_ref.model.use_remove_padding=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.kl_loss_coef=0.0 \
      actor_rollout_ref.actor.entropy_coeff=0.0 \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.do_sample=${ROLLOUT_DO_SAMPLE} \
      actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
      actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
      actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
      +actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
      +actor_rollout_ref.rollout.val_kwargs.do_sample=${ROLLOUT_DO_SAMPLE} \
      +actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_TEMPERATURE} \
      +actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_TOP_P} \
      +actor_rollout_ref.rollout.val_kwargs.top_k=${ROLLOUT_TOP_K} \
      actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
      actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
      actor_rollout_ref.rollout.enforce_eager=True \
      actor_rollout_ref.rollout.free_cache_engine=True \
      actor_rollout_ref.rollout.enable_chunked_prefill=True \
      actor_rollout_ref.rollout.max_response_per_turn=${ROLLOUT_MAX_TOKENS_PER_TURN} \
      actor_rollout_ref.rollout.disable_log_stats=False \
      +actor_rollout_ref.ref.use_ref=False \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      trainer.logger=['console'] \
      trainer.val_before_train=True \
      trainer.val_only=True \
      trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      trainer.project_name=nimloth_navigation \
      trainer.experiment_name=${EXPERIMENT_NAME}_a${SLURM_ARRAY_TASK_ID}_${split_name}_${shard_name} \
      trainer.default_local_dir='${RUN_DIR}/checkpoints_rollout' \
      trainer.validation_data_dir='${split_dir}' \
      trainer.val_generations_to_log_to_wandb=0 \
      trainer.total_epochs=1 \
      trainer.total_training_steps=1 \
      trainer.resume_mode=disable \
      rollout_manager.max_turns=${ROLLOUT_MAX_TURNS} \
      rollout_manager.max_trajectory_length=${ROLLOUT_MAX_TRAJECTORY_LENGTH} \
      rollout_manager.window_size=${ROLLOUT_WINDOW_SIZE} \
      rollout_manager.n_trajectory=1 \
      rollout_manager.use_service=True \
      rollout_manager.base_url='${service_base_url}' \
      rollout_manager.timeout=120 \
      rollout_manager.max_workers=${AGENT_NUM_WORKERS} \
      rollout_manager.use_multi_turn_reward=False \
      rollout_manager.use_loss_mask=True \
      rollout_manager.use_gae_mask=True \
      critic.model.path='${MODEL_PATH}'" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
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
  echo "SOURCE_CHECKPOINT_STEP=${CHECKPOINT_STEP} ROLLOUT_DUMP_STEP=${ROLLOUT_DUMP_STEP}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "ROLLOUT_PROTOCOL=${ROLLOUT_PROTOCOL} prompt=${ROLLOUT_PROMPT_FORMAT} turns=${ROLLOUT_MAX_TURNS} tokens_per_turn=${ROLLOUT_MAX_TOKENS_PER_TURN}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  echo "Eval-consistent: do_sample=${ROLLOUT_DO_SAMPLE} temperature=${ROLLOUT_TEMPERATURE} top_p=${ROLLOUT_TOP_P} top_k=${ROLLOUT_TOP_K} n=${VAL_N}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
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
  bash -lc "${COMMON_ENV}; ${RAY_CMD} stop --force >/dev/null 2>&1 || true; pkill -u \$(whoami) -f '[r]ay::' >/dev/null 2>&1 || true; sleep 2; ${RAY_CMD} start --head --node-ip-address=${local_ip} --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-gpus=${TRAIN_GPUS_PER_NODE} --num-cpus=${SLURM_CPUS_PER_TASK}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  sleep 20
  export RAY_ADDRESS="${local_ip}:${RAY_PORT}"
  bash -lc "${COMMON_ENV}; export RAY_ADDRESS='${RAY_ADDRESS}'; python -c \"import ray; ray.init(); print(ray.available_resources())\"" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

  if [ "${ROLLOUT_SMOKE}" = "1" ]; then
    if [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
      echo "ROLLOUT_SMOKE=1 only supports array task 0" >&2
      exit 2
    fi
    run_shard train "shard_smoke_seed_${ROLLOUT_SMOKE_SEED}" "${ROLLOUT_SMOKE_SEED}" "${ROLLOUT_SMOKE_SEED}"
    return 0
  fi
  if [ "${ROLLOUT_GATE_120}" = "1" ]; then
    if [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
      echo "ROLLOUT_GATE_120=1 only supports array task 0" >&2
      exit 2
    fi
    run_shard train shard_gate_001_040 1 40
    return 0
  fi
  if [ "${ROLLOUT_SOURCE_PARITY}" = "1" ]; then
    if [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
      echo "ROLLOUT_SOURCE_PARITY=1 only supports array task 0" >&2
      exit 2
    fi
    run_shard source_val shard_001_060 1 60
    return 0
  fi

  # Keep each VAGEN invocation at no more than 120 rows, matching the
  # accepted source-compatible quality replay. With max_workers=4 this caps
  # initial environment creation at 30 configs per manager worker and avoids
  # the 120-second create timeout seen with the former 540-row train shards.
  case "${SLURM_ARRAY_TASK_ID}" in
    0)
      for seed_start in $(seq 1 40 360); do
        seed_end=$((seed_start + 39))
        printf -v shard_name 'shard_%03d_%03d' "${seed_start}" "${seed_end}"
        run_shard train "${shard_name}" "${seed_start}" "${seed_end}"
      done
      ;;
    1)
      for seed_start in $(seq 361 40 720); do
        seed_end=$((seed_start + 39))
        printf -v shard_name 'shard_%03d_%03d' "${seed_start}" "${seed_end}"
        run_shard train "${shard_name}" "${seed_start}" "${seed_end}"
      done
      ;;
    2)
      for seed_start in $(seq 721 40 1080); do
        seed_end=$((seed_start + 39))
        printf -v shard_name 'shard_%03d_%03d' "${seed_start}" "${seed_end}"
        run_shard train "${shard_name}" "${seed_start}" "${seed_end}"
      done
      ;;
    3)
      for seed_start in 1081 1121 1161; do
        seed_end=$((seed_start + 39))
        printf -v shard_name 'shard_%03d_%03d' "${seed_start}" "${seed_end}"
        run_shard val "${shard_name}" "${seed_start}" "${seed_end}"
      done
      if [ "${ROLLOUT_INCLUDE_TEST}" = "1" ]; then
        run_shard test shard_001_024 1 24
        run_shard test shard_025_048 25 48
        run_shard test shard_049_060 49 60
      fi
      ;;
    *)
      echo "unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
      exit 2
      ;;
  esac
}

main "$@"
