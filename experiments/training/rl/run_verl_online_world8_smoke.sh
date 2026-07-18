#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT}"
: "${MODEL:?set MODEL}"
: "${ENV_URL:?set ENV_URL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${WANDB_PROJECT:?set WANDB_PROJECT}"
: "${WANDB_RUN_NAME:?set WANDB_RUN_NAME}"
: "${WANDB_RUN_ID:?set WANDB_RUN_ID}"

PY=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
export PYTHONDONTWRITEBYTECODE=1
[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${REPO}" status --porcelain)" ]]
[[ "$(git -C "${REPO}/external/VAGEN" rev-parse HEAD)" == \
   "${EXPECTED_VAGEN_COMMIT:?set EXPECTED_VAGEN_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/VAGEN/verl" rev-parse HEAD)" == \
   "${EXPECTED_VERL_COMMIT:?set EXPECTED_VERL_COMMIT}" ]]
[[ -d "${MODEL}" ]]
curl --fail --silent "${ENV_URL}/health" >/dev/null

mkdir -p "${OUTPUT_DIR}/data" "${OUTPUT_DIR}/train_records" "${OUTPUT_DIR}/checkpoints"
"${PY}" "${REPO}/experiments/training/rl/build_verl_online_smoke_dataset.py" \
  --output-dir "${OUTPUT_DIR}/data" --seed 30002 \
  >"${OUTPUT_DIR}/dataset_build.log" 2>&1

set -a
source /project/peilab/atst/flower/.env
set +a
# The secret file provides credentials only; experiment identity is immutable.
export WANDB_PROJECT WANDB_RUN_NAME WANDB_RUN_ID
export WANDB_RESUME=never
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm:${PYTHONPATH:-}"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export HOME=/project/peilab/atst/nimloth/.home
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=XFORMERS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export PYTHONHASHSEED=0
unset PYTORCH_CUDA_ALLOC_CONF

cd "${REPO}/external/VAGEN"
"${PY}" -m vagen.trainer.main_ppo \
  algorithm.adv_estimator=masked_gae \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  data.train_files="${OUTPUT_DIR}/data/train.parquet" \
  data.val_files="${OUTPUT_DIR}/data/val.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=8191 \
  data.max_trajectory_length=8192 \
  data.truncation=error \
  data.shuffle=false \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.external_lib=nimloth.training.rl.verl_runtime_patch \
  +actor_rollout_ref.model.override_config.use_cache=false \
  +actor_rollout_ref.model.override_config.tie_word_embeddings=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size=null \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=false \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.grad_norm_threshold=null \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.nimloth_parameter_audit=true \
  actor_rollout_ref.actor.nimloth_wm_aux.enabled=true \
  actor_rollout_ref.actor.nimloth_wm_aux.checkpoint_dir=null \
  actor_rollout_ref.actor.nimloth_wm_aux.allow_random_init=true \
  actor_rollout_ref.actor.nimloth_wm_aux.latent_token_count=8 \
  actor_rollout_ref.actor.nimloth_wm_aux.latent_query_mode=inject \
  actor_rollout_ref.actor.nimloth_wm_aux.loss_coef=0.1 \
  actor_rollout_ref.actor.nimloth_wm_aux.lr=1e-5 \
  actor_rollout_ref.ref.use_ref=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=null \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.nimloth_parameter_audit=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.limit_mm_per_prompt=3 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=1 \
  critic.model.path="${MODEL}" \
  critic.model.tokenizer_path="${MODEL}" \
  critic.model.external_lib=nimloth.training.rl.verl_runtime_patch \
  +critic.model.override_config.use_cache=false \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.use_remove_padding=false \
  +critic.model.fsdp_config.model_dtype=fp32 \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.optimizer_offload=true \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size=8 \
  critic.ppo_micro_batch_size=null \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.forward_micro_batch_size=null \
  critic.forward_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=8192 \
  critic.forward_max_token_len_per_gpu=8192 \
  critic.ppo_epochs=1 \
  critic.shuffle=false \
  critic.nimloth_parameter_audit=true \
  rollout_manager.service_manager_class=nimloth.training.rl.vagen_online_rollout.NimlothQwenVLRolloutManagerService \
  rollout_manager.use_service=true \
  rollout_manager.base_url="${ENV_URL}" \
  rollout_manager.timeout=1200 \
  rollout_manager.latent_token_count=8 \
  rollout_manager.latent_query_mode=inject \
  rollout_manager.max_think_tokens=512 \
  rollout_manager.max_turns=2 \
  rollout_manager.window_size=5 \
  rollout_manager.n_trajectory=8 \
  rollout_manager.mini_batch_size=8 \
  rollout_manager.use_multi_turn_reward=true \
  rollout_manager.use_loss_mask=true \
  rollout_manager.use_gae_mask=true \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_RUN_NAME}" \
  'trainer.logger=[console,wandb]' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.critic_warmup=0 \
  trainer.val_before_train=false \
  trainer.disable_validation=true \
  trainer.test_freq=-1 \
  trainer.save_freq=1 \
  trainer.resume_mode=disable \
  trainer.del_local_ckpt_after_load=false \
  trainer.remove_previous_ckpt_in_save=false \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.train_data_dir="${OUTPUT_DIR}/train_records" \
  trainer.nimloth_online_update_audit=true \
  2>&1 | tee "${OUTPUT_DIR}/trainer.log"
