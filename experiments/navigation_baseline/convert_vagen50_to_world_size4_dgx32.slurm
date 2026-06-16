#!/bin/bash
#SBATCH --job-name=vagen50-ws4-convert
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodelist=dgx-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:4
#SBATCH --mem=900G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
CONVERT_PY=${ROOT}/experiments/navigation_baseline/convert_vagen_actor_only_to_world_size.py
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
SOURCE_CHECKPOINT_STEP=50
SOURCE_HF_PATH=${BASE_RUN_DIR}/checkpoints/global_step_${SOURCE_CHECKPOINT_STEP}/actor/huggingface
EXPERIMENT_NAME=vagen50_world_size4_convert
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CHECKPOINT_OUT_DIR=${BASE_RUN_DIR}/checkpoints
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
RAY_PORT=$((28000 + (${SLURM_JOB_ID:-0} % 1000)))
TARGET_WORLD_SIZE=4

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export PYTHONPATH=/project/peilab/atst/nimloth:/project/peilab/atst/nimloth/external/VAGEN:/project/peilab/atst/nimloth/external/VAGEN/verl:${PYTHONPATH:-}; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0; export HYDRA_FULL_ERROR=1; export TORCHINDUCTOR_DISABLE=1; export TORCH_COMPILE_DISABLE=1; export TORCHDYNAMO_DISABLE=1; export NCCL_IB_DISABLE=1'

cleanup() {
  set +e
  $SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
}
trap cleanup EXIT

cat > "${CONTROL_DIR}/dummy.yaml" <<YAML
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

if [ -f "${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ]; then
  echo "ws${TARGET_WORLD_SIZE} checkpoint already exists; skip"
  exit 0
fi

HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[ -z "${HEAD_IP}" ] && HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 hostname -I | awk '{print $1}')

$SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true" || true
$SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=${RAY_PORT} --num-cpus=96 --num-gpus=${TARGET_WORLD_SIZE} --node-ip-address='${HEAD_IP}' --include-dashboard=false --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}.log" 2>&1 &
sleep 20

$SL/srun --overlap --nodes=1 --ntasks=1 -w dgx-32 bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  PYTHONUNBUFFERED=1 python3 ${CONVERT_PY} \
    --config-path='${BASEDIR}/vagen/configs' --config-name='vagen_multiturn' \
    hydra.searchpath=[file:///project/peilab/atst/nimloth/external/VAGEN/verl/verl/trainer/config] \
    data.train_files='${CONTROL_DIR}/dummy.yaml' data.val_files='${CONTROL_DIR}/dummy.yaml' \
    data.train_batch_size=${TARGET_WORLD_SIZE} data.val_batch_size=${TARGET_WORLD_SIZE} \
    data.max_prompt_length=3000 data.max_response_length=20000 \
    algorithm.adv_estimator=gae algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path='${SOURCE_HF_PATH}' \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=${TARGET_WORLD_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False +actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='${BASEDIR}/vagen/configs/agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=1 actor_rollout_ref.rollout.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False +actor_rollout_ref.ref.fsdp_config.sync_module_states=False \
    trainer.logger=['console'] trainer.val_before_train=False trainer.val_only=False \
    trainer.n_gpus_per_node=${TARGET_WORLD_SIZE} trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=0 \
    trainer.project_name=nimloth_navigation trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir='${CHECKPOINT_OUT_DIR}' trainer.total_training_steps=1 trainer.resume_mode=disable \
    +trainer.actor_only_convert=True +trainer.convert_global_step=${SOURCE_CHECKPOINT_STEP} \
    critic.optim.lr=1e-5 critic.model.use_remove_padding=True critic.model.path='${SOURCE_HF_PATH}' \
    critic.model.enable_gradient_checkpointing=True critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_max_token_len_per_gpu=32768 critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True critic.model.fsdp_config.use_torch_compile=False \
    +critic.model.fsdp_config.sync_module_states=False huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto
" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

test -f "${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt"
test -f "${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/critic/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt"
echo "=== ws${TARGET_WORLD_SIZE} conversion done ==="
