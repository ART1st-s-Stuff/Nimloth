#!/bin/bash
#SBATCH --job-name=vagen50-ws12-convert
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
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
EXPERIMENT_NAME=vagen50_world_size12_convert
RUN_DIR=${SCRIPTDIR}/runs/${EXPERIMENT_NAME}
CHECKPOINT_OUT_DIR=${BASE_RUN_DIR}/checkpoints
CONTROL_DIR=${RUN_DIR}/slurm_${SLURM_JOB_ID}_control
RAY_PORT=$((24000 + (${SLURM_JOB_ID:-0} % 1000)))
TRAIN_GPUS_PER_NODE=4
TRAIN_NODES=3
TARGET_WORLD_SIZE=12

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${CHECKPOINT_OUT_DIR}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python; export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache; export HOME=/project/peilab/atst/nimloth/.home; export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer; mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR"; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; cd /project/peilab/atst/nimloth/external/VAGEN; export PYTHONPATH=/project/peilab/atst/nimloth:/project/peilab/atst/nimloth/external/VAGEN:/project/peilab/atst/nimloth/external/VAGEN/verl:${PYTHONPATH:-}; export TOKENIZERS_PARALLELISM=true; export RAY_DEDUP_LOGS=0; export HYDRA_FULL_ERROR=1; export TORCHINDUCTOR_DISABLE=1; export TORCH_COMPILE_DISABLE=1; export TORCHDYNAMO_DISABLE=1; export TORCHINDUCTOR_COMPILE_THREADS=1; export NCCL_IB_DISABLE=1; export NCCL_SOCKET_IFNAME="^lo,docker0,virbr0"'

cleanup() {
  set +e
  echo "=== cleanup at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null || true))
  for node in "${NODES[@]:-}"; do
    $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc 'ray stop --force >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

cat > "${CONTROL_DIR}/dummy.yaml" <<YAML
# Minimal dataset only to satisfy VAGEN trainer construction during checkpoint conversion.
envs:
  - name: RemoteEnv
    n_envs: 12
    data_source: navigation_base_train
    seed: [1, 12, 1]
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

{
  echo "=== VAGEN global_step_${SOURCE_CHECKPOINT_STEP} world_size=${TARGET_WORLD_SIZE} conversion starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Nodes: ${SLURM_JOB_NODELIST}; ${TRAIN_NODES} nodes x ${TRAIN_GPUS_PER_NODE} GPU"
  echo "Source HF checkpoint: ${SOURCE_HF_PATH}"
  echo "Target checkpoint dir: ${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}"
  echo "Purpose: actor+critic checkpoint conversion only; actor_only_convert skips SGLang rollout."
} | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -d "${SOURCE_HF_PATH}" ]; then
  echo "ERROR missing source HF checkpoint ${SOURCE_HF_PATH}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 3
fi
if [ -f "${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ]; then
  echo "Target world_size=${TARGET_WORLD_SIZE} checkpoint already exists; skipping conversion." | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 0
fi

NODES=($($SL/scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "${HEAD_IP}" ]; then HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}'); fi
IP_HEAD=${HEAD_IP}:${RAY_PORT}
echo "Head node: ${HEAD_NODE} ${HEAD_IP}" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

for node in "${NODES[@]}"; do
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "${COMMON_ENV}; ray stop --force >/dev/null 2>&1 || true; pkill -u \$USER -f 'sglang|SGLangHttpServer|torch/_inductor/compile_worker' >/dev/null 2>&1 || true" || true
done
sleep 10

$SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=96 \
    --num-gpus=${TRAIN_GPUS_PER_NODE} \
    --node-ip-address='${HEAD_IP}' \
    --include-dashboard=false \
    --block
" > "${RUN_DIR}/ray_head_${SLURM_JOB_ID}_${HEAD_NODE}.log" 2>&1 &
sleep 20

for worker_node in "${NODES[@]:1}"; do
  $SL/srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --address='${IP_HEAD}' \
      --num-cpus=96 \
      --num-gpus=${TRAIN_GPUS_PER_NODE} \
      --block
  " > "${RUN_DIR}/ray_worker_${SLURM_JOB_ID}_${worker_node}.log" 2>&1 &
done
sleep 25

$SL/srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  set -euo pipefail
  ${COMMON_ENV}
  PYTHONUNBUFFERED=1 python3 ${CONVERT_PY} \
    --config-path='${BASEDIR}/vagen/configs' \
    --config-name='vagen_multiturn' \
    hydra.searchpath=[file:///project/peilab/atst/nimloth/external/VAGEN/verl/verl/trainer/config] \
    data.train_files='${CONTROL_DIR}/dummy.yaml' \
    data.val_files='${CONTROL_DIR}/dummy.yaml' \
    data.train_batch_size=${TARGET_WORLD_SIZE} \
    data.val_batch_size=${TARGET_WORLD_SIZE} \
    data.max_prompt_length=3000 \
    data.max_response_length=20000 \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path='${SOURCE_HF_PATH}' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TARGET_WORLD_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    +actor_rollout_ref.actor.fsdp_config.sync_module_states=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='${BASEDIR}/vagen/configs/agent.yaml' \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    +actor_rollout_ref.ref.fsdp_config.sync_module_states=False \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE} \
    trainer.nnodes=${TRAIN_NODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=0 \
    trainer.max_actor_ckpt_to_keep=50 \
    trainer.max_critic_ckpt_to_keep=50 \
    trainer.project_name=nimloth_navigation \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir='${CHECKPOINT_OUT_DIR}' \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    +trainer.actor_only_convert=True \
    +trainer.convert_global_step=${SOURCE_CHECKPOINT_STEP} \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path='${SOURCE_HF_PATH}' \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.use_torch_compile=False \
    +critic.model.fsdp_config.sync_module_states=False \
    huggingface_hub.hf_save_freq=null \
    +ray_kwargs.ray_init.address=auto
" 2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"

if [ ! -f "${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/actor/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ]; then
  echo "ERROR conversion finished without expected rank $((TARGET_WORLD_SIZE - 1)) shard" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 5
fi
if [ ! -f "${CHECKPOINT_OUT_DIR}/global_step_${SOURCE_CHECKPOINT_STEP}/critic/model_world_size_${TARGET_WORLD_SIZE}_rank_$((TARGET_WORLD_SIZE - 1)).pt" ]; then
  echo "ERROR conversion finished without expected critic rank $((TARGET_WORLD_SIZE - 1)) shard" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
  exit 5
fi
echo "${SOURCE_CHECKPOINT_STEP}" > "${CHECKPOINT_OUT_DIR}/latest_checkpointed_iteration.txt"
cat > "${CHECKPOINT_OUT_DIR}/conversion_manifest_ws${TARGET_WORLD_SIZE}.txt" <<EOF_MANIFEST
source_hf=${SOURCE_HF_PATH}
source_global_step=${SOURCE_CHECKPOINT_STEP}
target_world_size=${TARGET_WORLD_SIZE}
created_by_job=${SLURM_JOB_ID}
created_at=$(date)
EOF_MANIFEST

echo "=== Conversion finished at $(date) ===" | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log"
