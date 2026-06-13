#!/bin/bash
#SBATCH --job-name=vagen-nav-train-probe2x2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodelist=dgx-10,dgx-51
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --requeue

set -euo pipefail

echo "=== Training Probe Job starting at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

PROJECT_NAME="nimloth_navigation"
EXPERIMENT_NAME="vagen_nav_probe_2x2gpu_maxact1_turns20"
BASEDIR="/project/peilab/atst/nimloth/external/VAGEN"
SCRIPTDIR="/project/peilab/atst/nimloth/experiments/navigation_baseline"
RUN_DIR="${SCRIPTDIR}/runs/${EXPERIMENT_NAME}"
SAVE_CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
TOTAL_STEPS=2
GPUS_PER_NODE=2
TRAIN_BATCH_SIZE=16

ENV_HOSTFILE="${SCRIPTDIR}/env_server_probe_host.txt"
if [ ! -f "$ENV_HOSTFILE" ]; then
  echo "ERROR: probe env host file not found: $ENV_HOSTFILE"
  echo "Start env server first: sbatch ${SCRIPTDIR}/env_server_probe_1gpu.slurm"
  exit 1
fi
awk 'NF {print "Env server: http://" $0}' "$ENV_HOSTFILE"

TMPDIR=$(mktemp -d)
ENV_URL_FILE="${TMPDIR}/env_urls.txt"
awk 'NF {print "http://" $0}' "$ENV_HOSTFILE" > "$ENV_URL_FILE"
cp "${SCRIPTDIR}/train.yaml" "${TMPDIR}/train.yaml"
cp "${SCRIPTDIR}/val.yaml" "${TMPDIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMPDIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMPDIR}/val.yaml"
mkdir -p "$RUN_DIR"

export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv
export UV_PYTHON_INSTALL_DIR=/project/peilab/atst/nimloth/.local/python
export XDG_CACHE_HOME=/project/peilab/atst/nimloth/.cache
export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/vagen-nav-train-probe-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi
source /project/peilab/atst/nimloth/.venv/bin/activate

export TOKENIZERS_PARALLELISM=true
export RAY_DEDUP_LOGS=0
cd "$BASEDIR"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
NUM_NODES=${#NODES[@]}
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "Total nodes: $NUM_NODES"

ray start --head \
  --port=${RAY_PORT} \
  --num-cpus=32 \
  --num-gpus=${GPUS_PER_NODE} \
  --node-ip-address="${HEAD_IP}" \
  --include-dashboard=false
sleep 15

if [ "$NUM_NODES" -gt 1 ]; then
  WORKER_NODES=$(IFS=,; echo "${NODES[*]:1}")
  srun --nodes=$((NUM_NODES - 1)) --ntasks-per-node=1 --nodelist="$WORKER_NODES" \
    ray start --address="${HEAD_IP}:${RAY_PORT}" \
      --num-cpus=32 \
      --num-gpus=${GPUS_PER_NODE} \
      &
  sleep 15
fi

set +e
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
  --config-path="${BASEDIR}/vagen/configs" \
  --config-name='vagen_multiturn' \
  data.train_files="${TMPDIR}/train.yaml" \
  data.val_files="${TMPDIR}/val.yaml" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=4096 \
  data.max_response_length=16384 \
  algorithm.adv_estimator=gae \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
  actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  trainer.nnodes=${NUM_NODES} \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.max_critic_ckpt_to_keep=3 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
  trainer.validation_data_dir=${RUN_DIR}/validation \
  trainer.rollout_data_dir=${RUN_DIR}/rollout_data \
  trainer.log_val_generations=4 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.resume_mode=auto \
  critic.optim.lr=1e-5 \
  critic.model.use_remove_padding=True \
  critic.model.path=${MODEL_PATH} \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  huggingface_hub.hf_save_freq=null \
  +ray_kwargs.ray_init.address=auto \
  2>&1 | tee "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

ray stop --force || true
exit ${TRAIN_EXIT}
