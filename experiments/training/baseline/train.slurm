#!/bin/bash
#SBATCH --job-name=vagen-nav-train
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --requeue

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
CONFIG_DIR=${REPO}/configs/training/baseline
BASEDIR=${REPO}/external/VAGEN

PROJECT_NAME=${PROJECT_NAME:-nimloth_navigation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline_maxact1_turns20}
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
TOTAL_STEPS=${TOTAL_STEPS:-50}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
RUNTIME_DIR=${RUNTIME_DIR:-${REPO}/outputs/experiments/training/baseline/runtime}

mkdir -p "${RUN_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

echo "=== Training Job starting at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "RUN_DIR: ${RUN_DIR}"

ENV_HOSTFILE="${RUNTIME_DIR}/env_server_host.txt"
if [ -f "$ENV_HOSTFILE" ]; then
    ENV_COUNT=$(awk 'NF {n++} END {print n+0}' "$ENV_HOSTFILE")
    if [ "$ENV_COUNT" -lt 2 ]; then
        echo "ERROR: Expected at least 2 env servers, found ${ENV_COUNT} in $ENV_HOSTFILE"
        echo "Start env servers with: bash ${SCRIPTDIR}/launch_env_servers.sh"
        exit 1
    fi
    echo "Env servers from ${ENV_HOSTFILE}:"
    awk 'NF {print "  http://" $0}' "$ENV_HOSTFILE"
else
    echo "ERROR: Env server host file not found: $ENV_HOSTFILE"
    echo "Start env servers with: bash ${SCRIPTDIR}/launch_env_servers.sh"
    exit 1
fi

TMPDIR_CFG=$(mktemp -d)
ENV_URL_FILE="${TMPDIR_CFG}/env_urls.txt"
awk 'NF {print "http://" $0}' "$ENV_HOSTFILE" > "$ENV_URL_FILE"
cp "${CONFIG_DIR}/train.yaml" "${TMPDIR_CFG}/train.yaml"
cp "${CONFIG_DIR}/val.yaml" "${TMPDIR_CFG}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMPDIR_CFG}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMPDIR_CFG}/val.yaml"

export TMPDIR=/tmp/vagen-nav-train-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR_CFG" "$TMPDIR"' EXIT

# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
cd "${BASEDIR}"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
NUM_NODES=${#NODES[@]}
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "Total nodes: $NUM_NODES"

ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=64 \
    --num-gpus=8 \
    --node-ip-address="${HEAD_IP}" \
    --include-dashboard=false

sleep 15

if [ "$NUM_NODES" -gt 1 ]; then
    WORKER_NODES=$(IFS=,; echo "${NODES[*]:1}")
    echo "=== Starting Ray workers on: $WORKER_NODES ==="
    srun --nodes=$((NUM_NODES - 1)) --ntasks-per-node=1 --nodelist="$WORKER_NODES" \
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
            --num-cpus=64 \
            --num-gpus=8 &
    sleep 15
fi

echo "=== Ray cluster ready, starting VAGEN training ==="

set +e
PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="${BASEDIR}/vagen/configs" \
    --config-name='vagen_multiturn' \
    data.train_files="${TMPDIR_CFG}/train.yaml" \
    data.val_files="${TMPDIR_CFG}/val.yaml" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=50 \
    trainer.max_critic_ckpt_to_keep=50 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${RUN_DIR}/validation \
    trainer.rollout_data_dir=${RUN_DIR}/rollout_data \
    trainer.log_val_generations=32 \
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

python3 "${SCRIPTDIR}/prune_checkpoints.py" \
    --checkpoint-dir "${SAVE_CHECKPOINT_DIR}" \
    --validation-dir "${RUN_DIR}/validation" \
    --keep-last 10 \
    2>&1 | tee -a "${RUN_DIR}/${EXPERIMENT_NAME}.log" || true

for node in "${NODES[@]}"; do
    srun --nodes=1 --ntasks=1 --nodelist="$node" ray stop 2>/dev/null || true
done

echo "=== Training finished with exit code ${TRAIN_EXIT} at $(date) ==="
exit $TRAIN_EXIT
