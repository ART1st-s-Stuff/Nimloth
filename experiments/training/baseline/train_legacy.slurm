#!/bin/bash
#SBATCH --job-name=vagen-train-legacy
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --mem=256G
#SBATCH --requeue

set -euo pipefail

REPO=/project/peilab/atst/nimloth
SCRIPTDIR=${REPO}/experiments/training/baseline
BASEDIR=${REPO}/external/VAGEN
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-vagen_nav_baseline}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
RUNTIME_DIR=${RUNTIME_DIR:-${REPO}/outputs/experiments/training/baseline/runtime}

# Hyperparams (matching VAGEN paper Table 12, 19, 23 + task spec)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
TOTAL_STEPS=${TOTAL_STEPS:-60}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
MAX_TURNS=${MAX_TURNS:-4}
MAX_ACTIONS_PER_STEP=${MAX_ACTIONS_PER_STEP:-1}
PROMPT_FORMAT=${PROMPT_FORMAT:-wm}
PROJECT_NAME=${PROJECT_NAME:-nimloth_navigation}

mkdir -p "${RUN_DIR}" "${RUNTIME_DIR}" "${REPO}/outputs/experiments/training/baseline/slurm"

echo "=== Training Job starting at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "RUN_DIR: ${RUN_DIR}"

# ---------------------------------------------------------------------------
# Wait for env server
# ---------------------------------------------------------------------------
ENV_HOSTFILE="${RUNTIME_DIR}/env_legacy_host.txt"
MAX_WAIT=300
WAITED=0
while [ ! -f "$ENV_HOSTFILE" ]; do
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: env server host file not found after ${MAX_WAIT}s: $ENV_HOSTFILE"
    exit 1
  fi
  echo "Waiting for env server... (${WAITED}s)"
  sleep 5
  WAITED=$((WAITED + 5))
done

ENV_URL=$(cat "$ENV_HOSTFILE")
echo "Env server at http://${ENV_URL}"

# Wait for server to be healthy
for i in $(seq 1 60); do
  if curl -fsS "http://${ENV_URL}/health" >/dev/null 2>&1; then
    echo "Env server healthy (attempt $i)"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: env server not healthy after 60 attempts"
    exit 1
  fi
  sleep 3
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
export TMPDIR=/tmp/vagen-train-legacy-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

export UV_CACHE_DIR="${REPO}/.cache/uv"
export XDG_CACHE_HOME="${REPO}/.cache"
export HOME="${REPO}/.home"
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export FLASHINFER_WORKSPACE_DIR="${REPO}/.cache/flashinfer"
export WANDB_DIR="${REPO}/.cache/wandb"
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}_$(hostname)"
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_${SLURM_JOB_ID}_$(hostname)"
export PYTHONPATH="${REPO}/src:${BASEDIR}:${BASEDIR}/verl:${PYTHONPATH:-}"
export PATH="${REPO}/.local/bin:$PATH"

mkdir -p "$HOME" "$FLASHINFER_WORKSPACE_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=true
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker0,virbr0"

cd "${BASEDIR}"

# ---------------------------------------------------------------------------
# Create dataset via env server
# ---------------------------------------------------------------------------
echo "=== Creating dataset at $(date) ==="

# Generate env_config.yaml for create_dataset
DATASET_CONFIG="${RUN_DIR}/env_config.yaml"
cat > "${DATASET_CONFIG}" << EOF
env1:
    env_name: navigation
    env_config:
        render_mode: vision
        prompt_format: ${PROMPT_FORMAT}
        use_state_reward: false
        eval_set: base
        max_actions_per_step: ${MAX_ACTIONS_PER_STEP}
        max_action_penalty: -0.1
        format_reward: 0.5
        success_threshold: 1.5
        step_length: 0.5
        grounding_reward_weight: 0.5
        worldmodeling_reward_weight: 0.5
        gpu_device: 0
    train_size: 1200
    test_size: 60

env2:
    env_name: navigation
    env_config:
        render_mode: vision
        prompt_format: ${PROMPT_FORMAT}
        use_state_reward: false
        eval_set: common_sense
        max_actions_per_step: ${MAX_ACTIONS_PER_STEP}
        max_action_penalty: -0.1
        format_reward: 0.5
        success_threshold: 1.5
        step_length: 0.5
        grounding_reward_weight: 0.5
        worldmodeling_reward_weight: 0.5
        gpu_device: 0
    train_size: 1200
    test_size: 60
EOF

python3 -m vagen.env.create_dataset \
  --yaml_path "${DATASET_CONFIG}" \
  --train_path "${RUN_DIR}/train.parquet" \
  --test_path "${RUN_DIR}/test.parquet"

echo "=== Dataset created at $(date) ==="

# ---------------------------------------------------------------------------
# Ray cluster setup
# ---------------------------------------------------------------------------
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
NUM_NODES=${#NODES[@]}
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "Total nodes: $NUM_NODES"

# Stop any leftover ray
for node in "${NODES[@]}"; do
  srun --nodes=1 --ntasks=1 --nodelist="$node" ray stop 2>/dev/null || true
done
sleep 3

ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=48 \
    --num-gpus=6 \
    --node-ip-address="${HEAD_IP}" \
    --include-dashboard=false

sleep 15

if [ "$NUM_NODES" -gt 1 ]; then
    WORKER_NODES=$(IFS=,; echo "${NODES[*]:1}")
    echo "=== Starting Ray workers on: $WORKER_NODES ==="
    srun --nodes=$((NUM_NODES - 1)) --ntasks-per-node=1 --nodelist="$WORKER_NODES" \
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
            --num-cpus=48 \
            --num-gpus=6 &
    sleep 15
fi

echo "=== Ray cluster ready, starting VAGEN training ==="

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
set +e
PYTHONUNBUFFERED=1 python3 -m vagen.trainer.main_ppo \
  algorithm.adv_estimator=masked_gae \
  algorithm.high_level_gamma=1.0 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${RUN_DIR}/train.parquet" \
  data.val_files="${RUN_DIR}/test.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length=9000 \
  data.max_response_length=8000 \
  data.max_trajectory_length=20000 \
  data.image_key=images \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.05 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.limit_mm_per_prompt=20 \
  actor_rollout_ref.rollout.max_response_per_turn=2048 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=1e-5 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=6 \
  trainer.nnodes="${NUM_NODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.max_critic_ckpt_to_keep=null \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
  trainer.validation_data_dir="${RUN_DIR}/validation" \
  trainer.rollout_data_dir="${RUN_DIR}/rollout_data" \
  trainer.log_val_generations=32 \
  trainer.resume_mode=auto \
  rollout_manager.max_turns="${MAX_TURNS}" \
  rollout_manager.window_size="${MAX_TURNS}" \
  rollout_manager.use_multi_turn_reward=False \
  rollout_manager.use_loss_mask=True \
  rollout_manager.use_gae_mask=True \
  rollout_manager.n_trajectory=1 \
  rollout_manager.use_service=True \
  rollout_manager.max_workers=48 \
  rollout_manager.timeout=1200 \
  rollout_manager.base_url="http://${ENV_URL}" \
  +ray_kwargs.ray_init.address=auto \
  2>&1 | tee "${RUN_DIR}/${EXPERIMENT_NAME}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
for node in "${NODES[@]}"; do
  srun --nodes=1 --ntasks=1 --nodelist="$node" ray stop 2>/dev/null || true
done

echo "=== Training finished with exit code ${TRAIN_EXIT} at $(date) ==="
exit $TRAIN_EXIT
