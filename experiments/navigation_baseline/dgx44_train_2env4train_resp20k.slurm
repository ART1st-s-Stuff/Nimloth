#!/bin/bash
#SBATCH --job-name=vagen-dgx44-train-2env4train
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-44
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:6
#SBATCH --mem=512G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
RUN_DIR=${SCRIPTDIR}/runs/vagen_nav_dgx44_2env4train_original_base_common_sticky_semaphore_resp20k
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
ENV_PORT=8000
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TOTAL_STEPS=50
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
VAL_BATCH_SIZE=24
AGENT_NUM_WORKERS=4
AGENT_MAX_CONCURRENT_TRAJECTORIES=6
ENV_GPUS=2
TRAIN_GPUS=4

echo "=== DGX-44 opportunistic training starting at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
mkdir -p "$RUN_DIR"

export UV_CACHE_DIR=${ROOT}/.cache/uv
export UV_PYTHON_INSTALL_DIR=${ROOT}/.local/python
export XDG_CACHE_HOME=${ROOT}/.cache
export PATH=${ROOT}/.venv/bin:${ROOT}/.local/bin:$PATH
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TMPDIR=/tmp/vagen-dgx20-train-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
ENV_PIDS=()
cleanup() {
  set +e
  if ((${#ENV_PIDS[@]} > 0)); then
    for pid in "${ENV_PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${ENV_PIDS[@]}"; do
      wait "$pid" >/dev/null 2>&1 || true
    done
  fi
  ray stop --force >/dev/null 2>&1 || true
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [ -f /project/peilab/atst/flower/.env ]; then
  set -a; source /project/peilab/atst/flower/.env; set +a
elif [ -f /project/peilab/atst/.env ]; then
  set -a; source /project/peilab/atst/.env; set +a
fi
source ${ROOT}/.venv/bin/activate
source ${SCRIPTDIR}/setup_ai2thor_env.sh
cd "$BASEDIR"

IFS=',' read -r -a ALLOC_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
ALL_CUDA=$(IFS=,; echo "${ALLOC_GPU_IDS[*]}")
echo "Allocated physical GPU ids: ${ALLOC_GPU_IDS[*]}"
echo "All allocated CUDA_VISIBLE_DEVICES for env probing: ${ALL_CUDA}"

# Probe allocated GPUs for AI2-THOR and require 2 env GPUs + 4 training GPUs.
# If fewer than 3 GPUs can run AI2-THOR on dgx-44, fail explicitly; do not shrink or downgrade.
rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
GOOD_ORD=()
for idx in "${!ALLOC_GPU_IDS[@]}"; do
  phys="${ALLOC_GPU_IDS[$idx]}"
  phys=$(echo "$phys" | xargs)
  [ -z "$phys" ] && continue
  echo "=== Testing AI2-THOR on allocated ordinal ${idx} physical GPU ${phys} at $(date) ==="
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$phys" timeout 120s python - <<'PY'
import time
import ai2thor.controller
from ai2thor.platform import CloudRendering
print('create', flush=True)
t0=time.time()
c=ai2thor.controller.Controller(
    agentMode='default', gridSize=0.1, visibilityDistance=10,
    renderDepthImage=False, renderInstanceSegmentation=False,
    width=255, height=255, fieldOfView=100,
    platform=CloudRendering, gpu_device=0,
    server_timeout=60, server_start_timeout=60,
)
print(f'created {time.time()-t0:.1f}s', flush=True)
try:
    t1=time.time(); ev=c.reset(scene='FloorPlan1')
    print(f'reset {time.time()-t1:.1f}s frame={None if ev.frame is None else ev.frame.shape}', flush=True)
finally:
    c.stop()
print('OK', flush=True)
PY
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "Allocated ordinal ${idx} physical GPU ${phys}: AI2-THOR OK"
    GOOD_ORD+=("$idx")
  else
    echo "Allocated ordinal ${idx} physical GPU ${phys}: AI2-THOR FAILED/TIMEOUT rc=${rc}"
  fi
done

echo "AI2-THOR-good allocated ordinals: ${GOOD_ORD[*]:-NONE}"
if [ "${#GOOD_ORD[@]}" -lt "$ENV_GPUS" ]; then
  echo "ERROR: need ${ENV_GPUS} AI2-THOR-good GPUs for 2env4train; got ${#GOOD_ORD[@]}. Not downgrading."
  exit 2
fi

ENV_ORD=("${GOOD_ORD[@]:0:${ENV_GPUS}}")
TRAIN_ORD=()
for idx in "${!ALLOC_GPU_IDS[@]}"; do
  skip=0
  for env_idx in "${ENV_ORD[@]}"; do [ "$idx" = "$env_idx" ] && skip=1; done
  [ "$skip" -eq 0 ] && TRAIN_ORD+=("$idx")
done
if [ "${#TRAIN_ORD[@]}" -lt "$TRAIN_GPUS" ]; then
  echo "ERROR: need ${TRAIN_GPUS} training GPUs after reserving ${ENV_GPUS} env GPUs; got ${#TRAIN_ORD[@]}"
  exit 3
fi
TRAIN_ORD=("${TRAIN_ORD[@]:0:${TRAIN_GPUS}}")
ENV_DEVICES=$(python - <<PY
ords="${ENV_ORD[*]}".split()
print('[' + ', '.join(ords) + ']')
PY
)
TRAIN_CUDA=$(python - <<PY
alloc="${ALLOC_GPU_IDS[*]}".split()
ords=[int(x) for x in "${TRAIN_ORD[*]}".split()]
print(','.join(alloc[i] for i in ords))
PY
)
echo "Env allocated ordinals: ${ENV_ORD[*]} devices_arg=${ENV_DEVICES} visible=${ALL_CUDA}"
echo "Train allocated ordinals: ${TRAIN_ORD[*]} physical CUDA_VISIBLE_DEVICES=${TRAIN_CUDA} TRAIN_GPUS=${TRAIN_GPUS}"

HOSTFILE=${SCRIPTDIR}/env_server_dgx20_probe_host.txt
rm -f "$HOSTFILE" "${HOSTFILE}.lock"
: > "$HOSTFILE"
ENV_URLS=()

for env_i in "${!ENV_ORD[@]}"; do
  ord="${ENV_ORD[$env_i]}"
  phys="${ALLOC_GPU_IDS[$ord]}"
  port=$((ENV_PORT + env_i))
  url="http://$(hostname):${port}"
  ENV_URLS+=("$url")
  echo "$(hostname):${port}" >> "$HOSTFILE"

  echo "=== Starting env server ${env_i} on allocated ordinal ${ord} physical GPU ${phys}, port ${port} ==="
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  CUDA_VISIBLE_DEVICES="$phys" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
    --port ${port} \
    --devices='[0]' \
    --max_envs 48 \
    --max_inflight 48 \
    --thread_pool_size 48 \
    --session_timeout 7200.0 \
    > "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${env_i}.log" 2>&1 &
  ENV_PIDS+=("$!")
done

for env_i in "${!ENV_URLS[@]}"; do
  port=$((ENV_PORT + env_i))
  pid="${ENV_PIDS[$env_i]}"
  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "Env server ${env_i} health OK after ${i} tries"
      break
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "ERROR: env server ${env_i} exited early"
      tail -200 "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${env_i}.log" || true
      exit 4
    fi
    sleep 5
  done
  curl -fsS "http://127.0.0.1:${port}/health"
done

TMP_CONFIG_DIR=$(mktemp -d)
ENV_URL_FILE=${TMP_CONFIG_DIR}/env_urls.txt
printf "%s\n" "${ENV_URLS[@]}" > "$ENV_URL_FILE"
cp "${SCRIPTDIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${SCRIPTDIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

export TOKENIZERS_PARALLELISM=true
export RAY_DEDUP_LOGS=0
ray stop --force >/dev/null 2>&1 || true
HEAD_IP=$(hostname -I | awk '{print $1}')
RAY_PORT=6379

echo "=== Starting Ray on training GPUs ${TRAIN_CUDA} ==="
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" ray start --head \
  --port=${RAY_PORT} \
  --num-cpus=96 \
  --num-gpus=${TRAIN_GPUS} \
  --node-ip-address="${HEAD_IP}" \
  --include-dashboard=false
sleep 15

echo "=== Launching VAGEN training ==="
set +e
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
  --config-path="${BASEDIR}/vagen/configs" \
  --config-name='vagen_multiturn' \
  data.train_files="${TMP_CONFIG_DIR}/train.yaml" \
  data.val_files="${TMP_CONFIG_DIR}/val.yaml" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
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
  actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
  actor_rollout_ref.rollout.agent.max_concurrent_trajectories=${AGENT_MAX_CONCURRENT_TRAJECTORIES} \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=${TRAIN_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.max_actor_ckpt_to_keep=50 \
  trainer.max_critic_ckpt_to_keep=50 \
  trainer.project_name=nimloth_navigation \
  trainer.experiment_name=vagen_nav_dgx44_2env4train_original_base_common_sticky_semaphore_resp20k \
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
  critic.ppo_max_token_len_per_gpu=32768 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  huggingface_hub.hf_save_freq=null \
  +ray_kwargs.ray_init.address=auto \
  2>&1 | tee "${RUN_DIR}/vagen_nav_dgx44_2env4train_original_base_common_sticky_semaphore_resp20k.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "=== Env server log tails ==="
for env_i in "${!ENV_PIDS[@]}"; do
  echo "--- env server ${env_i} ---"
  tail -120 "${RUN_DIR}/env_server_${SLURM_JOB_ID}_${env_i}.log" || true
done
echo "=== DGX-44 training finished rc=${TRAIN_EXIT} at $(date) ==="
exit ${TRAIN_EXIT}
