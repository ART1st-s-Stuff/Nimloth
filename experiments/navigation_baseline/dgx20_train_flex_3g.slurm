#!/bin/bash
#SBATCH --job-name=vagen-dgx20-flex-3g
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodelist=dgx-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:3
#SBATCH --mem=256G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASEDIR=${ROOT}/external/VAGEN
RUN_DIR=${SCRIPTDIR}/runs/vagen_nav_dgx20_flex_3g
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
ENV_PORT=8000
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
TOTAL_STEPS=50
TRAIN_BATCH_SIZE=2
PPO_MINI_BATCH_SIZE=2
MIN_ENV_GPUS=1

echo "=== DGX-20 flex 3GPU training starting at $(date) ==="
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
export TMPDIR=/tmp/vagen-dgx20-flex-${USER}-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
cleanup() {
  set +e
  if [ -n "${ENV_PID:-}" ]; then kill "$ENV_PID" >/dev/null 2>&1 || true; wait "$ENV_PID" >/dev/null 2>&1 || true; fi
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

IFS=',' read -r -a ALLOC_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
ALL_CUDA=$(IFS=,; echo "${ALLOC_GPU_IDS[*]}")
echo "Allocated physical GPU ids: ${ALLOC_GPU_IDS[*]}"
echo "All allocated CUDA_VISIBLE_DEVICES for env probing: ${ALL_CUDA}"
rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true

GOOD_ORD=()
for ord in "${!ALLOC_GPU_IDS[@]}"; do
  phys="${ALLOC_GPU_IDS[$ord]}"
  rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
  echo "=== Testing AI2-THOR on allocated ordinal ${ord} physical GPU ${phys} at $(date) ==="
  set +e
  CUDA_VISIBLE_DEVICES="$ALL_CUDA" TEST_GPU_DEVICE="$ord" timeout 90s python - <<'PY'
import os, time
import ai2thor.controller
from ai2thor.platform import CloudRendering
gpu_device=int(os.environ['TEST_GPU_DEVICE'])
print(f'create gpu_device={gpu_device}', flush=True)
t0=time.time()
c=ai2thor.controller.Controller(
    agentMode='default', gridSize=0.1, visibilityDistance=10,
    renderDepthImage=False, renderInstanceSegmentation=False,
    width=255, height=255, fieldOfView=100,
    platform=CloudRendering, gpu_device=gpu_device,
    server_timeout=45, server_start_timeout=45,
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
    echo "allocated ordinal ${ord} physical ${phys}: AI2-THOR OK"
    GOOD_ORD+=("$ord")
  else
    echo "allocated ordinal ${ord} physical ${phys}: AI2-THOR FAILED/TIMEOUT rc=${rc}"
  fi
done

echo "Good AI2-THOR allocated ordinals: ${GOOD_ORD[*]:-NONE}"
if [ "${#GOOD_ORD[@]}" -lt "$MIN_ENV_GPUS" ]; then
  echo "ERROR: no AI2-THOR-good GPU in this 3-GPU allocation"
  exit 2
fi
ENV_ORD=("${GOOD_ORD[0]}")
TRAIN_ORD=()
for ord in "${!ALLOC_GPU_IDS[@]}"; do
  [ "$ord" = "${ENV_ORD[0]}" ] || TRAIN_ORD+=("$ord")
done
TRAIN_GPUS=${#TRAIN_ORD[@]}
if [ "$TRAIN_GPUS" -lt 1 ]; then echo "ERROR: no training GPU left"; exit 3; fi
ENV_DEVICES="[${ENV_ORD[0]}]"
TRAIN_CUDA=$(python - <<PY
alloc="${ALLOC_GPU_IDS[*]}".split()
ords=[int(x) for x in "${TRAIN_ORD[*]}".split()]
print(','.join(alloc[i] for i in ords))
PY
)

echo "Env allocated ordinal: ${ENV_ORD[*]} devices_arg=${ENV_DEVICES} visible=${ALL_CUDA}"
echo "Train allocated ordinals: ${TRAIN_ORD[*]} physical CUDA_VISIBLE_DEVICES=${TRAIN_CUDA} TRAIN_GPUS=${TRAIN_GPUS}"

rm -f "${HOME}/.ai2thor/cuda-vulkan-mapping.json" "${HOME}/cuda-vulkan-mapping.json" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="$ALL_CUDA" PYTHONUNBUFFERED=1 python -m vagen.envs.navigation.serve \
  --port ${ENV_PORT} \
  --devices="${ENV_DEVICES}" \
  --max_envs 16 \
  --max_inflight 16 \
  --thread_pool_size 16 \
  --session_timeout 7200.0 \
  > "${RUN_DIR}/env_server_${SLURM_JOB_ID}.log" 2>&1 &
ENV_PID=$!

for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${ENV_PORT}/health" >/dev/null 2>&1; then
    echo "Env server health OK after ${i} tries"
    break
  fi
  if ! kill -0 "$ENV_PID" >/dev/null 2>&1; then
    echo "ERROR: env server exited early"
    tail -200 "${RUN_DIR}/env_server_${SLURM_JOB_ID}.log" || true
    exit 4
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${ENV_PORT}/health"

TMP_CONFIG_DIR=$(mktemp -d)
ENV_URL_FILE=${TMP_CONFIG_DIR}/env_urls.txt
echo "http://$(hostname):${ENV_PORT}" > "$ENV_URL_FILE"
cp "${SCRIPTDIR}/train.yaml" "${TMP_CONFIG_DIR}/train.yaml"
cp "${SCRIPTDIR}/val.yaml" "${TMP_CONFIG_DIR}/val.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/train.yaml"
sed -i "s|ENV_URL_FILE|${ENV_URL_FILE}|g" "${TMP_CONFIG_DIR}/val.yaml"

export TOKENIZERS_PARALLELISM=true
export RAY_DEDUP_LOGS=0
ray stop --force >/dev/null 2>&1 || true
HEAD_IP=$(hostname -I | awk '{print $1}')
RAY_PORT=6379

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" ray start --head \
  --port=${RAY_PORT} \
  --num-cpus=48 \
  --num-gpus=${TRAIN_GPUS} \
  --node-ip-address="${HEAD_IP}" \
  --include-dashboard=false
sleep 10

set +e
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA" PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
  --config-path="${BASEDIR}/vagen/configs" \
  --config-name='vagen_multiturn' \
  data.train_files="${TMP_CONFIG_DIR}/train.yaml" \
  data.val_files="${TMP_CONFIG_DIR}/val.yaml" \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=${TRAIN_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=0 \
  trainer.max_actor_ckpt_to_keep=50 \
  trainer.max_critic_ckpt_to_keep=50 \
  trainer.project_name=nimloth_navigation \
  trainer.experiment_name=vagen_nav_dgx20_flex_3g \
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
  2>&1 | tee "${RUN_DIR}/vagen_nav_dgx20_flex_3g.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

tail -200 "${RUN_DIR}/env_server_${SLURM_JOB_ID}.log" || true
exit ${TRAIN_EXIT}
