#!/bin/bash
#SBATCH --job-name=sft1-source-parity
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=168
#SBATCH --gres=gpu:6
#SBATCH --mem=180G

# Replays the historical step-60 validation with its source VAGEN/runtime stack.
# This is intentionally separate from the pinned legacy rollout path: source
# validation used VAGEN f7aefd3 + verl d8e5210, torch 2.6, transformers 4.49,
# and vLLM 0.8.2. Production rollout must remain blocked until this gate passes.
set -euo pipefail

ROOT=${REPO:-/project/peilab/atst/nimloth/.worktree/vagen-legacy-wm-k8}
SOURCE_VAGEN_DIR=${SOURCE_VAGEN_DIR:-/project/peilab/atst/nimloth/.worktree/VAGEN-source-rollout}
PYTHON_ENV=${SOURCE_PYTHON_ENV:-/project/peilab/atst/nimloth/.venv}
RUN_DIR=${ROLLOUT_RUN_DIR:?set ROLLOUT_RUN_DIR}
RUN_NAME=${ROLLOUT_RUN_NAME:-source_runtime_parity120}
MODEL_PATH=${INIT_HF:?set INIT_HF}
EXPECTED_VAGEN_COMMIT=f7aefd3057bc462338065b75259f6bd946b84325
EXPECTED_VERL_COMMIT=d8e521045e51abbff1351a86aa6d256c6b36e8ac
CONTROL_DIR=${RUN_DIR}/source_runtime_control
ENV_URL_FILE=${CONTROL_DIR}/env_urls.txt
VALIDATION_DIR=${RUN_DIR}/validation/source_val/shard_001_060
LOG_FILE=${RUN_DIR}/source_runtime_parity.log
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}

mkdir -p "${RUN_DIR}" "${CONTROL_DIR}" "${VALIDATION_DIR}"
rm -f "${VALIDATION_DIR}/0.jsonl"
rm -rf "${VALIDATION_DIR}/image_0"

[ -x "${PYTHON_ENV}/bin/python" ] || { echo "missing source python: ${PYTHON_ENV}/bin/python"; exit 2; }
[ "$(git -C "${SOURCE_VAGEN_DIR}" rev-parse HEAD)" = "${EXPECTED_VAGEN_COMMIT}" ] || {
  echo "source VAGEN commit mismatch"; exit 2;
}
[ "$(git -C "${SOURCE_VAGEN_DIR}/verl" rev-parse HEAD)" = "${EXPECTED_VERL_COMMIT}" ] || {
  echo "source verl commit mismatch"; exit 2;
}
[ -z "$(git -C "${SOURCE_VAGEN_DIR}" status --porcelain=v1 --ignore-submodules=none)" ] || {
  echo "source VAGEN worktree is dirty"; git -C "${SOURCE_VAGEN_DIR}" status --short; exit 2;
}

export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${SOURCE_VAGEN_DIR}:${SOURCE_VAGEN_DIR}/verl:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export FLASHINFER_WORKSPACE_DIR=/project/peilab/atst/nimloth/.cache/flashinfer-source
export TMPDIR=/tmp/source-parity-${USER}-${SLURM_JOB_ID}
export RAY_TMPDIR=${TMPDIR}/ray
mkdir -p "${FLASHINFER_WORKSPACE_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

"${PYTHON_ENV}/bin/python" - <<'PY'
import importlib.metadata as m
import sys
expected = {
    "torch": "2.6.0",
    "transformers": "4.49.0",
    "vllm": "0.8.2",
}
actual = {name: m.version(name).split("+")[0] for name in expected}
print(f"python={sys.executable} versions={actual}", flush=True)
if actual != expected:
    raise SystemExit(f"source runtime mismatch: expected={expected} actual={actual}")
PY

# shellcheck disable=SC1091
source "${ROOT}/experiments/training/baseline/setup_ai2thor_env.sh"

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
if [ "${#ALLOCATED_GPUS[@]}" -ne 6 ]; then
  echo "expected exactly 6 allocated GPUs, got ${CUDA_VISIBLE_DEVICES:-unset}"
  exit 2
fi
ENV_GPUS="${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}"
POLICY_GPUS="${ALLOCATED_GPUS[2]},${ALLOCATED_GPUS[3]},${ALLOCATED_GPUS[4]},${ALLOCATED_GPUS[5]}"
NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
[ -n "${NODE_IP}" ] || NODE_IP=$(hostname -I | awk '{print $1}')
printf 'http://%s:8400\nhttp://%s:8401\n' "${NODE_IP}" "${NODE_IP}" > "${ENV_URL_FILE}"

ENV_PIDS=()
RAY_PID=""
cleanup() {
  set +e
  if [ -n "${RAY_PID}" ]; then
    kill "${RAY_PID}" >/dev/null 2>&1 || true
  fi
  "${PYTHON_ENV}/bin/ray" stop --force >/dev/null 2>&1 || true
  if [ "${#ENV_PIDS[@]}" -gt 0 ]; then
    kill "${ENV_PIDS[@]}" >/dev/null 2>&1 || true
    wait "${ENV_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

{
  echo "=== source runtime parity launch $(date -Iseconds) ==="
  echo "nimloth=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "source_vagen=${EXPECTED_VAGEN_COMMIT} source_verl=${EXPECTED_VERL_COMMIT}"
  echo "python=${PYTHON_ENV} model=${MODEL_PATH}"
  echo "env_gpus=${ENV_GPUS} policy_gpus=${POLICY_GPUS}"
} | tee "${LOG_FILE}"

for env_i in 0 1; do
  phys=${ALLOCATED_GPUS[$env_i]}
  port=$((8400 + env_i))
  CUDA_VISIBLE_DEVICES="${phys}" "${PYTHON_ENV}/bin/python" -m vagen.envs.navigation.serve \
    --port "${port}" --devices='[0]' --max_envs=48 --max_inflight=48 \
    --thread_pool_size=48 --session_timeout=7200.0 \
    > "${RUN_DIR}/source_env_${env_i}.log" 2>&1 &
  ENV_PIDS+=("$!")
done

for port in 8400 8401; do
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  [ "${ready}" -eq 1 ] || { echo "source env port ${port} failed" | tee -a "${LOG_FILE}"; exit 3; }
  echo "healthy source env port=${port}" | tee -a "${LOG_FILE}"
done

cat > "${CONTROL_DIR}/source_val.yaml" <<YAML
envs:
  - name: RemoteEnv
    n_envs: 60
    data_source: navigation_base
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
    data_source: navigation_common
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
YAML

# The historical validation used a four-replica async vLLM rollout.
CUDA_VISIBLE_DEVICES="${POLICY_GPUS}" "${PYTHON_ENV}/bin/ray" start --head \
  --port="${RAY_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --num-cpus=112 --num-gpus=4 --node-ip-address="${NODE_IP}" \
  --include-dashboard=false --disable-usage-stats --block \
  > "${RUN_DIR}/source_ray_head.log" 2>&1 &
RAY_PID=$!
for _ in $(seq 1 90); do
  if "${PYTHON_ENV}/bin/python" - <<PY >/dev/null 2>&1
import socket
s=socket.create_connection(("${NODE_IP}", ${RAY_PORT}), timeout=2); s.close()
PY
  then break; fi
  kill -0 "${RAY_PID}" >/dev/null 2>&1 || { echo "source Ray head exited"; exit 3; }
  sleep 2
done
sleep 5

export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
cd "${SOURCE_VAGEN_DIR}"
CUDA_VISIBLE_DEVICES="${POLICY_GPUS}" "${PYTHON_ENV}/bin/python" -m vagen.main_ppo \
  --config-path="${SOURCE_VAGEN_DIR}/vagen/configs" --config-name=vagen_multiturn \
  data.train_files="${CONTROL_DIR}/source_val.yaml" \
  data.val_files="${CONTROL_DIR}/source_val.yaml" \
  data.custom_cls.path="${SOURCE_VAGEN_DIR}/vagen/gym_agent_dataset.py" \
  data.train_batch_size=24 data.val_batch_size=24 \
  data.seed=42 +data.base_seed=42 data.validation_shuffle=False \
  data.max_prompt_length=3000 data.max_response_length=20000 \
  +trainer.assert_val_env_composition=True \
  '+trainer.val_env_composition.navigation_base={count:60,eval_set:base}' \
  '+trainer.val_env_composition.navigation_common={count:60,eval_set:common_sense}' \
  algorithm.adv_estimator=gae algorithm.use_kl_in_reward=True \
  algorithm.kl_ctrl.kl_coef=0.001 algorithm.gamma=1.0 algorithm.lam=1.0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${SOURCE_VAGEN_DIR}/vagen/configs/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.agent.max_concurrent_trajectories=6 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=1e-5 critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding=True critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 critic.ppo_max_token_len_per_gpu=32768 \
  critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True \
  trainer.logger='[console]' trainer.val_before_train=True +trainer.val_only=True \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.project_name=nimloth_navigation trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${RUN_DIR}/checkpoints_rollout" \
  trainer.validation_data_dir="${VALIDATION_DIR}" \
  trainer.log_val_generations=0 trainer.total_training_steps=1 trainer.resume_mode=disable \
  trainer.log_image.enable=True trainer.log_image.max_pending=1 trainer.log_image.png_compress_level=0 \
  huggingface_hub.hf_save_freq=null +ray_kwargs.ray_init.address=auto \
  2>&1 | tee -a "${LOG_FILE}"

[ -s "${VALIDATION_DIR}/0.jsonl" ] || { echo "missing source parity JSONL"; exit 4; }
echo "=== source runtime parity completed $(date -Iseconds) ===" | tee -a "${LOG_FILE}"
