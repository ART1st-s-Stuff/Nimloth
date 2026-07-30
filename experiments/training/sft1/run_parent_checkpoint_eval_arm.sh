#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?usage: run_parent_checkpoint_eval_arm.sh sft1|vagen}
if [[ "${ARM}" != sft1 && "${ARM}" != vagen ]]; then
  echo "unknown arm: ${ARM}" >&2
  exit 2
fi
: "${EVAL_WORKTREE:?EVAL_WORKTREE is required}"
: "${EVAL_COMMIT:?EVAL_COMMIT is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${ARM_OUTPUT:?ARM_OUTPUT is required}"
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
: "${WANDB_RUN_NAME:?WANDB_RUN_NAME is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"

ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
RAY_CLI=${ROOT}/.venv-vagen-main/bin/ray
VAGEN_DIR=${EVAL_WORKTREE}/external/VAGEN
SFT1_DIR=${EVAL_WORKTREE}/experiments/training/sft1
VAGEN_COMMIT=192c35a91f3941b72d5e1272af6603ef7a7d93e0
EVAL_SETS=(base common_sense complex_instruction visual_appearance long_horizon)
EPISODES_PER_SET=60
ALLOCATED_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}
IFS=',' read -r -a ALLOCATED_GPUS <<< "${ALLOCATED_CUDA_VISIBLE_DEVICES}"
for index in "${!ALLOCATED_GPUS[@]}"; do
  ALLOCATED_GPUS[$index]=$(echo "${ALLOCATED_GPUS[$index]}" | xargs)
done
if (( ${#ALLOCATED_GPUS[@]} != 6 )); then
  echo "each arm requires exactly 6 visible GPUs, got ${ALLOCATED_CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

ARM_PORT_OFFSET=0
[[ "${ARM}" == vagen ]] && ARM_PORT_OFFSET=1000
ENV_PORT=$((18400 + ARM_PORT_OFFSET + SLURM_JOB_ID % 800))
RAY_PORT=$((26400 + SLURM_JOB_ID % 800))
RAY_DASHBOARD_PORT=$((27400 + SLURM_JOB_ID % 800))
ENV_URL=http://127.0.0.1:${ENV_PORT}
# vLLM and Ray create AF_UNIX sockets below TMPDIR/RAY_TMPDIR. Keep this root
# short enough for Linux's 107-byte sockaddr_un limit even after their suffixes.
RUNTIME_ROOT=/tmp/npe-${SLURM_JOB_ID}-${ARM}
AI2THOR_SHARED_HOME=/project/peilab/atst/flower/.ai2thor-home
ENV_PID=
RAY_PID=
CHILD_PIDS=()
PROBE_PIDS=()
PROBE_GPUS=()
PROBE_HOMES=()
GOOD_ENV_GPUS=()
GOOD_ENV_HOMES=()

mkdir -p "${ARM_OUTPUT}" "${ARM_OUTPUT}/eval_sets" "${RUNTIME_ROOT}"
export PATH=${ROOT}/.venv-vagen-main/bin:${PATH}
export PYTHONPATH=${EVAL_WORKTREE}/src:${VAGEN_DIR}:${VAGEN_DIR}/verl:${PYTHONPATH:-}
export HOME=${ROOT}/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [[ "${ARM}" == sft1 ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
  # VERL's vLLM sharding manager uses CuMemAllocator memory pools, which reject
  # PyTorch expandable segments. Unset before Ray starts so workers inherit it.
  unset PYTORCH_CUDA_ALLOC_CONF
  # FlashInfer otherwise JIT-compiles its sampling kernels on first use. The
  # superpod's default /usr/bin/nvcc is a tutorial wrapper, not a CUDA compiler.
  # Greedy evaluation uses vLLM's equivalent PyTorch sampler instead.
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_DIR=${ARM_OUTPUT}/wandb
export RAY_TMPDIR=${RUNTIME_ROOT}/ray
export TMPDIR=${RUNTIME_ROOT}/tmp
mkdir -p "${HOME}" "${WANDB_DIR}" "${RAY_TMPDIR}" "${TMPDIR}"

cleanup() {
  set +e
  for pid in "${PROBE_PIDS[@]:-}" "${CHILD_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${PROBE_PIDS[@]:-}" "${CHILD_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${RAY_PID}" ]]; then
    kill "${RAY_PID}" >/dev/null 2>&1 || true
    wait "${RAY_PID}" >/dev/null 2>&1 || true
    "${PY}" "${RAY_CLI}" stop --force >/dev/null 2>&1 || true
  fi
  if [[ -n "${ENV_PID}" ]]; then
    kill "${ENV_PID}" >/dev/null 2>&1 || true
    wait "${ENV_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${RUNTIME_ROOT}" == "/tmp/npe-${SLURM_JOB_ID}-${ARM}" ]]; then
    rm -rf -- "${RUNTIME_ROOT}"
  fi
}
trap cleanup EXIT

prepare_ai2thor_home() {
  local ai2thor_home=$1
  mkdir -p "${ai2thor_home}/.ai2thor"
  if [[ ! -e "${ai2thor_home}/.ai2thor/releases" ]]; then
    ln -s "${AI2THOR_SHARED_HOME}/.ai2thor/releases" \
      "${ai2thor_home}/.ai2thor/releases"
  fi
  rm -f "${ai2thor_home}/.ai2thor/cuda-vulkan-mapping.json"
}

{
  echo "time=$(date --iso-8601=seconds)"
  echo "arm=${ARM}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "nimloth_commit=${EVAL_COMMIT}"
  echo "vagen_commit=${VAGEN_COMMIT}"
  echo "checkpoint=${MODEL_PATH}"
  echo "output=${ARM_OUTPUT}"
  echo "sampling=greedy_temp0_top_p1_top_k-1_n1_tokens512_turns20"
  echo "navigation=resolution255_step0.3_threshold1_format0_perturn0.01_success1"
  echo "allocated_gpus=${ALLOCATED_GPUS[*]}"
  echo "wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_NAME}/${WANDB_RUN_ID}"
} | tee "${ARM_OUTPUT}/controller.log"

[[ "$(git -C "${EVAL_WORKTREE}" rev-parse HEAD)" == "${EVAL_COMMIT}" ]]
git -C "${EVAL_WORKTREE}" diff --quiet
git -C "${EVAL_WORKTREE}" diff --cached --quiet
[[ "$(git -C "${VAGEN_DIR}" rev-parse HEAD)" == "${VAGEN_COMMIT}" ]]
git -C "${VAGEN_DIR}" diff --quiet
git -C "${VAGEN_DIR}" diff --cached --quiet
[[ -x "${PY}" && -f "${MODEL_PATH}/model.safetensors.index.json" ]]

"${PY}" - "${ARM}" "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

arm, root = sys.argv[1], Path(sys.argv[2])
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file()]
if missing:
    raise RuntimeError(f"missing model shards: {missing}")
if arm == "sft1":
    observed = (
        config.get("nimloth_latent_query_mode"),
        int(config.get("nimloth_latent_token_count", -1)),
    )
    if observed != ("inject", 16):
        raise RuntimeError(f"wrong SFT1 protocol: {observed}")
elif config.get("nimloth_latent_query_mode") is not None:
    raise RuntimeError("VAGEN parent unexpectedly declares the Nimloth protocol")
print(json.dumps({"checkpoint_preflight": "ok", "arm": arm, "shards": shards}))
PY

# AI2-THOR can stay alive while returning black frames on a bad Vulkan mapping.
# Probe every allocated GPU in parallel, then pin the service to a proven GPU.
for gpu in "${ALLOCATED_GPUS[@]}"; do
  safe_gpu=${gpu//[^a-zA-Z0-9_.-]/_}
  probe_home=${RUNTIME_ROOT}/ai2thor_probe_${safe_gpu}
  prepare_ai2thor_home "${probe_home}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES}"
    export AI2THOR_HOME_ROOT="${probe_home}"
    source "${EVAL_WORKTREE}/experiments/training/baseline/setup_ai2thor_env.sh"
    timeout --signal=TERM 150s "${PY}" \
      -m nimloth.environment.navigation.direct_render_probe \
      --gpu-device "${gpu}"
  ) > "${ARM_OUTPUT}/render_probe_${safe_gpu}.log" 2>&1 &
  PROBE_PIDS+=("$!")
  PROBE_GPUS+=("${gpu}")
  PROBE_HOMES+=("${probe_home}")
done
for index in "${!PROBE_PIDS[@]}"; do
  if wait "${PROBE_PIDS[$index]}"; then
    GOOD_ENV_GPUS+=("${PROBE_GPUS[$index]}")
    GOOD_ENV_HOMES+=("${PROBE_HOMES[$index]}")
  else
    safe_gpu=${PROBE_GPUS[$index]//[^a-zA-Z0-9_.-]/_}
    tail -100 "${ARM_OUTPUT}/render_probe_${safe_gpu}.log" >&2 || true
  fi
done
(( ${#GOOD_ENV_GPUS[@]} >= 1 )) || { echo "no GPU passed render probe" >&2; exit 4; }

ENV_GPU=${GOOD_ENV_GPUS[0]}
ENV_GPU_HOME=${GOOD_ENV_HOMES[0]}
POLICY_GPUS=()
for gpu in "${ALLOCATED_GPUS[@]}"; do
  [[ "${gpu}" == "${ENV_GPU}" ]] || POLICY_GPUS+=("${gpu}")
done
(( ${#POLICY_GPUS[@]} == 5 ))
echo "env_gpu=${ENV_GPU} policy_gpus=${POLICY_GPUS[*]}" | tee -a "${ARM_OUTPUT}/controller.log"

LATENT_COUNT=1
[[ "${ARM}" == sft1 ]] && LATENT_COUNT=16
(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES}"
  export NIMLOTH_LATENT_TOKEN_COUNT=${LATENT_COUNT}
  export AI2THOR_HOME_ROOT="${ENV_GPU_HOME}"
  source "${EVAL_WORKTREE}/experiments/training/baseline/setup_ai2thor_env.sh"
  cd "${VAGEN_DIR}"
  exec "${PY}" -m vagen.server.server \
    server.host=0.0.0.0 server.port="${ENV_PORT}" \
    use_state_reward=False navigation.devices="[${ENV_GPU}]" \
    navigation.max_workers=16
) > "${ARM_OUTPUT}/env_server.log" 2>&1 &
ENV_PID=$!
for _ in $(seq 1 120); do
  curl -fsS --max-time 10 "${ENV_URL}/health" >/dev/null 2>&1 && break
  kill -0 "${ENV_PID}" >/dev/null 2>&1 || {
    tail -200 "${ARM_OUTPUT}/env_server.log" >&2 || true
    exit 4
  }
  sleep 3
done
curl -fsS --max-time 10 "${ENV_URL}/health" >/dev/null
"${PY}" -m nimloth.environment.navigation.prewarm \
  --env-url "${ENV_URL}" --eval-set base --seed 1 \
  --env-id "parent-${ARM}-prewarm-${SLURM_JOB_ID}" \
  | tee -a "${ARM_OUTPUT}/controller.log"

if [[ "${ARM}" == sft1 ]]; then
  for index in "${!EVAL_SETS[@]}"; do
    eval_set=${EVAL_SETS[$index]}
    gpu=${POLICY_GPUS[$index]}
    dataset_dir=${ARM_OUTPUT}/eval_sets/${eval_set}
    mkdir -p "${dataset_dir}"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export NIMLOTH_LATENT_TOKEN_COUNT=16
      exec "${PY}" "${EVAL_WORKTREE}/experiments/training/rl/rollout_env.py" \
        --model "${MODEL_PATH}" --env-url "${ENV_URL}" \
        --output-dir "${dataset_dir}" --resume-existing-rollouts \
        --num-episodes "${EPISODES_PER_SET}" --max-steps 20 \
        --eval-set "${eval_set}" --split test --seed-offset 1 \
        --seed-per-eval-set \
        --temperature 0 --top-p 1 --credit-assignment token \
        --max-response-tokens 512 --navigation-profile vagen_eval \
        --backend vllm --tensor-parallel-size 1 --max-model-len 32768 \
        --gpu-memory-utilization 0.6 --vllm-enforce-eager \
        --vllm-mm-processor-cache-gb 0
    ) > "${ARM_OUTPUT}/${eval_set}.log" 2>&1 &
    CHILD_PIDS+=("$!")
  done
  failed=0
  set +e
  for index in "${!CHILD_PIDS[@]}"; do
    if ! wait "${CHILD_PIDS[$index]}"; then
      failed=1
      tail -120 "${ARM_OUTPUT}/${EVAL_SETS[$index]}.log" >&2 || true
    fi
  done
  set -e
  (( failed == 0 )) || exit 5
  VAGEN_JSONL_ARGS=()
else
  POLICY_GPUS=("${POLICY_GPUS[@]:0:4}")
  POLICY_GPU_CSV=$(IFS=,; echo "${POLICY_GPUS[*]}")
  CONTROL_DIR=${ARM_OUTPUT}/control
  VALIDATION_DIR=${ARM_OUTPUT}/validation
  mkdir -p "${CONTROL_DIR}" "${VALIDATION_DIR}"
  "${PY}" - "${CONTROL_DIR}" <<'PY'
import sys
from pathlib import Path
from datasets import Dataset

root = Path(sys.argv[1])
specs = (
    ("navigation_base_test", "base"),
    ("navigation_common_test", "common_sense"),
    ("navigation_complex_instruction_test", "complex_instruction"),
    ("navigation_visual_appearance_test", "visual_appearance"),
    ("navigation_long_horizon_test", "long_horizon"),
)
rows = []
for data_source, eval_set in specs:
    for seed in range(1, 61):
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": ""}],
            "extra_info": {
                "split": "test",
                "env_name": "navigation",
                "env_config": {
                    "eval_set": eval_set,
                    "resolution": 255,
                    "prompt_format": "source_eval_mode",
                    "max_actions_per_step": 1,
                    "action_sep": "|",
                    "example_count": 0,
                    "step_length": 0.3,
                    "success_threshold": 1.0,
                    "format_reward": 0.0,
                    "per_turn_format_reward": 0.01,
                    "success_reward": 1.0,
                    "use_state_reward": False,
                },
                "seed": seed,
            },
        })
Dataset.from_list(rows).to_parquet(root / "test300.parquet")
Dataset.from_list(rows[:24]).to_parquet(root / "train_placeholder.parquet")
print(f"wrote {len(rows)} exact heldout evaluation rows")
PY
  NODE_IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
  [[ -n "${NODE_IP}" ]] || NODE_IP=$(hostname -I | awk '{print $1}')
  CUDA_VISIBLE_DEVICES="${POLICY_GPU_CSV}" "${PY}" "${RAY_CLI}" start --head \
    --node-ip-address="${NODE_IP}" --port="${RAY_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus=4 --num-cpus=112 \
    --include-dashboard=false --disable-usage-stats --block \
    > "${ARM_OUTPUT}/ray_head.log" 2>&1 &
  RAY_PID=$!
  for _ in $(seq 1 90); do
    "${PY}" - "${NODE_IP}" "${RAY_PORT}" <<'PY' >/dev/null 2>&1 && break
import socket
import sys
s = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2)
s.close()
PY
    kill -0 "${RAY_PID}" >/dev/null 2>&1 || { tail -200 "${ARM_OUTPUT}/ray_head.log" >&2; exit 5; }
    sleep 2
  done
  export RAY_ADDRESS=${NODE_IP}:${RAY_PORT}
  export CUDA_VISIBLE_DEVICES=${POLICY_GPU_CSV}
  cd "${VAGEN_DIR}"
  "${PY}" -m vagen.trainer.main_ppo \
    data.train_files="${CONTROL_DIR}/train_placeholder.parquet" \
    data.val_files="${CONTROL_DIR}/test300.parquet" \
    data.train_batch_size=24 data.val_batch_size=24 \
    +data.seed=42 +data.base_seed=42 +data.validation_shuffle=False \
    data.max_prompt_length=3000 data.max_response_length=20000 \
    data.max_trajectory_length=23000 data.truncation=left \
    algorithm.adv_estimator=reinforce_plus_plus algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.gamma=1.0 algorithm.lam=1.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.do_sample=False actor_rollout_ref.rollout.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=24000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_response_per_turn=512 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console']" trainer.val_before_train=True trainer.val_only=True \
    trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=-1 \
    trainer.assert_val_env_composition=True \
    'trainer.val_env_composition={navigation_base_test:{count:60,eval_set:base},navigation_common_test:{count:60,eval_set:common_sense},navigation_complex_instruction_test:{count:60,eval_set:complex_instruction},navigation_visual_appearance_test:{count:60,eval_set:visual_appearance},navigation_long_horizon_test:{count:60,eval_set:long_horizon}}' \
    trainer.project_name=nimloth-sft1 trainer.experiment_name="${WANDB_RUN_NAME}" \
    trainer.default_local_dir="${ARM_OUTPUT}/checkpoints_unused" \
    trainer.validation_data_dir="${VALIDATION_DIR}" \
    trainer.val_generations_to_log_to_wandb=0 trainer.total_epochs=1 \
    trainer.total_training_steps=1 trainer.resume_mode=disable \
    rollout_manager.max_turns=20 rollout_manager.max_trajectory_length=23000 \
    rollout_manager.n_trajectory=1 rollout_manager.use_service=True \
    rollout_manager.base_url="${ENV_URL}" rollout_manager.timeout=500 \
    rollout_manager.max_workers=8 rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True rollout_manager.use_gae_mask=True \
    critic.model.path="${MODEL_PATH}" +ray_kwargs.ray_init.address=auto \
    2>&1 | tee "${ARM_OUTPUT}/vagen_eval.log"
  [[ -s "${VALIDATION_DIR}/0.jsonl" ]]
  VAGEN_JSONL_ARGS=(--vagen-jsonl "${VALIDATION_DIR}/0.jsonl")
fi

"${PY}" "${SFT1_DIR}/finalize_parent_checkpoint_eval.py" \
  --arm "${ARM}" --output-dir "${ARM_OUTPUT}" --checkpoint "${MODEL_PATH}" \
  --commit "${EVAL_COMMIT}" --vagen-commit "${VAGEN_COMMIT}" \
  --episodes-per-set "${EPISODES_PER_SET}" "${VAGEN_JSONL_ARGS[@]}" \
  --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" --wandb-run-id "${WANDB_RUN_ID}"

echo "arm=${ARM} status=ALL_OK time=$(date --iso-8601=seconds)" \
  | tee -a "${ARM_OUTPUT}/controller.log"
