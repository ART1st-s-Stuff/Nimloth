#!/usr/bin/env bash
#SBATCH --job-name=step60-original-val200
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=256G
#SBATCH --time=06:00:00
set -euo pipefail
: "${REPO:?}" "${VAGEN_DIR:?}" "${VERL_DIR:?}" "${PYTHON_ENV:?}"
: "${NIMLOTH_COMMIT:?}" "${VAGEN_COMMIT:?}" "${VERL_COMMIT:?}"
: "${PREPARED_DIR:?}" "${INIT_HF:?}" "${RUN_OUT:?}"
PY=${PYTHON_ENV}/bin/python3
export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${VAGEN_DIR}:${VERL_DIR}:${REPO}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=true
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME} TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN CUDA_DEVICE_ORDER=PCI_BUS_ID WANDB_MODE=disabled
for entry in "${REPO}:${NIMLOTH_COMMIT}" "${VAGEN_DIR}:${VAGEN_COMMIT}" "${VERL_DIR}:${VERL_COMMIT}"; do
  tree=${entry%:*}; expected=${entry##*:}
  [[ $(git -C "$tree" rev-parse HEAD) == "$expected" ]]
  [[ -z $(git -C "$tree" status --porcelain=v1 --untracked-files=all) ]]
done
[[ -x "$PY" && -s "${PREPARED_DIR}/validation.parquet" && -s "${PREPARED_DIR}/manifest.json" ]]
[[ ! -e "$RUN_OUT" ]] || { echo "Output already exists: $RUN_OUT" >&2; exit 2; }
"$PY" - "$INIT_HF" <<'PY'
import importlib.metadata as m,json,sys
from pathlib import Path
from safetensors import safe_open
expected={'vllm':'0.8.5.post1','torch':'2.6.0','transformers':'4.49.0','ray':'2.55.1'}
for key,value in expected.items():
    actual=m.version(key).split('+')[0]
    assert actual==value,(key,actual,value)
root=Path(sys.argv[1]);cfg=json.loads((root/'config.json').read_text())
assert cfg['model_type']=='qwen2_5_vl' and cfg['num_attention_heads']%4==0
index=json.loads((root/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    with safe_open(root/name,framework='pt',device='cpu') as shard:
        keys=set(shard.keys())
        assert all(k in keys for k,v in index['weight_map'].items() if v==name)
print('CPU_CHECKPOINT_AND_VERSIONS_OK',flush=True)
PY
PORT_BASE=${PORT_BASE:-21860}
ENV_PORT=$PORT_BASE
RAY_PORT=$((PORT_BASE+1))
DASHBOARD_PORT=$((PORT_BASE+2))
CMD=("$PY" -m vagen.trainer.main_ppo
  "data.train_files=${PREPARED_DIR}/validation.parquet" "data.val_files=${PREPARED_DIR}/validation.parquet"
  data.train_batch_size=32 data.val_batch_size=1 data.max_prompt_length=1024
  data.max_response_length=256 data.max_trajectory_length=16000 data.truncation=left
  algorithm.adv_estimator=reinforce_plus_plus algorithm.kl_ctrl.kl_coef=0
  "actor_rollout_ref.model.path=${INIT_HF}"
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.do_sample=True actor_rollout_ref.rollout.temperature=0.7
  actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_trajectory_length=6144
  actor_rollout_ref.rollout.max_model_len=6144 actor_rollout_ref.rollout.limit_mm_per_prompt=6
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.enable_chunked_prefill=False actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=False actor_rollout_ref.rollout.disable_log_stats=False
  +actor_rollout_ref.ref.use_ref=False actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  trainer.logger='[console]' trainer.n_gpus_per_node=8 trainer.nnodes=1
  trainer.val_before_train=True trainer.val_only=True trainer.save_freq=-1 trainer.test_freq=-1
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.resume_mode=disable
  trainer.val_generations_to_log_to_wandb=0
  "trainer.default_local_dir=${RUN_OUT}/unused_checkpoints"
  "+trainer.original_validation_output_dir=${RUN_OUT}/rollouts"
  rollout_manager.max_turns=20 rollout_manager.window_size=5 rollout_manager.max_trajectory_length=16000
  rollout_manager.n_trajectory=1 rollout_manager.use_service=True
  "rollout_manager.base_url=http://127.0.0.1:${ENV_PORT}"
  rollout_manager.timeout=1200 rollout_manager.max_workers=2 rollout_manager.use_multi_turn_reward=False
)
cd "$VAGEN_DIR"
if [[ ${1:-} == --preflight ]]; then
  "${CMD[@]}" --cfg job --resolve
  exit 0
fi
[[ ${SLURM_JOB_NUM_NODES:?} == 1 ]]
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ ${#GPUS[@]} == 8 ]] || { echo 'Eight visible GPUs required' >&2; exit 2; }
mkdir "$RUN_OUT"
mkdir "${RUN_OUT}/runtime" "${RUN_OUT}/env_home" "${RUN_OUT}/env_home/.ai2thor"
ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases "${RUN_OUT}/env_home/.ai2thor/releases"
export ORIGINAL_VALIDATION_RUNTIME_DIR="${RUN_OUT}/runtime"
export TRITON_CACHE_DIR="${RUN_OUT}/runtime/triton" FLASHINFER_WORKSPACE_DIR="${RUN_OUT}/runtime/flashinfer"
export XDG_CACHE_HOME="${RUN_OUT}/runtime/cache" TORCH_EXTENSIONS_DIR="${RUN_OUT}/runtime/extensions"
VULKAN_ROOT=/project/peilab/atst/flower/.local-vulkan
export LD_LIBRARY_PATH="${VULKAN_ROOT}/extracted/usr/lib/x86_64-linux-gnu:${VULKAN_ROOT}/runtime/extracted/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VULKAN_ROOT}/icd.d/nvidia_icd.json" VK_DRIVER_FILES="${VULKAN_ROOT}/icd.d/nvidia_icd.json"
IP=$(hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print;exit}')
[[ -n "$IP" ]]
export VLLM_HOST_IP=$IP
IFACE=$(ip -o -4 addr show | awk -v ip="$IP" '$4 ~ ("^" ip "/") {print $2;exit}')
[[ -n "$IFACE" ]]
export NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE
export RAY_TMPDIR="/tmp/step60-original-val-${SLURM_JOB_ID}"
ENV_PID= RAY_PID= POLICY_PID=
cleanup() {
  rc=$?
  trap - EXIT
  for pid in "$POLICY_PID" "$ENV_PID" "$RAY_PID"; do
    if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  for pid in "$POLICY_PID" "$ENV_PID" "$RAY_PID"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
  printf '%s\n' "$rc" > "${RUN_OUT}/exit_code.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
{
  date -u --iso-8601=seconds
  printf 'job=%s node=%s gpus=%s ranks=8 tp=4 cpus=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$CUDA_VISIBLE_DEVICES" "$SLURM_CPUS_PER_TASK"
  printf 'nimloth=%s vagen=%s verl=%s\n' "$NIMLOTH_COMMIT" "$VAGEN_COMMIT" "$VERL_COMMIT"
  printf '%q ' "${CMD[@]}"; printf '\n'
} > "${RUN_OUT}/launch_contract.txt"
"$PY" - "$IP" "$ENV_PORT" "$RAY_PORT" "$DASHBOARD_PORT" <<'PY'
import socket,sys
for port in sys.argv[2:]:
    with socket.socket() as s:s.bind(('0.0.0.0',int(port)))
PY
# Original topology: environment service and 8 policy ranks share the allocation.
setsid env HOME="${RUN_OUT}/env_home" "$PY" -m vagen.server.server \
  "server.port=${ENV_PORT}" use_state_reward=False navigation.max_workers=2 \
  'navigation.devices=[0,1,2,3,4,5,6,7]' > "${RUN_OUT}/environment.log" 2>&1 &
ENV_PID=$!
setsid "$PY" -m ray.scripts.scripts start --head --block --include-dashboard=false \
  "--node-ip-address=${IP}" "--port=${RAY_PORT}" "--dashboard-port=${DASHBOARD_PORT}" \
  "--temp-dir=${RAY_TMPDIR}" --num-gpus=8 "--num-cpus=${SLURM_CPUS_PER_TASK}" > "${RUN_OUT}/ray.log" 2>&1 &
RAY_PID=$!
printf 'env=%s ray=%s\n' "$ENV_PID" "$RAY_PID" > "${RUN_OUT}/pids.txt"
"$PY" - "$ENV_PORT" "$IP" "$RAY_PORT" "$ENV_PID" "$RAY_PID" <<'PY'
import os,socket,sys,time
end=time.monotonic()+180
while time.monotonic()<end:
    for pid in sys.argv[4:]:os.kill(int(pid),0)
    try:
        for host,port in [('127.0.0.1',sys.argv[1]),(sys.argv[2],sys.argv[3])]:
            with socket.create_connection((host,int(port)),timeout=2):pass
        break
    except OSError:time.sleep(2)
else:raise RuntimeError('Environment or Ray startup timeout')
PY
export RAY_ADDRESS="${IP}:${RAY_PORT}"
setsid "${CMD[@]}" > "${RUN_OUT}/validation.log" 2>&1 &
POLICY_PID=$!
printf 'policy=%s\n' "$POLICY_PID" >> "${RUN_OUT}/pids.txt"
wait "$POLICY_PID"
POLICY_PID=
"$PY" "${REPO}/experiments/training/sft1/original_validation200.py" summarize \
  --manifest "${PREPARED_DIR}/manifest.json" --rollouts-dir "${RUN_OUT}/rollouts" --output "${RUN_OUT}/success_summary.json"
printf 'ALL_200_COLLECTION_COMPLETE\n' > "${RUN_OUT}/collection_done.flag"
