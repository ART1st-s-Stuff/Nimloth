#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO must be the pinned remote worktree}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
VAGEN=${REPO}/external/VAGEN
VERL=${VAGEN}/verl
PY=${ROOT}/.venv-vagen-main/bin/python3
PARENT_COMMIT=${EXPECTED_PARENT_COMMIT}
VAGEN_COMMIT=3fc2509144bb8d1c1ebd57aab30dbece5c3794e4
VERL_COMMIT=3fe0a29975e1b02ae2bd1dec249f7807dd7966f5
MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
RUN_NAME=159_smoke_vagenlite_id74_k16_oneturn_base0_1g
RUN_OUT=${ROOT}/outputs/experiments/training/rl/2026-08-13/${RUN_NAME}
RUN_RESULT=${RUN_OUT}/one_turn_result.json
ENV_PORT=$((18300 + SLURM_JOB_ID % 500))
ENV_URL=http://127.0.0.1:${ENV_PORT}
RUNTIME_ROOT=/tmp/nvl159-${SLURM_JOB_ID}
RAY_TMPDIR=${RUNTIME_ROOT}/ray
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
NVIDIA_PID=
SMOKE_PID=
STARTED_AT=$(date --iso-8601=seconds)

ALLOCATED_NODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-${SLURM_STEP_NUM_NODES:-}}}
[[ "${ALLOCATED_NODES}" == "1" ]]
[[ "${SLURM_JOB_PARTITION:-}" == "normal" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == "16" ]]
[[ "${SLURM_MEM_PER_NODE:-}" == "131072" ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
IFS=, read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#ALLOCATED_GPUS[@]} == 1 )) || {
  echo "expected exactly one visible GPU, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
[[ "${GPU_NAME}" == *H800* ]] || {
  echo "expected one H800, got ${GPU_NAME}" >&2
  exit 2
}
JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}" -o)
grep -q "Partition=normal" <<<"${JOB_DETAILS}"
grep -q "TimeLimit=00:30:00" <<<"${JOB_DETAILS}"
grep -Eq "ReqTRES=[^ ]*mem=128G[^ ]*gres/gpu=1|ReqTRES=[^ ]*gres/gpu=1[^ ]*mem=128G" <<<"${JOB_DETAILS}"
[[ ! -e "${RUN_OUT}" ]] || {
  echo "refusing nonempty/reused output: ${RUN_OUT}" >&2
  exit 2
}
mkdir -p "${RUN_OUT}" "${RUNTIME_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}" "${AI2THOR_HOME_ROOT}/.ai2thor"
printf '%s\n' "${JOB_DETAILS}" >"${RUN_OUT}/allocation.txt"
printf '%s\n' "${GPU_NAME}" >"${RUN_OUT}/gpu_name.txt"

export PATH=${ROOT}/.venv-vagen-main/bin:/usr/bin:/bin
export PYTHONPATH=${REPO}/src:${VAGEN}:${VERL}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NIMLOTH_LATENT_TOKEN_COUNT=16
export RAY_TMPDIR TMPDIR AI2THOR_HOME_ROOT
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

terminate_owned_pid() {
  local pid=${1:-}
  [[ -n "${pid}" ]] || return 0
  kill "${pid}" >/dev/null 2>&1 || return 0
  for _ in $(seq 1 20); do
    kill -0 "${pid}" >/dev/null 2>&1 || { wait "${pid}" >/dev/null 2>&1 || true; return 0; }
    sleep 0.5
  done
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
}

terminate_owned_process_group() {
  local pgid=${1:-}
  [[ -n "${pgid}" ]] || return 0
  kill -TERM -- "-${pgid}" >/dev/null 2>&1 || return 0
  for _ in $(seq 1 20); do
    kill -0 -- "-${pgid}" >/dev/null 2>&1 || return 0
    sleep 0.5
  done
  kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
}

cleanup() {
  status=$?
  trap - EXIT
  set +e
  terminate_owned_process_group "${SMOKE_PID}"
  terminate_owned_pid "${NVIDIA_PID}"
  terminate_owned_pid "${ENV_PID}"
  ss -ltnp >"${RUN_OUT}/ports_after.log" 2>&1 || true
  pgrep -af "${RUNTIME_ROOT}|vagen.envs.navigation.serve.*${ENV_PORT}" >"${RUN_OUT}/owned_processes_after.log" 2>&1 || true
  if ss -ltnH "sport = :${ENV_PORT}" | grep -q .; then
    echo "owned environment port still listening after cleanup" >&2
    status=90
  fi
  if [[ -s "${RUN_OUT}/owned_processes_after.log" ]]; then
    echo "owned Ray/vLLM/environment processes remain after cleanup" >&2
    status=91
  fi
  "${PY}" - "${RUN_OUT}" "${status}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
out=Path(sys.argv[1]); status=int(sys.argv[2])
result=out/'one_turn_result.json'
payload={
  'finished_at': datetime.now(timezone.utc).isoformat(),
  'exit_code': status,
  'status': 'passed' if status == 0 and result.is_file() else 'failed',
  'result_exists': result.is_file(),
}
fd,name=tempfile.mkstemp(prefix='.final_status.',suffix='.tmp',dir=out)
try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:
    json.dump(payload,f,indent=2,allow_nan=False); f.write('\n')
  os.replace(name,out/'final_status.json')
except BaseException:
  try: os.unlink(name)
  except FileNotFoundError: pass
  raise
PY
  if [[ "${RUNTIME_ROOT}" == "/tmp/nvl159-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${RUNTIME_ROOT}"
  fi
  exit "${status}"
}
trap cleanup EXIT

{
  echo "experiment_id=159"
  echo "project=nimloth-rl"
  echo "run_name=${RUN_NAME}"
  echo "wandb_mode=disabled_by_design_no_training_metrics"
  echo "started_at=${STARTED_AT}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "node=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "gpu_name=${GPU_NAME}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK}"
  echo "slurm_mem_per_node_mib=${SLURM_MEM_PER_NODE}"
  echo "python=${PY}"
  echo "parent_worktree=${REPO}"
  echo "parent_commit=${PARENT_COMMIT}"
  echo "vagen_commit=${VAGEN_COMMIT}"
  echo "verl_commit=${VERL_COMMIT}"
  echo "checkpoint=${MODEL}"
  echo "checkpoint_use=HF_policy_weights_only"
  echo "sidecars_not_loaded=state_proj,wm_predictor,value_head"
  echo "data=external/VAGEN/vagen/envs/navigation/assets/base.json"
  echo "data_sha256=6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a"
  echo "split=heldout_base_seed0_FloorPlan11_Bread"
  echo "modules=all_frozen_inference_only"
  echo "optimizer=none"
  echo "joint_policy=false"
  echo "checkpoint_output=none"
  echo "resume=none_retry_requires_new_id_and_empty_output"
  echo "resources=normal_1node_1H800_16CPU_128GiB_30min_Unity_vLLM_same_gpu"
  echo "env_url=${ENV_URL}"
  echo "env_profile=current_prompt_nimloth_K16_step0.5_threshold1.5_success10_format0"
  echo "sampling=greedy_temp0_top_p1_response512_one_turn"
  echo "vllm_gpu_memory_utilization=0.6"
  echo "ray_tmp=${RAY_TMPDIR}"
} | tee "${RUN_OUT}/controller.log"

{
  cat <<EOF
# ID159 optimizer-free VAGEN-Lite one-turn smoke

- Goal: ID74 HF policy load -> real held-out Navigation reset -> model-generated CoT -> forced K16 protocol -> one sampled action -> one real environment step -> reward/decision-ledger validation.
- This is inference-only. It does not load StateProjector, WM predictor, ValueHead, frozen-Q guidance, actor/critic trainers, optimizer, FSDP, or checkpoints.
- Source: Nimloth ${PARENT_COMMIT}; VAGEN ${VAGEN_COMMIT}; VERL ${VERL_COMMIT}.
- Data: held-out base seed 0, FloorPlan11 / Bread; base asset SHA256 6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a; no overlap with base_train tasks or scenes.
- Checkpoint: ${MODEL}, HF policy weights only.
- Resource: normal, one node, one H800 shared by Unity and vLLM, 16 CPU, 128 GiB, 30 minutes.
- Resume: none. Existing output is never overwritten; any retry uses a new numeric ID and empty directory.
- W&B identity: project nimloth-rl, run name ${RUN_NAME}. W&B logging is disabled because the smoke has no training metrics; identity was checked unused before launch.
EOF
} >"${RUN_OUT}/README.md"

printf '%q ' "${PY}" -m vagen.standalone_one_turn_smoke \
  --model "${MODEL}" --env-url "${ENV_URL}" --output "${RUN_RESULT}" \
  --run-name "${RUN_NAME}" --agent-loop-config "${VAGEN}/vagen/configs/agent_no_concat.yaml" \
  --eval-set base --seed 0 --latent-token-count 16 --prompt-length 9000 \
  --response-length 512 --temperature 0 --top-p 1 --gpu-memory-utilization 0.6 \
  --env-timeout 300 >"${RUN_OUT}/command.txt"
printf '\n' >>"${RUN_OUT}/command.txt"

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${PARENT_COMMIT}" ]]
[[ "$(git -C "${VAGEN}" rev-parse HEAD)" == "${VAGEN_COMMIT}" ]]
[[ "$(git -C "${VERL}" rev-parse HEAD)" == "${VERL_COMMIT}" ]]
git -C "${REPO}" diff --quiet
git -C "${REPO}" diff --cached --quiet
git -C "${VAGEN}" diff --quiet
git -C "${VAGEN}" diff --cached --quiet
[[ -x "${PY}" && -f "${MODEL}/config.json" && -f "${MODEL}/model.safetensors.index.json" ]]
[[ -f "${MODEL}/model-00001-of-00002.safetensors" && -f "${MODEL}/model-00002-of-00002.safetensors" ]]
[[ -f "${MODEL}/state_proj.pt" && -f "${MODEL}/value_head/value_head.pt" && -f "${MODEL}/wm_predictor/predictor.pt" ]]
sha256sum -c - <<EOF | tee "${RUN_OUT}/checkpoint_sha256.log"
7b510cb37e16b11c1c00cf46365037943dfd61be1650ae18ce6a2e7b7e296690  ${MODEL}/config.json
32acf7bf413e8b87f295e816fe3d68c965e0ab196fbf30b32858b52df41cc97e  ${MODEL}/model.safetensors.index.json
be27f6714980c2cf8f63e9f119a7fd3b055709e7f67c9523da108595f143eca3  ${MODEL}/tokenizer.json
63c933b6ebadae3ee64a4663b5bd1ec71676f64629faf2cda6c15393e534e563  ${MODEL}/model-00001-of-00002.safetensors
1939ec8a9b041c8142acdf5ac4043243ed018360a000420e2a711b9bea5000  ${MODEL}/model-00002-of-00002.safetensors
e789a67246022c785521324bbd800d903f46024d8e8d05c504fcbcdedd9d4063  ${MODEL}/state_proj.pt
b0059ba1eb842cedcbba884dff88a67cd2da127583cea14a800f4215d835c87d  ${MODEL}/value_head/value_head.pt
85cedd95e5fc6d89cdad7248a85e2dd51b10e1dcf8302d19d5cd3b489af82bb8  ${MODEL}/wm_predictor/predictor.pt
EOF

"${PY}" - "${MODEL}" "${VAGEN}" <<'PY' | tee "${RUN_OUT}/checkpoint_preflight.json"
import hashlib, json, sys
from pathlib import Path
model, vagen = Path(sys.argv[1]), Path(sys.argv[2])
cfg=json.loads((model/'config.json').read_text())
idx=json.loads((model/'model.safetensors.index.json').read_text())
shards=sorted(set(idx['weight_map'].values()))
assert cfg['model_type']=='qwen2_5_vl'
assert cfg['nimloth_latent_query_mode']=='inject'
assert cfg['nimloth_latent_token_count']==16
assert cfg['nimloth_query_tune']=='freeze'
assert len(shards)==2 and all((model/s).is_file() and (model/s).stat().st_size>0 for s in shards)
asset=vagen/'vagen/envs/navigation/assets/base.json'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(asset)=='6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a'
data=json.loads(asset.read_text())['tasks']
task=data[0]
assert (task['scene'],task['targetObjectType'])==('FloorPlan11','Bread')
print(json.dumps({'status':'CHECKPOINT_PREFLIGHT_OK','model_type':cfg['model_type'],'latent_mode':cfg['nimloth_latent_query_mode'],'latent_count':cfg['nimloth_latent_token_count'],'query_tune':cfg['nimloth_query_tune'],'shards':shards,'asset_sha256':sha(asset),'task':{'scene':task['scene'],'target':task['targetObjectType'],'instruction':task['instruction']}}))
PY

# Keep a private AI2-THOR mapping file while sharing only immutable Unity releases.
ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases "${AI2THOR_HOME_ROOT}/.ai2thor/releases"
rm -f "${AI2THOR_HOME_ROOT}/.ai2thor/cuda-vulkan-mapping.json"
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh" \
  > >(tee -a "${RUN_OUT}/controller.log") 2>&1

# Direct render proves the allocated visible ordinal before the remote service starts.
timeout --signal=TERM --kill-after=10s 150s "${PY}" -m nimloth.environment.navigation.direct_render_probe \
  --gpu-device 0 | tee "${RUN_OUT}/render_probe.json"

cd "${VAGEN}"
if ss -ltnH "sport = :${ENV_PORT}" | grep -q .; then
  echo "refusing occupied environment port ${ENV_PORT}" >&2
  exit 4
fi
"${PY}" -m vagen.envs.navigation.serve \
  --host=127.0.0.1 --port="${ENV_PORT}" --devices='[0]' \
  --max_envs=1 --max_inflight=1 --thread_pool_size=1 --session_timeout=900 \
  >"${RUN_OUT}/env_server.log" 2>&1 &
ENV_PID=$!
ready=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 5 "${ENV_URL}/health" >"${RUN_OUT}/health.json" 2>/dev/null; then
    kill -0 "${ENV_PID}" >/dev/null 2>&1 || { echo "health came from a foreign service" >&2; exit 4; }
    LISTENER_PID=$(ss -ltnpH "sport = :${ENV_PORT}" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | head -1)
    [[ "${LISTENER_PID}" == "${ENV_PID}" ]] || {
      echo "environment listener pid ${LISTENER_PID:-missing} != owned pid ${ENV_PID}" >&2
      exit 4
    }
    ready=1
    break
  fi
  kill -0 "${ENV_PID}" >/dev/null 2>&1 || {
    tail -200 "${RUN_OUT}/env_server.log" >&2 || true
    exit 4
  }
  sleep 2
done
(( ready == 1 )) || { echo "environment health timeout" >&2; exit 4; }

# Full create/reset/prompt/close wall clock is bounded to 300 seconds.
timeout --signal=TERM --kill-after=10s 300s "${PY}" -m nimloth.environment.navigation.prewarm \
  --env-url "${ENV_URL}" --eval-set base --seed 0 --timeout-seconds 300 \
  --env-id "nvl159-prewarm-${SLURM_JOB_ID}" | tee "${RUN_OUT}/prewarm.json"

# Record allocation-owned peak VRAM throughout vLLM load and one real turn.
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 1 >"${RUN_OUT}/nvidia_smi.csv" 2>"${RUN_OUT}/nvidia_smi.err" &
NVIDIA_PID=$!

setsid "${PY}" -m vagen.standalone_one_turn_smoke \
  --model "${MODEL}" --env-url "${ENV_URL}" --output "${RUN_RESULT}" \
  --run-name "${RUN_NAME}" --agent-loop-config "${VAGEN}/vagen/configs/agent_no_concat.yaml" \
  --eval-set base --seed 0 --latent-token-count 16 --prompt-length 9000 \
  --response-length 512 --temperature 0 --top-p 1 --gpu-memory-utilization 0.6 \
  --env-timeout 300 >"${RUN_OUT}/smoke.log" 2>&1 &
SMOKE_PID=$!
set +e
wait "${SMOKE_PID}"
SMOKE_STATUS=$?
set -e
cat "${RUN_OUT}/smoke.log"
terminate_owned_process_group "${SMOKE_PID}"
SMOKE_PID=
(( SMOKE_STATUS == 0 )) || exit "${SMOKE_STATUS}"

"${PY}" - "${RUN_RESULT}" <<'PY' | tee "${RUN_OUT}/validator.json"
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text())
assert x['status']=='passed'
assert x['optimizer'] is None and x['checkpoint_output'] is None
assert x['eval_set']=='base' and x['seed']==0 and x['latent_token_count']==16
ledger=x['decision_ledger']
assert ledger['action_space']=='navigation_v1'
assert ledger['decision_sources']==['llm_text']
assert ledger['decision_is_policy_sampled']==[False]
assert ledger['format_valid'] is True
assert len(ledger['executed_action_ids'])==1
assert math.isfinite(float(x['env_turn_reward']))
assert '<think>' in x['environment_response']
assert '<|action_start|>' in x['environment_response']
assert '<|action_end|>' in x['environment_response']
print(json.dumps({'status':'ALL_OK','response_tokens':x['response_token_count'],'action_id':ledger['executed_action_ids'][0],'reward':x['env_turn_reward'],'reward_anchor_index':x['reward_anchor_index']}))
PY
