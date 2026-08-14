#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO must be the pinned remote worktree}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
VAGEN=${REPO}/external/VAGEN
VERL=${VAGEN}/verl
PY=${ROOT}/.venv-vagen-main/bin/python3
PARENT_COMMIT=${EXPECTED_PARENT_COMMIT}
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
VAGEN_COMMIT=${EXPECTED_VAGEN_COMMIT}
VERL_COMMIT=084f042b71b8fe03785a279cf227f4085def0391
VERL_BASE_COMMIT=3fe0a29975e1b02ae2bd1dec249f7807dd7966f5
MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
EXPERIMENT_ID=${EXPERIMENT_ID:?EXPERIMENT_ID is required}
RUN_NAME=${RUN_NAME:?RUN_NAME is required}
RUN_DATE=${RUN_DATE:?RUN_DATE is required}
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}
RUN_RESULT=${RUN_OUT}/one_turn_result.json
ENV_PORT=$((18300 + SLURM_JOB_ID % 500))
ENV_URL=http://127.0.0.1:${ENV_PORT}
RUNTIME_ROOT=/tmp/nvl${EXPERIMENT_ID}-${SLURM_JOB_ID}
RAY_TMPDIR=${RUNTIME_ROOT}/ray
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
NVIDIA_PID=
SMOKE_PID=
STARTED_AT=$(date --iso-8601=seconds)
EXPECTED_HOLD_GPUS=${EXPECTED_HOLD_GPUS:-8}
EXPECTED_STEP_GPUS=${EXPECTED_STEP_GPUS:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
EXPECTED_HOLD_WALLTIME=${EXPECTED_HOLD_WALLTIME:-00:45:00}
SMOKE_TIMEOUT_SECONDS=${SMOKE_TIMEOUT_SECONDS:-1500}
EVIDENCE_MODE=${EVIDENCE_MODE:-tp8_gate}
GUIDED_MODE=false
JOINT_ALPHA=${JOINT_ALPHA:-}
JOINT_BETA=${JOINT_BETA:-}
JOINT_PRIOR_TEMPERATURE=${JOINT_PRIOR_TEMPERATURE:-}
JOINT_SCORE_DTYPE=${JOINT_SCORE_DTYPE:-}
JOINT_RUN_SEED=${JOINT_RUN_SEED:-}
JOINT_SNAPSHOT_SOURCE_STEP=${JOINT_SNAPSHOT_SOURCE_STEP:-}

[[ "${EXPECTED_HOLD_GPUS}" =~ ^[1-9][0-9]*$ ]]
[[ "${EXPECTED_STEP_GPUS}" =~ ^[1-9][0-9]*$ ]]
[[ "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]
[[ "${SMOKE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]
(( EXPECTED_STEP_GPUS <= EXPECTED_HOLD_GPUS ))
(( TENSOR_PARALLEL_SIZE == EXPECTED_STEP_GPUS ))
case "${EVIDENCE_MODE}" in
  tp8_gate)
    [[ "${EXPERIMENT_ID}" == "161" ]]
    [[ "${EXPECTED_HOLD_GPUS}:${EXPECTED_STEP_GPUS}:${TENSOR_PARALLEL_SIZE}:${EXPECTED_HOLD_WALLTIME}:${SMOKE_TIMEOUT_SECONDS}" == "8:8:8:00:45:00:1500" ]]
    ;;
  tp4_interim_diagnostic)
    [[ "${EXPERIMENT_ID}" == "162" ]]
    [[ "${EXPECTED_HOLD_GPUS}:${EXPECTED_STEP_GPUS}:${TENSOR_PARALLEL_SIZE}:${EXPECTED_HOLD_WALLTIME}:${SMOKE_TIMEOUT_SECONDS}" == "7:4:4:00:20:00:900" ]]
    ;;
  guided_tp8_gate)
    [[ "${EXPERIMENT_ID}" == "163" || "${EXPERIMENT_ID}" == "164" ]]
    [[ "${EXPECTED_HOLD_GPUS}:${EXPECTED_STEP_GPUS}:${TENSOR_PARALLEL_SIZE}:${EXPECTED_HOLD_WALLTIME}:${SMOKE_TIMEOUT_SECONDS}" == "8:8:8:00:45:00:1500" ]]
    [[ "${JOINT_ALPHA}:${JOINT_BETA}:${JOINT_PRIOR_TEMPERATURE}:${JOINT_SCORE_DTYPE}:${JOINT_RUN_SEED}:${JOINT_SNAPSHOT_SOURCE_STEP}" == "1:1:1:float32:42:776" ]]
    GUIDED_MODE=true
    ;;
  *) echo "unsupported evidence mode: ${EVIDENCE_MODE}" >&2; exit 2 ;;
esac
export EVIDENCE_MODE GUIDED_MODE JOINT_ALPHA JOINT_BETA
export JOINT_PRIOR_TEMPERATURE JOINT_SCORE_DTYPE JOINT_RUN_SEED
export JOINT_SNAPSHOT_SOURCE_STEP
SMOKE_EXTRA_ARGS=()
if [[ "${GUIDED_MODE}" == true ]]; then
  SMOKE_EXTRA_ARGS+=(
    --guided
    --critic-checkpoint "${MODEL}"
    --critic-qwen-hidden-dim 2048
    --critic-state-dim 1024
    --joint-alpha "${JOINT_ALPHA}"
    --joint-beta "${JOINT_BETA}"
    --joint-prior-temperature "${JOINT_PRIOR_TEMPERATURE}"
    --joint-score-dtype "${JOINT_SCORE_DTYPE}"
    --joint-run-seed "${JOINT_RUN_SEED}"
    --joint-snapshot-source-step "${JOINT_SNAPSHOT_SOURCE_STEP}"
  )
fi

ALLOCATED_NODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-${SLURM_STEP_NUM_NODES:-}}}
[[ "${ALLOCATED_NODES}" == "1" ]]
[[ "${SLURM_JOB_PARTITION:-}" == "normal" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == "64" ]]
[[ "${SLURM_MEM_PER_NODE:-}" == "262144" ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
IFS=, read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#ALLOCATED_GPUS[@]} == EXPECTED_STEP_GPUS )) || {
  echo "expected exactly ${EXPECTED_STEP_GPUS} visible GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
}
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == EXPECTED_STEP_GPUS )) || {
  echo "expected ${EXPECTED_STEP_GPUS} allocated GPUs, got ${#GPU_NAMES[@]}" >&2
  exit 2
}
for name in "${GPU_NAMES[@]}"; do
  [[ "${name}" == *H800* ]] || {
    echo "expected only H800 GPUs, got ${name}" >&2
    exit 2
  }
done
GPU_NAME="${EXPECTED_STEP_GPUS}xNVIDIA H800"
JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}" -o)
grep -q "Partition=normal" <<<"${JOB_DETAILS}"
grep -q "TimeLimit=${EXPECTED_HOLD_WALLTIME}" <<<"${JOB_DETAILS}"
grep -Eq "ReqTRES=[^ ]*cpu=64([, ]|$)" <<<"${JOB_DETAILS}"
grep -Eq "AllocTRES=[^ ]*cpu=64([, ]|$)" <<<"${JOB_DETAILS}"
grep -Eq "ReqTRES=[^ ]*mem=256G[^ ]*gres/gpu=${EXPECTED_HOLD_GPUS}|ReqTRES=[^ ]*gres/gpu=${EXPECTED_HOLD_GPUS}[^ ]*mem=256G" <<<"${JOB_DETAILS}"
grep -Eq "AllocTRES=[^ ]*gres/gpu=${EXPECTED_HOLD_GPUS}([, ]|$)" <<<"${JOB_DETAILS}"
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
export PYTHONDONTWRITEBYTECODE=1
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
  'evidence_mode': os.environ['EVIDENCE_MODE'],
  'guided': os.environ['GUIDED_MODE'] == 'true',
  'does_not_substitute_for_tp8_gate': os.environ['EVIDENCE_MODE'] == 'tp4_interim_diagnostic',
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
  if [[ "${RUNTIME_ROOT}" == "/tmp/nvl${EXPERIMENT_ID}-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${RUNTIME_ROOT}"
  fi
  exit "${status}"
}
trap cleanup EXIT

{
  echo "experiment_id=${EXPERIMENT_ID}"
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
  echo "verl_base_commit=${VERL_BASE_COMMIT}"
  echo "checkpoint=${MODEL}"
  if [[ "${GUIDED_MODE}" == true ]]; then
    echo "checkpoint_use=HF_policy_plus_frozen_state_proj_and_value_head"
    echo "sidecars_loaded=state_proj,value_head"
    echo "sidecars_not_loaded=wm_predictor"
    echo "capture=async_same_generation_K16x2048_raw8_CPU_frozen_Q_guided_TP${TENSOR_PARALLEL_SIZE}_mm_encoder_data"
  else
    echo "checkpoint_use=HF_policy_weights_only"
    echo "sidecars_not_loaded=state_proj,wm_predictor,value_head"
    echo "capture=async_same_generation_K16x2048_raw8_TP${TENSOR_PARALLEL_SIZE}_mm_encoder_data"
  fi
  echo "evidence_mode=${EVIDENCE_MODE}"
  echo "does_not_substitute_for_tp8_gate=$([[ "${EVIDENCE_MODE}" == tp4_interim_diagnostic ]] && echo true || echo false)"
  echo "data=external/VAGEN/vagen/envs/navigation/assets/base.json"
  echo "data_sha256=6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a"
  echo "split=heldout_base_seed0_FloorPlan11_Bread"
  echo "modules=all_frozen_inference_only"
  echo "optimizer=none"
  echo "joint_policy=${GUIDED_MODE}"
  if [[ "${GUIDED_MODE}" == true ]]; then
    echo "joint_policy_config=alpha${JOINT_ALPHA}_beta${JOINT_BETA}_priorT${JOINT_PRIOR_TEMPERATURE}_${JOINT_SCORE_DTYPE}"
    echo "joint_draw_run_seed=${JOINT_RUN_SEED}"
    echo "joint_snapshot_source_step=${JOINT_SNAPSHOT_SOURCE_STEP}"
    echo "joint_snapshot_refresh=none_optimizer_free_one_batch"
  fi
  echo "checkpoint_output=none"
  echo "resume=none_retry_requires_new_id_and_empty_output"
  echo "resources=normal_1node_hold${EXPECTED_HOLD_GPUS}H800_step${EXPECTED_STEP_GPUS}H800_64CPU_256GiB_${EXPECTED_HOLD_WALLTIME}_Unity_vLLM_TP${TENSOR_PARALLEL_SIZE}"
  echo "env_url=${ENV_URL}"
  echo "env_profile=current_prompt_nimloth_K16_step0.5_threshold1.5_success10_format0"
  echo "sampling=greedy_temp0_top_p1_response512_one_turn_TP${TENSOR_PARALLEL_SIZE}"
  echo "vllm_gpu_memory_utilization=0.6"
  echo "ray_tmp=${RAY_TMPDIR}"
} | tee "${RUN_OUT}/controller.log"

if [[ "${GUIDED_MODE}" == true ]]; then
  cat >"${RUN_OUT}/README.md" <<EOF
# ID${EXPERIMENT_ID} optimizer-free guided VAGEN-Lite one-turn smoke

- Goal: ID74 policy plus frozen StateProjector/ValueHead -> held-out Navigation reset -> real model CoT and K16 same-generation capture -> CPU frozen all-action Q -> manager-keyed Scheme-B draw -> response/behavior authorization -> one explicit guided environment action -> reward/ledger/DataProto validation.
- This is inference-only. All model and critic modules are frozen; there is no actor/critic optimizer, backward, update, FSDP, snapshot refresh, training checkpoint, or W&B metric stream.
- Scheme-B diagnostic contract: alpha=${JOINT_ALPHA}, beta=${JOINT_BETA}, prior_temperature=${JOINT_PRIOR_TEMPERATURE}, score_dtype=${JOINT_SCORE_DTYPE}, draw run seed=${JOINT_RUN_SEED}, snapshot source step=${JOINT_SNAPSHOT_SOURCE_STEP}. These protocol-gate values are not a training hyperparameter conclusion.
- Source: Nimloth ${PARENT_COMMIT}; VAGEN ${VAGEN_COMMIT}; VERL ${VERL_COMMIT} (direct parent ${VERL_BASE_COMMIT}).
- Data: held-out base seed 0, FloorPlan11 / Bread; base asset SHA256 6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a; no overlap with base_train tasks or scenes.
- Checkpoint: ${MODEL}; HF policy, state_proj.pt, and value_head are loaded read-only. wm_predictor is not loaded.
- Resource: normal, one node, hold owns ${EXPECTED_HOLD_GPUS} H800 and this step owns ${EXPECTED_STEP_GPUS}; Unity uses step-visible ordinal 0, vLLM uses TP${TENSOR_PARALLEL_SIZE}, and the frozen-Q Ray actor uses one CPU/zero GPU. Allocation is 64 CPU, 256 GiB, walltime ${EXPECTED_HOLD_WALLTIME}; expected execution is about 10--15 minutes based on ID161.
- Resume: none. Existing output is never overwritten; any retry uses a new numeric ID and empty directory.
- W&B identity: project nimloth-rl, run name ${RUN_NAME}. The project currently has no matching run; W&B logging stays disabled because this smoke has no training metrics.
EOF
else
  cat >"${RUN_OUT}/README.md" <<EOF
# ID${EXPERIMENT_ID} optimizer-free VAGEN-Lite one-turn smoke

- Goal: ID74 HF policy load -> real held-out Navigation reset -> model-generated CoT -> forced K16 protocol -> same-generation K16 hidden/raw 8-action logit capture -> one sampled action -> one real environment step -> reward/decision-ledger validation.
- This is inference-only. It does not load StateProjector, WM predictor, ValueHead, frozen-Q guidance, actor/critic trainers, optimizer, FSDP, or checkpoints; capture does not alter the environment action.
- Evidence mode: ${EVIDENCE_MODE}. A TP4 interim diagnostic does not substitute for the pending ID161 TP8 gate.
- Source: Nimloth ${PARENT_COMMIT}; VAGEN ${VAGEN_COMMIT}; VERL ${VERL_COMMIT} (direct parent ${VERL_BASE_COMMIT}).
- Data: held-out base seed 0, FloorPlan11 / Bread; base asset SHA256 6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a; no overlap with base_train tasks or scenes.
- Checkpoint: ${MODEL}, HF policy weights only.
- Resource: normal, one node, hold owns ${EXPECTED_HOLD_GPUS} H800 and this step owns ${EXPECTED_STEP_GPUS}; Unity uses step-visible ordinal 0 and vLLM uses TP${TENSOR_PARALLEL_SIZE}, with 64 CPU, 256 GiB, walltime ${EXPECTED_HOLD_WALLTIME}.
- Resume: none. Existing output is never overwritten; any retry uses a new numeric ID and empty directory.
- W&B identity: project nimloth-rl, run name ${RUN_NAME}. W&B logging is disabled because the smoke has no training metrics; identity was checked unused before launch.
EOF
fi

SMOKE_COMMAND=(
  "${PY}" -m vagen.standalone_one_turn_smoke
  --model "${MODEL}" --env-url "${ENV_URL}" --output "${RUN_RESULT}"
  --run-name "${RUN_NAME}" --agent-loop-config "${VAGEN}/vagen/configs/agent_no_concat.yaml"
  --eval-set base --seed 0 --latent-token-count 16 --prompt-length 9000
  --response-length 512 --temperature 0 --top-p 1 --gpu-memory-utilization 0.6
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" --env-timeout 300
)
if [[ "${GUIDED_MODE}" == true ]]; then
  SMOKE_COMMAND+=("${SMOKE_EXTRA_ARGS[@]}")
fi
printf '%q ' "${SMOKE_COMMAND[@]}" >"${RUN_OUT}/command.txt"
printf '\n' >>"${RUN_OUT}/command.txt"

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${PARENT_COMMIT}" ]]
[[ "$(git -C "${VAGEN}" rev-parse HEAD)" == "${VAGEN_COMMIT}" ]]
[[ "$(git -C "${VERL}" rev-parse HEAD)" == "${VERL_COMMIT}" ]]
[[ "$(git -C "${VERL}" rev-parse HEAD^)" == "${VERL_BASE_COMMIT}" ]]
EXPECTED_LEWM_COMMIT=$(git -C "${REPO}" ls-tree HEAD external/le-wm | awk '{print $3}')
[[ "${EXPECTED_LEWM_COMMIT}" == "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac" ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == "${EXPECTED_LEWM_COMMIT}" ]]
[[ -f "${REPO}/external/le-wm/module.py" ]]
git -C "${REPO}/external/le-wm" diff --quiet
git -C "${REPO}/external/le-wm" diff --cached --quiet
[[ -z "$(git -C "${REPO}/external/le-wm" status --porcelain --untracked-files=all)" ]]
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
1939ec8a9b041c8142acdf5acac4043243ed018360a000420e2a711b9bea5000  ${MODEL}/model-00002-of-00002.safetensors
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
"${PY}" - <<'PY' | tee "${RUN_OUT}/runtime_import_preflight.json"
from nimloth.backbone.qwen25vl.policy import reasoning_forbidden_token_ids
from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
from nimloth.wm._vendor_lewm import ARPredictor
print('{"status":"RUNTIME_IMPORT_OK","imports":["reasoning_forbidden_token_ids","TurnGenerationSpec","ARPredictor"]}')
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
  --env-id "nvl${EXPERIMENT_ID}-prewarm-${SLURM_JOB_ID}" | tee "${RUN_OUT}/prewarm.json"

# Record allocation-owned peak VRAM throughout vLLM load and one real turn.
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 1 >"${RUN_OUT}/nvidia_smi.csv" 2>"${RUN_OUT}/nvidia_smi.err" &
NVIDIA_PID=$!

setsid timeout --signal=TERM --kill-after=20s "${SMOKE_TIMEOUT_SECONDS}s" \
  "${SMOKE_COMMAND[@]}" >"${RUN_OUT}/smoke.log" 2>&1 &
SMOKE_PID=$!
set +e
wait "${SMOKE_PID}"
SMOKE_STATUS=$?
set -e
cat "${RUN_OUT}/smoke.log"
terminate_owned_process_group "${SMOKE_PID}"
SMOKE_PID=
(( SMOKE_STATUS == 0 )) || exit "${SMOKE_STATUS}"

"${PY}" - "${RUN_RESULT}" "${EVIDENCE_MODE}" "${EXPECTED_HOLD_GPUS}" \
  "${EXPECTED_STEP_GPUS}" "${TENSOR_PARALLEL_SIZE}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
p=Path(sys.argv[1])
x=json.loads(p.read_text())
x['launch_evidence']={
  'mode': sys.argv[2],
  'hold_gpu_count': int(sys.argv[3]),
  'step_gpu_count': int(sys.argv[4]),
  'tensor_parallel_size': int(sys.argv[5]),
  'guided': os.environ['GUIDED_MODE'] == 'true',
  'does_not_substitute_for_tp8_gate': sys.argv[2] == 'tp4_interim_diagnostic',
}
if x['launch_evidence']['guided']:
  x['launch_evidence']['joint_policy']={
    'alpha': float(os.environ['JOINT_ALPHA']),
    'beta': float(os.environ['JOINT_BETA']),
    'prior_temperature': float(os.environ['JOINT_PRIOR_TEMPERATURE']),
    'score_dtype': os.environ['JOINT_SCORE_DTYPE'],
    'run_seed': int(os.environ['JOINT_RUN_SEED']),
    'snapshot_source_step': int(os.environ['JOINT_SNAPSHOT_SOURCE_STEP']),
  }
fd,name=tempfile.mkstemp(prefix='.one_turn_result.',suffix='.tmp',dir=p.parent)
try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:
    json.dump(x,f,indent=2,allow_nan=False); f.write('\n')
  os.replace(name,p)
except BaseException:
  try: os.unlink(name)
  except FileNotFoundError: pass
  raise
PY

"${PY}" - "${RUN_RESULT}" <<'PY' | tee "${RUN_OUT}/validator.json"
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text())
assert x['status']=='passed'
assert x['optimizer'] is None and x['checkpoint_output'] is None
assert x['eval_set']=='base' and x['seed']==0 and x['latent_token_count']==16
ledger=x['decision_ledger']
state=x['policy_state']
evidence=x['launch_evidence']
guided=evidence['mode']=='guided_tp8_gate'
assert evidence['mode'] in {'tp8_gate','tp4_interim_diagnostic','guided_tp8_gate'}
assert evidence['step_gpu_count']==evidence['tensor_parallel_size']
if guided:
  assert evidence['tensor_parallel_size']==8
assert evidence['guided']==guided and x['guided']==guided
assert evidence['does_not_substitute_for_tp8_gate']==(evidence['mode']=='tp4_interim_diagnostic')
assert state['schema']=='nimloth_policy_state_v2'
assert len(state['latent_token_ids'])==16 and len(set(state['latent_token_ids']))==16
assert state['action_start_token_id'] not in state['latent_token_ids']
assert len(state['action_token_ids'])==8 and len(set(state['action_token_ids']))==8
assert len(state['latent_hidden'])==16
assert all(len(row)==2048 and all(math.isfinite(float(v)) for v in row) for row in state['latent_hidden'])
assert len(state['action_logits'])==8 and all(math.isfinite(float(v)) for v in state['action_logits'])
assert isinstance(state['request_id'],str) and state['request_id']
assert isinstance(state['generation_id'],str) and state['generation_id']
assert state['generation_id']!=state['request_id']
assert ledger['action_space']=='navigation_v1'
assert ledger['format_valid'] is True
assert len(ledger['executed_action_ids'])==1
assert math.isfinite(float(x['env_turn_reward']))
assert '<think>' in x['environment_response']
assert '<|action_start|>' in x['environment_response']
assert '<|action_end|>' in x['environment_response']
prior_action_id=ledger['executed_action_ids'][0]
if guided:
  assert ledger['schema']=='vagen_decision_ledger_v2_frozen_q_guided'
  assert ledger['decision_sources']==['frozen_q_guided']
  assert ledger['decision_is_policy_sampled']==[True]
  pin=x['joint_policy_batch_pin']
  score=x['frozen_q_scoring']
  trace=x['policy_response_trace']
  draw=x['guided_action_draw']
  execution=x['guided_action_execution']
  behavior=ledger['behavior_record']
  key=draw['draw_key']
  config=draw['policy_config']
  assert x['guided_turn_index']==0 and key['turn_index']==0
  assert pin['schema']=='nimloth_frozen_q_batch_pin_v1'
  assert pin['policy_step']==0 and pin['snapshot_source_step']==776
  assert pin['snapshot_id']==score['snapshot_id']==draw['draw_key']['snapshot_id']==ledger['snapshot_id']
  assert pin['contract_id']==score['contract_id']==draw['contract_id']==ledger['contract_id']
  assert score['schema']=='nimloth_frozen_q_scoring_v1'
  assert score['snapshot_source_step']==776 and score['score_dtype']=='float32'
  assert score['request_id']==trace['request_id']==state['request_id']
  assert score['generation_id']==trace['generation_id']==state['generation_id']
  assert score['action_token_ids']==state['action_token_ids']==draw['action_token_ids']
  assert len(score['prior_logits'])==len(score['frozen_all_action_q'])==8
  assert all(math.isfinite(float(v)) for v in score['prior_logits']+score['frozen_all_action_q'])
  assert all(math.isclose(float(a),float(b),rel_tol=0.0,abs_tol=1e-6) for a,b in zip(score['prior_logits'],state['action_logits'],strict=True))
  assert trace['schema']=='nimloth_policy_response_trace_v1'
  assert trace['raw_response']==x['environment_response']
  assert len(trace['response_ids'])==len(trace['response_mask'])==len(trace['response_logprobs'])==x['response_token_count']
  assert draw['schema']=='vagen_frozen_q_guided_action_draw_v2'
  assert key['schema']=='vagen_guided_action_draw_key_v1'
  assert key['run_seed']==42 and key['policy_step']==0
  assert key['rollout_sample_id']=='standalone:navigation:base:0'
  assert key['rollout_repeat_index']==0 and key['is_validation'] is True
  assert config=={'implementation':'frozen_q_guided_v1','alpha':1.0,'beta':1.0,'prior_temperature':1.0,'backprop_to_llm':True,'score_dtype':'float32'}
  assert draw['prior_logits']==score['prior_logits']
  assert draw['frozen_all_action_q']==score['frozen_all_action_q']
  assert 0.0<=float(draw['uniform_draw'])<1.0
  assert draw['guided_action_id']==behavior['guided_action_id']==ledger['executed_action_ids'][0]
  assert execution['schema']=='vagen_guided_action_execution_v3'
  assert execution['behavior_record']==behavior
  assert execution['behavior_record_id']==ledger['behavior_record_id']
  assert behavior['snapshot_id']==pin['snapshot_id']
  assert behavior['frozen_all_action_q']==score['frozen_all_action_q']
  prior_action_id=behavior['prior_action_id']
  assert evidence['joint_policy']=={'alpha':1.0,'beta':1.0,'prior_temperature':1.0,'score_dtype':'float32','run_seed':42,'snapshot_source_step':776}
else:
  assert ledger['schema']=='vagen_decision_ledger_v1'
  assert ledger['decision_sources']==['llm_text']
  assert ledger['decision_is_policy_sampled']==[False]
print(json.dumps({'status':'ALL_OK','evidence_mode':evidence['mode'],'guided':guided,'does_not_substitute_for_tp8_gate':evidence['does_not_substitute_for_tp8_gate'],'response_tokens':x['response_token_count'],'prior_action_id':prior_action_id,'executed_action_id':ledger['executed_action_ids'][0],'reward':x['env_turn_reward'],'reward_anchor_index':x['reward_anchor_index'],'policy_state_shape':[len(state['latent_hidden']),len(state['latent_hidden'][0])],'action_logits_shape':[len(state['action_logits'])],'request_id':state['request_id'],'generation_id':state['generation_id']}))
PY
