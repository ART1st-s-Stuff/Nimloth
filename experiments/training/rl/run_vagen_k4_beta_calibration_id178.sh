#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${RUN_DATE:?RUN_DATE is required}"
VAGEN=${REPO}/external/VAGEN
VERL=${VAGEN}/verl
PY=${ROOT}/.venv-vagen-main/bin/python3
REPAIR_ROOT=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8
MODEL=${REPAIR_ROOT}/checkpoint
PLANNING_MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}
RUN_META=${RUN_OUT}.metadata.md
CONTROL=${ROOT}/outputs/experiments/training/rl/slurm/id178-k4-${SLURM_JOB_ID}
ENV_PORT=$((19400 + SLURM_JOB_ID % 400))
ENV_URL=http://127.0.0.1:${ENV_PORT}
RUNTIME_ROOT=/tmp/i178-${SLURM_JOB_ID}
RAY_TMPDIR=${RUNTIME_ROOT}
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
CALIBRATION_PID=
NVIDIA_PID=
CALIBRATION_TIMEOUT_SECONDS=2700

[[ "${RUN_NAME}" == 178_calibration_k4mcts_tp8_actionrepair176_train3x8_t20_s100_c1_a1_b0_t1_cot07p095 ]]
[[ "${RUN_DATE}" == 2026-08-15 ]]
[[ "${EXPECTED_VERL_COMMIT}" == 494f264494b2525f2c13595f63ac4912963e6d2f ]]
[[ "${SLURM_JOB_PARTITION:-}" == normal ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 64 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 262144 ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
IFS=, read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
(( ${#VISIBLE_GPUS[@]} == 8 ))
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == 8 ))
for name in "${GPU_NAMES[@]}"; do [[ "${name}" == *H800* ]]; done
for excluded in dgx-13 dgx-23 dgx-32 dgx-37 dgx-51; do
  [[ "$(hostname)" != "${excluded}" ]]
done

JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}" -o)
grep -q 'Partition=normal' <<<"${JOB_DETAILS}"
grep -q 'TimeLimit=01:00:00' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*mem=256G([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -q 'MinMemoryNode=256G' <<<"${JOB_DETAILS}"

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_PARENT_COMMIT}" ]]
[[ "$(git -C "${VAGEN}" rev-parse HEAD)" == "${EXPECTED_VAGEN_COMMIT}" ]]
[[ "$(git -C "${VERL}" rev-parse HEAD)" == "${EXPECTED_VERL_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac ]]
[[ "$(git -C "${REPO}/external/RCDM" rev-parse HEAD)" == 71daaf10a73bb2012864f0827c68d209fc92b0a5 ]]
for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
[[ -x "${PY}" ]]
[[ -d "${MODEL}" ]]
[[ -d "${PLANNING_MODEL}" ]]
[[ -f "${PLANNING_MODEL}/training_state.pt" ]]
[[ -f "${REPAIR_ROOT}/complete.marker" ]]
[[ ! -e "${RUN_OUT}" ]] || { echo "ID178 output already exists" >&2; exit 2; }
[[ ! -e "${RUN_META}" ]] || { echo "ID178 metadata already exists" >&2; exit 2; }
[[ ! -e "${CONTROL}" ]] || { echo "ID178 control output already exists" >&2; exit 2; }
mkdir -p "${CONTROL}" "$(dirname "${RUN_OUT}")" "${RUNTIME_ROOT}" "${TMPDIR}" "${AI2THOR_HOME_ROOT}/.ai2thor"
printf '%s\n' "${JOB_DETAILS}" >"${CONTROL}/allocation.txt"

export PATH=${ROOT}/.venv-vagen-main/bin:/usr/bin:/bin
export PYTHONPATH=${REPO}/src:${VAGEN}:${VERL}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NIMLOTH_LATENT_TOKEN_COUNT=16
export RAY_TMPDIR TMPDIR AI2THOR_HOME_ROOT
export WANDB_MODE=disabled
export WANDB_PROJECT=vagen
export WANDB_NAME=${RUN_NAME}
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

terminate_pid() {
  local pid=${1:-}
  [[ -n "${pid}" ]] || return 0
  kill -TERM "${pid}" >/dev/null 2>&1 || return 0
  for _ in $(seq 1 30); do
    kill -0 "${pid}" >/dev/null 2>&1 || { wait "${pid}" >/dev/null 2>&1 || true; return 0; }
    sleep 1
  done
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
}

terminate_group() {
  local pid=${1:-}
  [[ -n "${pid}" ]] || return 0
  kill -TERM -- "-${pid}" >/dev/null 2>&1 || return 0
  for _ in $(seq 1 30); do
    kill -0 -- "-${pid}" >/dev/null 2>&1 || return 0
    sleep 1
  done
  kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
}

terminate_runtime_processes() {
  local signal=${1:-TERM}
  local pid
  while read -r pid; do
    [[ -n "${pid}" && "${pid}" != "$$" && "${pid}" != "${PPID}" ]] || continue
    kill -"${signal}" "${pid}" >/dev/null 2>&1 || true
  done < <(pgrep -f "${RUNTIME_ROOT}" || true)
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  terminate_group "${CALIBRATION_PID}"
  terminate_pid "${NVIDIA_PID}"
  terminate_pid "${ENV_PID}"
  terminate_runtime_processes TERM
  sleep 5
  terminate_runtime_processes KILL
  sleep 2
  pgrep -af "${RUNTIME_ROOT}|vagen.envs.navigation.serve.*${ENV_PORT}" >"${CONTROL}/owned_processes_after.log" 2>&1 || true
  ss -ltnp >"${CONTROL}/ports_after.log" 2>&1 || true
  if [[ -s "${CONTROL}/owned_processes_after.log" ]]; then status=91; fi
  if ss -ltnH "sport = :${ENV_PORT}" | grep -q .; then status=92; fi
  "${PY}" - "${CONTROL}" "${RUN_OUT}" "${status}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
control=Path(sys.argv[1]); run=Path(sys.argv[2]); status=int(sys.argv[3])
payload={"experiment_id":178,"exit_code":status,"status":"passed" if status==0 else "failed","finished_at":datetime.now(timezone.utc).isoformat()}
for folder in (control, run if run.is_dir() else None):
    if folder is None:
        continue
    fd,name=tempfile.mkstemp(prefix='.controller_status.',suffix='.tmp',dir=folder)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:
        json.dump(payload,handle,indent=2); handle.write('\n')
    os.replace(name,folder/'controller_status.json')
PY
  [[ "${RUNTIME_ROOT}" == /tmp/i178-* ]] && rm -rf -- "${RUNTIME_ROOT}"
  exit "${status}"
}
trap cleanup EXIT

cat >"${RUN_META}" <<EOF
# ID178 optimizer-free K4 beta calibration diagnostic retry

- project/run identity: vagen / ${RUN_NAME}; W&B transport is disabled because this calibration has no optimizer or training metrics.
- purpose: measure real same-generation repaired action-head LLM-logit and frozen K4-MCTS root-mean scales, then propose one fixed Scheme-B beta for human approval.
- provenance: ID174 proved the original ID74 action-token rows had median zero spread. ID176 repaired only those eight rows and passed heldout NLL/BF16-spread plus exact frozen-component gates. ID177 then failed before rollout because it incorrectly treated the Qwen repair export as the full planning root. ID178 keeps all approved behavior/search values and separately binds the immutable ID176 Qwen root and immutable ID74 planning root.
- parent/VAGEN/VERL: ${EXPECTED_PARENT_COMMIT} / ${EXPECTED_VAGEN_COMMIT} / ${EXPECTED_VERL_COMMIT}.
- data: Navigation base_train, common_sense_train, and long_horizon_train; seeds 0..7 in each split; 24 complete trajectories; max 20 real actions each.
- initialization: --model is the completed ID176 Qwen action-row repair; --critic-checkpoint is the original corrected ID74 root owning training_state.pt, SharedSlotProjector, history-1/horizon-4 wm_predictor, and 8-action ValueHead; planning source step remains 776.
- frozen modules: Qwen, vision tower, projector, predictor, and ValueHead. No optimizer, backward, parameter update, checkpoint, resume, or W&B run is allowed.
- planning: TP8 eager vLLM with mm_encoder_tp_mode=data; planner only on TP rank 0; fixed K4, 100 UCT simulations, exploration constant 1.0; direct Q remains separate.
- behavior: calibration applies beta=0; alpha=1, prior temperature=1, float32; CoT temperature=0.7/top-p=0.95, response cap 512.
- beta rule: median population std of 8 LLM logits divided by median population std of 8 MCTS root means; median MCTS spread must exceed 1e-8.
- reward/environment: per-turn format 0.01, terminal format 0, success 1; only train-scene assets are used.
- output: ${RUN_OUT}; a failed or interrupted ID178 is not resumable and must never be reused.
- monitoring: trajectory/turn completion, terminal reasons, direct-Q/MCTS schema, visit sum, candidate trace, finite spreads, planner latency, GPU utilization, process/port cleanup.
- resources: normal partition, one node, 8 H800, 64 CPU, 256 GiB, 60-minute batch-owned hold; calibration timeout 45 minutes; dgx-13/23/32/37/51 excluded.
- stop boundary: after writing the calibrated beta, stop for human approval. This entrypoint cannot start the 10-update canary.
EOF

"${PY}" - "${VAGEN}" "${CONTROL}" <<'PY'
import hashlib, json, sys
from pathlib import Path
vagen=Path(sys.argv[1]); out=Path(sys.argv[2])
expected={
 "base_train":"eb0aa69186604cedc6dc6c2a8874393beae09b7ac1dadae5458e87492b5e01e9",
 "common_sense_train":"dd74a0f02c48e59efda445a68dc717278ffe6fe828f0a431418f205eb67d403b",
 "long_horizon_train":"27d3c95fc0b73fd7f3b89fb6cbad6a93fd9dc91eb42b0ff636b78ddc1d2499e1",
}
result={}
for name,digest in expected.items():
 p=vagen/'vagen/envs/navigation/assets'/f'{name}.json'
 raw=p.read_bytes(); actual=hashlib.sha256(raw).hexdigest(); assert actual==digest
 tasks=json.loads(raw)['tasks']; assert len(tasks)==1200
 result[name]={"sha256":actual,"task_count":len(tasks),"first_scene":tasks[0]['scene'],"first_target":tasks[0]['targetObjectType']}
(out/'source_hashes.json').write_text(json.dumps(result,indent=2)+'\n')
PY

"${PY}" - "${REPAIR_ROOT}" "${CONTROL}" <<'PY'
import hashlib, json, sys
from pathlib import Path
repair=Path(sys.argv[1]); model=repair/'checkpoint'; out=Path(sys.argv[2])
expected={
 'config.json':'f0fbb6c34bb4ce9056f83ad5e92c72084bfe66bb64fee757bf1b91c6831932f2',
 'model.safetensors.index.json':'32acf7bf413e8b87f295e816fe3d68c965e0ab196fbf30b32858b52df41cc97e',
 'model-00001-of-00002.safetensors':'63c933b6ebadae3ee64a4663b5bd1ec71676f64629faf2cda6c15393e534e563',
 'model-00002-of-00002.safetensors':'fcfec9497bc08d1faeb91c07e954b8a9638a1dfa7882f7c3f8b6824d269e2d51',
 'state_proj.pt':'e789a67246022c785521324bbd800d903f46024d8e8d05c504fcbcdedd9d4063',
 'wm_predictor/config.json':'94f58c7e9a0f3fd64ffe58e74480ee9b629dc564198a8f6c9e9dd90d8339801c',
 'wm_predictor/predictor.pt':'85cedd95e5fc6d89cdad7248a85e2dd51b10e1dcf8302d19d5cd3b489af82bb8',
 'value_head/value_head.pt':'b0059ba1eb842cedcbba884dff88a67cd2da127583cea14a800f4215d835c87d',
}
result={}
for rel,digest in expected.items():
 p=model/rel; assert p.is_file(),p
 h=hashlib.sha256()
 with p.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
 actual=h.hexdigest(); assert actual==digest,(rel,actual,digest)
 result[rel]={"bytes":p.stat().st_size,"sha256":actual}
marker=repair/'complete.marker'; assert marker.is_file()
marker_sha=hashlib.sha256(marker.read_bytes()).hexdigest()
assert marker_sha=='37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1'
result['../complete.marker']={"bytes":marker.stat().st_size,"sha256":marker_sha}
(out/'checkpoint_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
PY

"${PY}" - "${PLANNING_MODEL}" "${CONTROL}" <<'PY'
import hashlib, json, sys
from pathlib import Path
model=Path(sys.argv[1]); out=Path(sys.argv[2])
expected={
 'training_state.pt':'cce9c81d6257e0f61dbedf6e075f9d873756f10ad0a98b72ea240788027c0e5e',
 'state_proj.pt':'e789a67246022c785521324bbd800d903f46024d8e8d05c504fcbcdedd9d4063',
 'wm_predictor/config.json':'94f58c7e9a0f3fd64ffe58e74480ee9b629dc564198a8f6c9e9dd90d8339801c',
 'wm_predictor/predictor.pt':'85cedd95e5fc6d89cdad7248a85e2dd51b10e1dcf8302d19d5cd3b489af82bb8',
 'value_head/value_head.pt':'b0059ba1eb842cedcbba884dff88a67cd2da127583cea14a800f4215d835c87d',
}
result={}
for rel,digest in expected.items():
 p=model/rel; assert p.is_file(),p
 h=hashlib.sha256()
 with p.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
 actual=h.hexdigest(); assert actual==digest,(rel,actual,digest)
 result[rel]={"bytes":p.stat().st_size,"sha256":actual}
(out/'planning_checkpoint_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
PY

ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases "${AI2THOR_HOME_ROOT}/.ai2thor/releases"
rm -f "${AI2THOR_HOME_ROOT}/.ai2thor/cuda-vulkan-mapping.json"
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh" > >(tee -a "${CONTROL}/controller.log") 2>&1

timeout --signal=TERM --kill-after=10s 150s "${PY}" -m nimloth.environment.navigation.direct_render_probe --gpu-device 0 | tee "${CONTROL}/render_probe.json"
cd "${VAGEN}"
! ss -ltnH "sport = :${ENV_PORT}" | grep -q .
"${PY}" -m vagen.envs.navigation.serve --host=127.0.0.1 --port="${ENV_PORT}" --devices='[0]' --max_envs=24 --max_inflight=24 --thread_pool_size=24 --session_timeout=3600 >"${CONTROL}/env_server.log" 2>&1 &
ENV_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS --max-time 5 "${ENV_URL}/health" >"${CONTROL}/health.json" 2>/dev/null; then break; fi
  kill -0 "${ENV_PID}" || { tail -100 "${CONTROL}/env_server.log"; exit 4; }
  sleep 2
done
curl -fsS --max-time 5 "${ENV_URL}/health" >/dev/null

timeout --signal=TERM --kill-after=10s 300s "${PY}" - "${ENV_URL}" "${SLURM_JOB_ID}" <<'PY' | tee "${CONTROL}/prewarm.jsonl"
import subprocess,sys
url,job=sys.argv[1:]
for split in ('base_train','common_sense_train','long_horizon_train'):
 command=[sys.executable,'-m','nimloth.environment.navigation.prewarm','--env-url',url,'--eval-set',split,'--seed','0','--timeout-seconds','300','--env-id',f'id178-prewarm-{split}-{job}']
 subprocess.run(command,check=True)
PY

nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 >"${CONTROL}/nvidia_smi.csv" 2>"${CONTROL}/nvidia_smi.err" &
NVIDIA_PID=$!

COMMAND=(
  "${PY}" -m vagen.k4_beta_calibration
  --model "${MODEL}"
  --critic-checkpoint "${PLANNING_MODEL}"
  --agent-loop-config "${VAGEN}/vagen/configs/agent_no_concat.yaml"
  --output-dir "${RUN_OUT}"
  --env-url "${ENV_URL}"
  --run-name "${RUN_NAME}"
  --latent-token-count 16
  --critic-qwen-hidden-dim 2048
  --critic-state-dim 1024
  --joint-snapshot-source-step 776
  --joint-run-seed 172001
  --joint-alpha 1.0
  --joint-beta 0.0
  --joint-prior-temperature 1.0
  --joint-score-dtype float32
  --planning-horizon 4
  --mcts-num-simulations 100
  --mcts-exploration-constant 1.0
  --trajectory-count 24
  --seeds-per-split 8
  --seed-start 0
  --max-turns 20
  --prompt-length 16384
  --response-length 512
  --temperature 0.7
  --top-p 0.95
  --per-turn-format-reward 0.01
  --format-reward 0.0
  --success-reward 1.0
  --tensor-parallel-size 8
  --gpu-memory-utilization 0.6
  --max-num-seqs 24
  --agent-loop-num-workers 24
  --env-timeout 120
  --env-retries 0
  --minimum-median-planner-spread 1e-8
)
printf '%q ' "${COMMAND[@]}" >"${CONTROL}/command.sh"; printf '\n' >>"${CONTROL}/command.sh"
setsid timeout --signal=TERM --kill-after=30s "${CALIBRATION_TIMEOUT_SECONDS}s" "${COMMAND[@]}" >"${CONTROL}/calibration.log" 2>&1 &
CALIBRATION_PID=$!
set +e
wait "${CALIBRATION_PID}"
CALIBRATION_STATUS=$?
set -e
cat "${CONTROL}/calibration.log"
terminate_group "${CALIBRATION_PID}"
CALIBRATION_PID=
(( CALIBRATION_STATUS == 0 )) || exit "${CALIBRATION_STATUS}"

"${PY}" - "${RUN_OUT}" <<'PY'
import json,math,os,sys,tempfile
from pathlib import Path
out=Path(sys.argv[1]); summary=json.loads((out/'summary.json').read_text())
assert summary['schema']=='vagen_k4_beta_calibration_summary_v1'
assert summary['status'] in {'passed','requires_human_review'}
assert summary['optimizer'] is None and summary['checkpoint_output'] is None
assert summary['beta_applied_during_calibration']==0.0
accepted=summary['calibration_accepted']; assert isinstance(accepted,bool)
beta=summary['calibrated_beta_requires_human_approval']
if accepted:
 assert summary['status']=='passed' and summary['review_reason'] is None
 beta=float(beta); assert math.isfinite(beta) and beta>0
else:
 assert summary['status']=='requires_human_review'
 assert summary['review_reason'] in {'llm_median_action_spread_is_zero','mcts_median_action_spread_too_small'}
 assert beta is None or (math.isfinite(float(beta)) and float(beta)==0.0)
assert summary['trajectory_count']==24
assert summary['train_splits']==['base_train','common_sense_train','long_horizon_train']
assert summary['seeds_per_split']==8 and summary['max_turns']==20
assert summary['cot_temperature']==0.7 and summary['cot_top_p']==0.95 and summary['response_length']==512
assert summary['per_turn_format_reward']==0.01 and summary['format_reward']==0.0 and summary['success_reward']==1.0
planning=summary['planning_snapshot']
assert planning['planning_horizon']==4 and planning['mcts_num_simulations']==100 and planning['mcts_exploration_constant']==1.0
assert summary['median_mcts_action_spread']>summary['minimum_median_planner_spread']
rows=[json.loads(line) for line in (out/'turn_records.jsonl').read_text().splitlines()]
assert len(rows)==summary['executed_turn_count']
assert all(sum(row['scoring_record']['planner_root_visit_counts'])==100 for row in rows)
assert all(len(row['scoring_record']['planner_root_mean_values'])==8 for row in rows)
assert all(row['policy_action_logits']==row['scoring_record']['prior_logits'] for row in rows)
assert all(math.isfinite(row['prior_action_spread']) and row['prior_action_spread']>=0 for row in rows)
assert all(math.isfinite(row['mcts_action_spread']) and row['mcts_action_spread']>=0 for row in rows)
assert not (out/'checkpoints').exists()
payload={'status':'passed' if accepted else 'requires_human_review','experiment_id':178,'calibration_accepted':accepted,'review_reason':summary['review_reason'],'calibrated_beta_requires_human_approval':beta,'optimizer':None,'checkpoint':None,'canary_started':False}
fd,name=tempfile.mkstemp(prefix='.final_status.',suffix='.tmp',dir=out)
with os.fdopen(fd,'w',encoding='utf-8') as handle: json.dump(payload,handle,indent=2); handle.write('\n')
os.replace(name,out/'final_status.json')
print(json.dumps(payload,sort_keys=True))
PY
install -m 0644 "${CONTROL}/allocation.txt" "${CONTROL}/source_hashes.json" "${CONTROL}/checkpoint_preflight.json" "${CONTROL}/planning_checkpoint_preflight.json" "${CONTROL}/command.sh" "${CONTROL}/calibration.log" "${RUN_OUT}/"
cat "${RUN_META}" >"${RUN_OUT}/README.md"
"${PY}" - "${RUN_OUT}" >>"${RUN_OUT}/README.md" <<'PY'
import json,sys
from pathlib import Path
summary=json.loads((Path(sys.argv[1])/'summary.json').read_text())
print('\n## Result\n')
print(f"- status: {summary['status']}")
print(f"- complete trajectories / executed turns: {summary['trajectory_count']} / {summary['executed_turn_count']}")
print(f"- median LLM-logit spread: {summary['median_prior_action_spread']:.12g}")
print(f"- median MCTS-root-mean spread: {summary['median_mcts_action_spread']:.12g}")
beta=summary['calibrated_beta_requires_human_approval']
print(f"- proposed beta requiring human approval: {beta if beta is None else format(beta,'.12g')}")
print(f"- calibration accepted / review reason: {summary['calibration_accepted']} / {summary['review_reason']}")
print(f"- prior spread min/median/max/zero-count: {summary['prior_action_spreads']}")
print(f"- MCTS spread min/median/max/zero-count: {summary['mcts_action_spreads']}")
print(f"- planner latency mean/median/max seconds: {summary['planner_latency_seconds']}")
print('- no optimizer, backward, checkpoint, resume, W&B run, or canary was created.')
PY

for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
