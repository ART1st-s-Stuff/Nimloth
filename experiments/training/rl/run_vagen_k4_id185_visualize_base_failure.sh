#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${RUN_DATE:?RUN_DATE is required}"
: "${ID185_VIS_EXPECTED_RUN_NAME:?ID185_VIS_EXPECTED_RUN_NAME is required}"
: "${ID185_VIS_EXPECTED_RUN_DATE:?ID185_VIS_EXPECTED_RUN_DATE is required}"
: "${ID185_VIS_EXPECTED_OUTCOME:?ID185_VIS_EXPECTED_OUTCOME is required}"
: "${ID185_VIS_ENABLE_WANDB:?ID185_VIS_ENABLE_WANDB is required}"
VAGEN=${REPO}/external/VAGEN
VERL=${VAGEN}/verl
PY=${ROOT}/.venv-vagen-main/bin/python3
REPAIR_ROOT=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8
ACTOR_MODEL=${REPAIR_ROOT}/checkpoint
PLANNING_MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
ID184_SOURCE_RUN_OUT=${ROOT}/outputs/experiments/training/rl/2026-08-17/184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8_retry1
SOURCE_CHECKPOINT=${ID184_SOURCE_RUN_OUT}/checkpoints/global_step_20
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}
PHASE_NAME=visualization
PHASE_TAG=vis1
PHASE_OUT=${RUN_OUT}/${PHASE_NAME}
: "${ID185_HEAD_IP:?ID185_HEAD_IP is required}"
: "${ID185_EXPECTED_NNODES:?ID185_EXPECTED_NNODES is required}"
: "${ID185_EXPECTED_GPUS_PER_NODE:?ID185_EXPECTED_GPUS_PER_NODE is required}"
: "${ID185_CLUSTER_NODES:?ID185_CLUSTER_NODES is required}"
: "${RAY_ADDRESS:?RAY_ADDRESS is required}"
: "${RAY_EXPECTED_NODE_IPS:?RAY_EXPECTED_NODE_IPS is required}"
ENV_PORT=$((19700 + SLURM_JOB_ID % 300))
ENV_URL=http://${ID185_HEAD_IP}:${ENV_PORT}
RUNTIME_ROOT=/tmp/i185-${SLURM_JOB_ID}-${PHASE_TAG}
RAY_TMPDIR=${RUNTIME_ROOT}
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
TRAIN_PID=
NVIDIA_PID=
PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS:-16200}

[[ "${RUN_NAME}" == "${ID185_VIS_EXPECTED_RUN_NAME}" ]]
[[ "${RUN_DATE}" == "${ID185_VIS_EXPECTED_RUN_DATE}" ]]
[[ "${ID185_VIS_EXPECTED_OUTCOME}" =~ ^(failure|success|any)$ ]]
[[ "${ID185_VIS_ENABLE_WANDB}" =~ ^(true|false)$ ]]
[[ "${EXPECTED_VERL_COMMIT}" == 494f264494b2525f2c13595f63ac4912963e6d2f ]]
[[ "${SLURM_JOB_PARTITION:-}" == normal ]]
[[ "${ID185_EXPECTED_NNODES}" == 4 ]]
[[ "${ID185_EXPECTED_GPUS_PER_NODE}" == 2 ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 4 ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 16 ]]
[[ "${PHASE_TIMEOUT_SECONDS}" == 16200 ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
IFS=, read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
(( ${#VISIBLE_GPUS[@]} == 2 ))
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == 2 ))
for name in "${GPU_NAMES[@]}"; do [[ "${name}" == *H800* ]]; done
for excluded in dgx-09 dgx-13 dgx-32 dgx-51; do
  [[ "$(hostname)" != "${excluded}" ]]
done

JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}" -o)
grep -q 'Partition=normal' <<<"${JOB_DETAILS}"
grep -q 'NumNodes=4' <<<"${JOB_DETAILS}"
grep -q 'TimeLimit=05:00:00' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*mem=256G([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -q 'MinMemoryNode=64G' <<<"${JOB_DETAILS}"
IFS=, read -r -a CLUSTER_NODES <<<"${ID185_CLUSTER_NODES}"
(( ${#CLUSTER_NODES[@]} == 4 ))
[[ "$(hostname -s)" == "${CLUSTER_NODES[0]}" ]]

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_PARENT_COMMIT}" ]]
[[ "$(git -C "${VAGEN}" rev-parse HEAD)" == "${EXPECTED_VAGEN_COMMIT}" ]]
[[ "$(git -C "${VERL}" rev-parse HEAD)" == "${EXPECTED_VERL_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac ]]
[[ "$(git -C "${REPO}/external/RCDM" rev-parse HEAD)" == 71daaf10a73bb2012864f0827c68d209fc92b0a5 ]]
for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
[[ -x "${PY}" ]]
[[ -d "${ACTOR_MODEL}" ]]
[[ -d "${PLANNING_MODEL}" ]]
[[ -f "${REPAIR_ROOT}/complete.marker" ]]
[[ -f "${PLANNING_MODEL}/training_state.pt" ]]
IFS=, read -r -a EXPECTED_NODE_IPS <<<"${RAY_EXPECTED_NODE_IPS}"
(( ${#EXPECTED_NODE_IPS[@]} == 4 ))
[[ "${ID185_HEAD_IP}" == "${EXPECTED_NODE_IPS[0]}" ]]
[[ "${RAY_ADDRESS}" == "${ID185_HEAD_IP}:"* ]]
"${PY}" - "${RAY_ADDRESS}" "${RAY_EXPECTED_NODE_IPS}" <<'PY'
import json, sys, ray
address=sys.argv[1]; expected=sorted(sys.argv[2].split(','))
ray.init(address=address)
alive=[node for node in ray.nodes() if node['Alive']]
rows=sorted(
 [
  {
   'address':str(node['NodeManagerAddress']),
   'gpus':float(node['Resources'].get('GPU',0)),
  }
  for node in alive
 ],
 key=lambda row:row['address'],
)
ray.shutdown()
assert [row['address'] for row in rows]==expected, (rows,expected)
assert [row['gpus'] for row in rows]==[2.0,2.0,2.0,2.0], rows
print(json.dumps({'status':'ID185_RAY_4X2_OK','nodes':rows}))
PY

set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
export WANDB_PROJECT=vagen
export WANDB_NAME=${RUN_NAME}
export WANDB_RUN_ID=${WANDB_RUN_ID:-nimloth-id185-k4-visualize-base-fail-seed2-retry2}
WANDB_PREFLIGHT_JSON=$("${PY}" - "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_RUN_ID}" "${RUN_NAME}" <<'PY'
import json, sys, wandb
from wandb.errors import CommError
entity,project,run_id,run_name=sys.argv[1:]
path=f'{entity}/{project}/{run_id}'
try:
 run=wandb.Api().run(path)
except CommError as exc:
 response=getattr(exc,'response',None)
 status=getattr(response,'status_code',None)
 missing=status==404 or 'not found' in str(exc).lower() or 'could not find run' in str(exc).lower()
 if not missing:
  raise
 print(json.dumps({'path':path,'exists':False,'name':run_name}))
else:
 raise RuntimeError(
  f'ID185 W&B identity already exists: {path} '
  f'name={run.name!r} state={run.state!r}'
 )
PY
)

[[ ! -e "${RUN_OUT}" ]] || { echo "ID185 output already exists" >&2; exit 2; }
[[ -f "${SOURCE_CHECKPOINT}/joint_checkpoint_complete.json" ]]
[[ -f "${SOURCE_CHECKPOINT}/data.pt" ]]
[[ -f "${ID184_SOURCE_RUN_OUT}/final_status.json" ]]
[[ -f "${ID184_SOURCE_RUN_OUT}/continue_step10_to20/validator.json" ]]
[[ -f "${ID184_SOURCE_RUN_OUT}/continue_step10_to20/wandb_final.json" ]]
"${PY}" - "${SOURCE_CHECKPOINT}" "${ID184_SOURCE_RUN_OUT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[1]); run=Path(sys.argv[2])
marker=json.loads((source/'joint_checkpoint_complete.json').read_text())
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
 return f'sha256:{h.hexdigest()}'
assert digest(source/marker['sidecar'])==marker['sidecar_sha256']
assert digest(source/'data.pt')==marker['dataloader_sha256']
final=json.loads((run/'final_status.json').read_text())
validator=json.loads((run/'continue_step10_to20/validator.json').read_text())
wandb=json.loads((run/'continue_step10_to20/wandb_final.json').read_text())
assert marker['global_step']==20 and marker['source_step']==796
assert final['status']=='passed' and final['global_step']==20
assert validator['status']=='ALL_OK' and validator['checkpoint_steps']==[15,20]
assert wandb['state']=='finished' and wandb['steps']==list(range(10,21))
print(json.dumps({'status':'ID185_SOURCE_STEP20_OK','marker':marker}))
PY
mkdir -p "${RUN_OUT}" "${PHASE_OUT}" "${RUNTIME_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}" "${AI2THOR_HOME_ROOT}/.ai2thor"
printf '%s\n' "${JOB_DETAILS}" >"${PHASE_OUT}/allocation.txt"
printf '%s\n' "${WANDB_PREFLIGHT_JSON}" >"${PHASE_OUT}/wandb_preflight.json"

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
export WANDB_RESUME=never
export WANDB_DIR=${RUN_OUT}/wandb
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

runtime_process_ids() {
  "${PY}" - "${RUNTIME_ROOT}" <<'PY'
import os, sys
from pathlib import Path
root=sys.argv[1].encode()
ancestors=set()
pid=os.getpid()
while pid > 1 and pid not in ancestors:
 ancestors.add(pid)
 try:
  fields=(Path('/proc')/str(pid)/'stat').read_text().split()
  pid=int(fields[3])
 except (FileNotFoundError, PermissionError, ValueError, IndexError):
  break
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit():
  continue
 candidate=int(entry.name)
 if candidate in ancestors:
  continue
 try:
  environ=(entry/'environ').read_bytes()
 except (FileNotFoundError, PermissionError, ProcessLookupError):
  continue
 if root in environ:
  print(candidate)
PY
}

terminate_runtime_processes() {
  local signal=${1:-TERM}
  local pid
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -"${signal}" "${pid}" >/dev/null 2>&1 || true
  done < <(runtime_process_ids)
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  terminate_group "${TRAIN_PID}"
  terminate_pid "${NVIDIA_PID}"
  terminate_pid "${ENV_PID}"
  terminate_runtime_processes TERM
  sleep 5
  terminate_runtime_processes KILL
  sleep 2
  : >"${PHASE_OUT}/owned_processes_after.log"
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    ps -ww -p "${pid}" -o pid=,ppid=,args= >>"${PHASE_OUT}/owned_processes_after.log" 2>&1 || true
  done < <(runtime_process_ids)
  pgrep -af "vagen.envs.navigation.serve.*${ENV_PORT}" >>"${PHASE_OUT}/owned_processes_after.log" 2>&1 || true
  ss -ltnp >"${PHASE_OUT}/ports_after.log" 2>&1 || true
  if [[ -s "${PHASE_OUT}/owned_processes_after.log" ]]; then status=91; fi
  if ss -ltnH "sport = :${ENV_PORT}" | grep -q .; then status=92; fi
  "${PY}" - "${PHASE_OUT}" "${status}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
out=Path(sys.argv[1]); status=int(sys.argv[2])
payload={"phase":"visualization","exit_code":status,"status":"passed" if status==0 else "failed","finished_at":datetime.now(timezone.utc).isoformat()}
fd,name=tempfile.mkstemp(prefix='.phase_status.',suffix='.tmp',dir=out)
with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(payload,f,indent=2); f.write('\n')
os.replace(name,out/'phase_status.json')
PY
  [[ "${RUNTIME_ROOT}" == /tmp/i185-* ]] && rm -rf -- "${RUNTIME_ROOT}"
  exit "${status}"
}
trap cleanup EXIT

"${PY}" - "${VAGEN}" "${PHASE_OUT}" "${ENV_URL}" \
  "${ID184_SOURCE_RUN_OUT}/continue_step10_to20/train_navigation_joint_id184.yaml" <<'PY'
import hashlib, json, sys
from pathlib import Path
vagen=Path(sys.argv[1]); out=Path(sys.argv[2]); url=sys.argv[3]
source_train=Path(sys.argv[4])
train_text=source_train.read_text()
for reward_field in (
 'per_turn_format_reward: 0.01',
 'format_reward: 0.0',
 'success_reward: 1.0',
):
 assert reward_field in train_text
(out/'train_navigation_joint_id185.yaml').write_text(train_text)
val_src=vagen/'examples/train/navigation/val_navigation_joint_id185.yaml'
val_text=val_src.read_text()
assert 'http://127.0.0.1:8000' in val_text
for reward_field in (
 'per_turn_format_reward: 0.01',
 'format_reward: 0.0',
 'success_reward: 1.0',
):
 assert reward_field in val_text
(out/'val_navigation_joint_id185.yaml').write_text(
 val_text.replace('http://127.0.0.1:8000',url)
)
expected={
 'base_train':(1200,'eb0aa69186604cedc6dc6c2a8874393beae09b7ac1dadae5458e87492b5e01e9'),
 'common_sense_train':(1200,'dd74a0f02c48e59efda445a68dc717278ffe6fe828f0a431418f205eb67d403b'),
 'long_horizon_train':(1200,'27d3c95fc0b73fd7f3b89fb6cbad6a93fd9dc91eb42b0ff636b78ddc1d2499e1'),
 'base':(60,'6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a'),
 'common_sense':(60,'3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf'),
 'long_horizon':(60,'ff23dcb171ff8008721a8a74ee7c677f6535f84c25407715326a8b313771bdaf'),
 'complex_instruction':(60,'767730e5b83812a199a27be41477da98def2becfd5cc8bd3e45d8cfdce260b9b'),
 'visual_appearance':(60,'e66bc8aab0141c662761ef1a1d857aa6297972c6a0890526b008990eded8ddc1'),
}
assets={}; scene_sets={}
for name,(task_count,digest) in expected.items():
 asset=vagen/'vagen/envs/navigation/assets'/f'{name}.json'
 tasks=json.loads(asset.read_text())['tasks']
 assert len(tasks)==task_count
 scenes={str(item['scene']) for item in tasks}
 actual=hashlib.sha256(asset.read_bytes()).hexdigest()
 assert actual==digest
 scene_sets[name]=scenes
 assets[name]={'sha256':actual,'task_count':len(tasks),'scene_count':len(scenes)}
train_scenes=set().union(*(scene_sets[name] for name in ('base_train','common_sense_train','long_horizon_train')))
for name in ('base','common_sense','long_horizon','complex_instruction','visual_appearance'):
 assert len(scene_sets[name])==60
 assert scene_sets[name].isdisjoint(train_scenes)
(out/'source_hashes.json').write_text(json.dumps({
 'train_config_sha256':hashlib.sha256(train_text.encode()).hexdigest(),
 'source_train_config_sha256':hashlib.sha256(source_train.read_bytes()).hexdigest(),
 'val_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/val_navigation_joint_id185.yaml').read_bytes()).hexdigest(),
 'heldout_train_scene_overlap':0,
 'assets':assets,
},indent=2)+'\n')
PY
export ID185_VIS_TRAIN_CONFIG=${PHASE_OUT}/train_navigation_joint_id185.yaml
export ID185_VIS_VAL_CONFIG=${PHASE_OUT}/val_navigation_joint_id185.yaml
export ID185_VIS_ACTOR_MODEL=${ACTOR_MODEL}
export ID185_VIS_PLANNING_CHECKPOINT=${PLANNING_MODEL}
export ID185_VIS_AGENT_CONFIG=${VAGEN}/vagen/configs/agent_no_concat.yaml
export ID185_VIS_RUN_NAME=${RUN_NAME}
export ID185_VIS_RUN_OUT=${RUN_OUT}
export ID185_VIS_SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}
export ID185_VIS_SEED=${ID185_VIS_SEED:-2}

"${PY}" - "${ID185_TRAIN_CONFIG}" "${ID185_VAL_CONFIG}" "${PHASE_OUT}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from vagen.gym_agent_dataset import AgenticDataset
train_path=Path(sys.argv[1]); val_path=Path(sys.argv[2]); out=Path(sys.argv[3])
def rows_for(path):
 dataset=AgenticDataset(data_files=str(path),config={'base_seed':0})
 return [{
  'dataset_index':index,
  'data_source':str(dataset[index]['data_source']),
  'seed':int(dataset[index]['seed']),
  'rollout_sample_id':str(dataset[index]['rollout_sample_id']),
 } for index in range(len(dataset))]
train_rows=rows_for(train_path); val_rows=rows_for(val_path)
train_counts=Counter(row['data_source'] for row in train_rows)
val_counts=Counter(row['data_source'] for row in val_rows)
assert len(train_rows)==180
assert train_counts=={
 'navigation_base_train_id184':60,
 'navigation_common_sense_train_id184':60,
 'navigation_long_horizon_train_id184':60,
}
assert all(
 len({row['seed'] for row in train_rows if row['data_source']==source})==60
 for source in train_counts
)
expected_val_sources={
 'navigation_base_test_id185',
 'navigation_common_sense_test_id185',
 'navigation_complex_instruction_test_id185',
 'navigation_visual_appearance_test_id185',
 'navigation_long_horizon_test_id185',
}
assert len(val_rows)==300
assert val_counts==Counter({source:60 for source in expected_val_sources})
assert all(
 sorted(row['seed'] for row in val_rows if row['data_source']==source)==list(range(1,61))
 for source in expected_val_sources
)
assert len({row['rollout_sample_id'] for row in train_rows})==180
assert len({row['rollout_sample_id'] for row in val_rows})==300
(out/'dataset_manifest.json').write_text(json.dumps({
 'base_seed':0,
 'train_seed_directive':'inclusive [0,1199], unique within each split',
 'validation_seeds':'explicit historical VAGEN seeds 1..60',
 'train_counts':dict(sorted(train_counts.items())),
 'validation_counts':dict(sorted(val_counts.items())),
 'train_rows':train_rows,
 'validation_rows':val_rows,
},indent=2)+'\n')
PY

cat >"${RUN_OUT}/README.md" <<EOF
# One-rollout K4 evaluation evidence canary

- source: immutable ID184 step20/source796 and frozen snapshot sha256:6648780b3791cb4b937974b151b9e119ed9bf74602d1bc21dabfc30a3914d969.
- selected episode: Base seed ${ID185_VIS_SEED}. Selection is bound to data_source+seed because the current environment transport URL is intentionally excluded from semantic task selection and changes the derived rollout_sample_id.
- execution: a new same-contract rollout, val-only, K4/100 UCT/c1 Scheme-B alpha1 beta85.78297006578457, TP8/DP1 rollout and DP8 restore. All model modules are frozen; there is no optimizer update or checkpoint.
- audit: every real action step persists the true observation image, actual generated CoT, prior and executed action, direct all-action Q, Frozen-V current state value, MCTS root predictions/visits, and all candidate 4-action sequences.
- expected outcome: ${ID185_VIS_EXPECTED_OUTCOME}. This controls only final validation and never changes policy execution.
- results: `evaluation_browser/global_step_20/index.html` is the unified browser; `visualization/rollout_audit/index.html` remains the legacy single-rollout view for cross-checking.
- W&B: project `vagen`, run `${RUN_NAME}`, enabled=${ID185_VIS_ENABLE_WANDB}, identity `${WANDB_RUN_ID}`.
EOF

"${PY}" - "${REPAIR_ROOT}" "${PHASE_OUT}" <<'PY'
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
 result[rel]={'bytes':p.stat().st_size,'sha256':actual}
marker=repair/'complete.marker'; assert marker.is_file()
marker_sha=hashlib.sha256(marker.read_bytes()).hexdigest()
assert marker_sha=='37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1'
result['../complete.marker']={'bytes':marker.stat().st_size,'sha256':marker_sha}
(out/'actor_checkpoint_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
PY

"${PY}" - "${PLANNING_MODEL}" "${PHASE_OUT}" <<'PY'
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
 result[rel]={'bytes':p.stat().st_size,'sha256':actual}
(out/'planning_checkpoint_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
PY

"${PY}" - "${PHASE_OUT}" <<'PY'
import json, sys, torch
from pathlib import Path
from nimloth.wm import SequenceSIGReg
out=Path(sys.argv[1]); device=torch.device('cuda',0)
module=SequenceSIGReg(knots=17,num_proj=1024).to(device)
assert all(buffer.device==device for buffer in module.buffers())
states=torch.randn(2,2,1024,device=device,requires_grad=True)
loss=module(states)
assert loss is not None and torch.isfinite(loss)
loss.backward()
assert states.grad is not None and torch.isfinite(states.grad).all()
(out/'sigreg_device_preflight.json').write_text(json.dumps({
 'status':'SIGREG_CUDA_FORWARD_BACKWARD_OK','device':str(device),
 'loss':float(loss.detach().item()),'buffer_devices':sorted({str(x.device) for x in module.buffers()}),
},indent=2)+'\n')
PY

ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases "${AI2THOR_HOME_ROOT}/.ai2thor/releases"
rm -f "${AI2THOR_HOME_ROOT}/.ai2thor/cuda-vulkan-mapping.json"
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh" > >(tee -a "${PHASE_OUT}/controller.log") 2>&1

timeout --signal=TERM --kill-after=10s 150s "${PY}" -m nimloth.environment.navigation.direct_render_probe --gpu-device 0 | tee "${PHASE_OUT}/render_probe.json"
cd "${VAGEN}"
! ss -ltnH "sport = :${ENV_PORT}" | grep -q .
"${PY}" -m vagen.envs.navigation.serve --host="${ID185_HEAD_IP}" --port="${ENV_PORT}" --devices='[0]' --max_envs=40 --max_inflight=40 --thread_pool_size=40 --session_timeout=14400 >"${PHASE_OUT}/env_server.log" 2>&1 &
ENV_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS --max-time 5 "${ENV_URL}/health" >"${PHASE_OUT}/health.json" 2>/dev/null; then break; fi
  kill -0 "${ENV_PID}" || { tail -100 "${PHASE_OUT}/env_server.log"; exit 4; }
  sleep 2
done
curl -fsS --max-time 5 "${ENV_URL}/health" >/dev/null
for split in base common_sense complex_instruction visual_appearance long_horizon; do
  timeout --signal=TERM --kill-after=10s 300s "${PY}" -m nimloth.environment.navigation.prewarm --env-url "${ENV_URL}" --eval-set "${split}" --seed 0 --timeout-seconds 300 --env-id "id185-prewarm-${split}-${SLURM_JOB_ID}" | tee "${PHASE_OUT}/prewarm_${split}.json"
done

srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=4 --ntasks=4 \
  --ntasks-per-node=1 --gres=gpu:2 --label \
  nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 \
  >"${PHASE_OUT}/nvidia_smi.csv" 2>"${PHASE_OUT}/nvidia_smi.err" &
NVIDIA_PID=$!

cd "${VAGEN}"
COMMAND=(
  "${PY}" -m vagen.main_ppo
  --config-path="${VAGEN}/vagen/configs"
  --config-name=joint_id185_visualize_one
  "hydra.run.dir=${PHASE_OUT}/hydra"
  hydra.job.chdir=false
  trainer.resume_mode=resume_path
  trainer.total_training_steps=20
  trainer.total_epochs=20
  trainer.val_before_train=true
  trainer.val_only=true
  trainer.test_freq=-1
  trainer.save_freq=5
  trainer.joint_dataloader_resume_policy=exact
)
if [[ "${ID185_VIS_ENABLE_WANDB}" == true ]]; then
  COMMAND+=("trainer.logger=[console,wandb]")
fi
printf '%q ' "${COMMAND[@]}" >"${PHASE_OUT}/command.sh"; printf '\n' >>"${PHASE_OUT}/command.sh"
setsid timeout --signal=TERM --kill-after=30s "${PHASE_TIMEOUT_SECONDS}s" "${COMMAND[@]}" >"${PHASE_OUT}/train.log" 2>&1 &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
cat "${PHASE_OUT}/train.log"
terminate_group "${TRAIN_PID}"
TRAIN_PID=
(( TRAIN_STATUS == 0 )) || exit "${TRAIN_STATUS}"

"${PY}" "${REPO}/experiments/training/rl/render_id185_rollout_visualization.py" \
  --audit-dir "${PHASE_OUT}/rollout_audit" \
  --output "${PHASE_OUT}/rollout_audit/index.html"
[[ -f "${PHASE_OUT}/rollout_audit/index.html" ]]

"${PY}" - "${SOURCE_CHECKPOINT}" "${PHASE_OUT}" "${RUN_OUT}" \
  "${ID185_VIS_SEED}" "${ID185_VIS_EXPECTED_OUTCOME}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
from vagen.joint_policy.checkpoint import load_complete_joint_checkpoint
source=Path(sys.argv[1]); out=Path(sys.argv[2]); run=Path(sys.argv[3]); expected_seed=int(sys.argv[4])
expected_outcome=sys.argv[5]
manifest=json.loads((out/'dataset_manifest.json').read_text())
selected=[
 row for row in manifest['validation_rows']
 if row['data_source']=='navigation_base_test_id185' and row['seed']==expected_seed
]
assert len(selected)==1
expected_id=selected[0]['rollout_sample_id']
def atomic_json(path,payload):
 fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent)
 with os.fdopen(fd,'w',encoding='utf-8') as handle:
  json.dump(payload,handle,indent=2,allow_nan=False); handle.write('\n')
 os.replace(name,path)
payload=load_complete_joint_checkpoint(source)
active=payload['frozen_q_owner']['active_snapshot_state']
assert payload['global_step']==20
assert payload['actor_critic']['source_step']==796
assert active['snapshot_source_step']==796
assert active['snapshot_id']=='sha256:6648780b3791cb4b937974b151b9e119ed9bf74602d1bc21dabfc30a3914d969'
assert not list((run/'checkpoints').glob('global_step_*'))
log=(out/'train.log').read_text()
assert 'ID185_K4_VISUALIZATION_RESTORE_OK global_step=20' in log
assert 'SINGLE_ROLLOUT_VISUALIZATION_AUDIT_COMPLETE' in log
assert 'VALIDATION_BATCH_JOURNAL_COMPLETE batches=1 rows=1' in log
audit_dir=out/'rollout_audit'
audit=json.loads((audit_dir/'rollout_audit.json').read_text())
assert audit['schema']=='vagen_single_rollout_visualization_audit_v1'
assert audit['rollout_sample_id']==expected_id
assert audit['rollout_repeat_index']==0
assert audit['turn_count']==len(audit['turns'])
assert audit['turn_count']>=1
success=bool(audit['success'])
if expected_outcome == 'failure':
 assert success is False
elif expected_outcome == 'success':
 assert success is True
else:
 assert expected_outcome == 'any'
for index,turn in enumerate(audit['turns']):
 assert turn['turn_index']==index and turn['turn_number']==index+1
 assert turn['cot'].startswith('<think>') and turn['cot'].endswith('</think>')
 assert len(turn['action_ranking'])==8
 assert len(turn['predicted_action_sequences'])>=1
 assert sum(row['root_visits'] for row in turn['action_ranking'])==100
 assert sum(row['visits'] for row in turn['predicted_action_sequences'])==100
 assert sum(row['is_executed_action'] for row in turn['action_ranking'])==1
 assert (audit_dir/turn['observation_image']).is_file()
 assert turn['snapshot_source_step']==796
 assert turn['snapshot_id']==active['snapshot_id']
journal=out/'validation_batch_journal'
complete=json.loads((journal/'complete.json').read_text())
assert complete['batch_count']==1 and complete['row_count']==1
rows=[json.loads(line) for line in (run/'validation/20.jsonl').read_text().splitlines() if line]
assert len(rows)==1 and rows[0]['rollout_sample_id']==expected_id
assert bool(float(rows[0]['traj_success'])) is success
assert (audit_dir/'index.html').is_file()
browser=run/'evaluation_browser/global_step_20'
complete=json.loads((browser/'complete.json').read_text())
manifest_bytes=(browser/'manifest.json').read_bytes()
import hashlib
assert complete['manifest_sha256']=='sha256:'+hashlib.sha256(manifest_bytes).hexdigest()
manifest=json.loads(manifest_bytes)
assert manifest['status']=='complete' and manifest['rollout_count']==1
assert manifest['expected_rollouts']==1 and len(manifest['rollouts'])==1
browser_row=manifest['rollouts'][0]
assert browser_row['identity']['rollout_sample_id']==expected_id
assert bool(browser_row['success']) is success
assert browser_row['turn_count']==audit['turn_count']
assert browser_row['capabilities']['mcts_candidates'] is True
artifact=browser/browser_row['artifact']
assert artifact.is_file()
browser_audit=artifact.parent/'rollout.json'
assert 'sha256:'+hashlib.sha256(browser_audit.read_bytes()).hexdigest()==browser_row['audit_sha256']
assert (browser/'index.html').is_file()
summary={
 'status':'ALL_OK','phase':'visualization','global_step':20,'source_step':796,
 'snapshot_id':active['snapshot_id'],'rollout_sample_id':expected_id,
 'turn_count':audit['turn_count'],'success':success,'checkpoint_steps':[],
 'rollout_audit':str(audit_dir/'rollout_audit.json'),
 'html':str(audit_dir/'index.html'),
 'rollout_browser':str(browser/'index.html'),
 'rollout_browser_manifest_sha256':complete['manifest_sha256'],
}
atomic_json(out/'validator.json',summary)
atomic_json(run/'final_status.json',{**summary,'status':'passed'})
print(json.dumps(summary,allow_nan=False))
PY

for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
