#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
: "${PHASE:?PHASE must be train_to_5 or resume_to_10}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${RUN_DATE:?RUN_DATE is required}"
VAGEN=${REPO}/external/VAGEN
VERL=${VAGEN}/verl
PY=${ROOT}/.venv-vagen-main/bin/python3
REPAIR_ROOT=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8
ACTOR_MODEL=${REPAIR_ROOT}/checkpoint
PLANNING_MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}
CHECKPOINT_DIR=${RUN_OUT}/checkpoints
if [[ "${PHASE}" == train_to_5 ]]; then
  PHASE_NAME=phase1_train_to_5
  PHASE_TAG=p1
else
  [[ "${PHASE}" == resume_to_10 ]]
  PHASE_NAME=phase2_fresh_resume_to_10
  PHASE_TAG=p2
fi
PHASE_OUT=${RUN_OUT}/${PHASE_NAME}
ENV_PORT=$((19100 + SLURM_JOB_ID % 300 + ($([[ "${PHASE}" == train_to_5 ]] && echo 0 || echo 300))))
ENV_URL=http://127.0.0.1:${ENV_PORT}
RUNTIME_ROOT=/tmp/i183-${SLURM_JOB_ID}-${PHASE_TAG}
RAY_TMPDIR=${RUNTIME_ROOT}
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
TRAIN_PID=
NVIDIA_PID=
PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS:-13200}

[[ "${PHASE}" == train_to_5 || "${PHASE}" == resume_to_10 ]]
[[ "${RUN_NAME}" == 183_canary_k4schemeb_jointupdate_dp8_tp8_u10_r5_train3x8_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8 ]]
[[ "${EXPECTED_VERL_COMMIT}" == 494f264494b2525f2c13595f63ac4912963e6d2f ]]
[[ "${SLURM_JOB_PARTITION:-}" == normal ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 64 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 262144 ]]
[[ "${PHASE_TIMEOUT_SECONDS}" == 13200 ]]
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
grep -q 'TimeLimit=04:00:00' <<<"${JOB_DETAILS}"
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
[[ -d "${ACTOR_MODEL}" ]]
[[ -d "${PLANNING_MODEL}" ]]
[[ -f "${REPAIR_ROOT}/complete.marker" ]]
[[ -f "${PLANNING_MODEL}/training_state.pt" ]]

if [[ "${PHASE}" == train_to_5 ]]; then
  [[ ! -e "${RUN_OUT}" ]] || { echo "ID183 output already exists" >&2; exit 2; }
  mkdir -p "${RUN_OUT}" "${CHECKPOINT_DIR}"
else
  [[ -f "${CHECKPOINT_DIR}/global_step_5/joint_checkpoint_complete.json" ]]
  [[ -f "${RUN_OUT}/phase1_train_to_5/validator.json" ]]
  grep -q '"status": "ALL_OK"' "${RUN_OUT}/phase1_train_to_5/validator.json"
  grep -q '"global_step": 5' "${RUN_OUT}/phase1_train_to_5/validator.json"
  [[ -f "${RUN_OUT}/validation/0.jsonl" ]]
  [[ ! -e "${CHECKPOINT_DIR}/global_step_10" ]]
  [[ ! -e "${RUN_OUT}/validation/10.jsonl" ]]
fi
[[ ! -e "${PHASE_OUT}" ]]
mkdir -p "${PHASE_OUT}" "${RUNTIME_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}" "${AI2THOR_HOME_ROOT}/.ai2thor"
printf '%s\n' "${JOB_DETAILS}" >"${PHASE_OUT}/allocation.txt"

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
set -a
source /project/peilab/atst/flower/.env
set +a
export WANDB_PROJECT=vagen
export WANDB_NAME=${RUN_NAME}
export WANDB_RUN_ID=nimloth-id183-k4-10update-canary
export WANDB_RESUME=allow
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
  terminate_group "${TRAIN_PID}"
  terminate_pid "${NVIDIA_PID}"
  terminate_pid "${ENV_PID}"
  terminate_runtime_processes TERM
  sleep 5
  terminate_runtime_processes KILL
  sleep 2
  pgrep -af "${RUNTIME_ROOT}|vagen.envs.navigation.serve.*${ENV_PORT}" >"${PHASE_OUT}/owned_processes_after.log" 2>&1 || true
  ss -ltnp >"${PHASE_OUT}/ports_after.log" 2>&1 || true
  if [[ -s "${PHASE_OUT}/owned_processes_after.log" ]]; then status=91; fi
  if ss -ltnH "sport = :${ENV_PORT}" | grep -q .; then status=92; fi
  "${PY}" - "${PHASE_OUT}" "${PHASE}" "${status}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
out=Path(sys.argv[1]); phase=sys.argv[2]; status=int(sys.argv[3])
payload={"phase":phase,"exit_code":status,"status":"passed" if status==0 else "failed","finished_at":datetime.now(timezone.utc).isoformat()}
fd,name=tempfile.mkstemp(prefix='.phase_status.',suffix='.tmp',dir=out)
with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(payload,f,indent=2); f.write('\n')
os.replace(name,out/'phase_status.json')
PY
  [[ "${RUNTIME_ROOT}" == /tmp/i183-* ]] && rm -rf -- "${RUNTIME_ROOT}"
  exit "${status}"
}
trap cleanup EXIT

"${PY}" - "${VAGEN}" "${PHASE_OUT}" "${ENV_URL}" <<'PY'
import hashlib, json, sys
from pathlib import Path
vagen=Path(sys.argv[1]); out=Path(sys.argv[2]); url=sys.argv[3]
for name in ('train','val'):
 src=vagen/'examples/train/navigation'/f'{name}_navigation_joint_id183.yaml'
 text=src.read_text()
 assert 'http://127.0.0.1:8000' in text
 for reward_field in (
  'per_turn_format_reward: 0.01',
  'format_reward: 0.0',
  'success_reward: 1.0',
 ):
  assert reward_field in text
 dst=out/f'{name}_navigation_joint_id183.yaml'
 dst.write_text(text.replace('http://127.0.0.1:8000',url))
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
 'train_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/train_navigation_joint_id183.yaml').read_bytes()).hexdigest(),
 'val_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/val_navigation_joint_id183.yaml').read_bytes()).hexdigest(),
 'heldout_train_scene_overlap':0,
 'assets':assets,
},indent=2)+'\n')
PY
export ID183_TRAIN_CONFIG=${PHASE_OUT}/train_navigation_joint_id183.yaml
export ID183_VAL_CONFIG=${PHASE_OUT}/val_navigation_joint_id183.yaml
export ID183_ACTOR_MODEL=${ACTOR_MODEL}
export ID183_PLANNING_CHECKPOINT=${PLANNING_MODEL}
export ID183_AGENT_CONFIG=${VAGEN}/vagen/configs/agent_no_concat.yaml
export ID183_RUN_NAME=${RUN_NAME}
export ID183_RUN_OUT=${RUN_OUT}

"${PY}" - "${ID183_TRAIN_CONFIG}" "${ID183_VAL_CONFIG}" "${PHASE_OUT}" <<'PY'
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
assert len(train_rows)==24
assert train_counts=={
 'navigation_base_train_id183':8,
 'navigation_common_sense_train_id183':8,
 'navigation_long_horizon_train_id183':8,
}
expected_val_sources={
 'navigation_base_val_id183',
 'navigation_common_sense_val_id183',
 'navigation_long_horizon_val_id183',
 'navigation_complex_instruction_val_id183',
 'navigation_visual_appearance_val_id183',
}
assert len(val_rows)==40
assert val_counts==Counter({source:8 for source in expected_val_sources})
assert all(
 sorted(row['seed'] for row in val_rows if row['data_source']==source)==list(range(8))
 for source in expected_val_sources
)
assert len({row['rollout_sample_id'] for row in train_rows})==24
assert len({row['rollout_sample_id'] for row in val_rows})==40
(out/'dataset_manifest.json').write_text(json.dumps({
 'base_seed':0,
 'train_seed_directive':'inclusive [0,8] sampled deterministically with replacement',
 'validation_seeds':'explicit 0..7 per held-out asset',
 'train_counts':dict(sorted(train_counts.items())),
 'validation_counts':dict(sorted(val_counts.items())),
 'train_rows':train_rows,
 'validation_rows':val_rows,
},indent=2)+'\n')
PY

if [[ "${PHASE}" == train_to_5 ]]; then
  cat >"${RUN_OUT}/README.md" <<EOF
# ID183 K4 Scheme-B 10-update canary

- project/run: vagen / ${RUN_NAME}
- approval: prepare a bounded 10-update canary with held-out 5x8 validation before step1, atomic checkpoint at step5, fresh-runtime resume for steps6--10, held-out 5x8 validation after step10, and atomic checkpoint at step10.
- stop boundary: the separate 5x60 held-out evaluation must not start from this entrypoint; long training and general production enablement remain forbidden.
- parent/VAGEN/VERL: ${EXPECTED_PARENT_COMMIT} / ${EXPECTED_VAGEN_COMMIT} / ${EXPECTED_VERL_COMMIT}
- training data: base_train, common_sense_train and long_horizon_train; 8 deterministic dataset instances per split from inclusive seed directive [0,8]; each update has 24 trajectories capped at20 real actions.
- validation data: held-out base/common_sense/long_horizon/complex_instruction/visual_appearance, explicit seeds0..7 per asset (5x8); all 60 held-out scenes are disjoint from the three train assets.
- reward: per-turn format0.01, terminal format0, success1.
- actor initialization: immutable completed ID176 repaired Qwen checkpoint.
- planning initialization: immutable corrected ID74 projector, horizon-4 predictor and 8-action ValueHead at source step776.
- behavior: K4/100 UCT/c1, alpha1, approved beta85.78297006578457, prior temperature1, float32, keyed sampling, CoT temperature0.7/top-p0.95, response cap512.
- update: actor lr1e-7, PPO clip0.2, one epoch, token KL0.01, guided entropy0.01; unified projector/predictor/ValueHead AdamW lr1e-4, state/DINO/SIGReg weights1/0.5/0.1, selected Huber delta1, gamma1, lambda0.95.
- checkpoint/resume: phase1 writes only complete step5/source781; phase2 is a separate fresh runtime that must load step5 and writes only complete step10/source786. Both step5 and step10 remain; no intermediate checkpoint is valid.
- resources per separately approved phase: normal, one node, 8 H800, 64 CPU, 256 GiB, four-hour allocation; excluded nodes dgx-13/23/32/37/51.
EOF
fi

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
"${PY}" -m vagen.envs.navigation.serve --host=127.0.0.1 --port="${ENV_PORT}" --devices='[0]' --max_envs=40 --max_inflight=40 --thread_pool_size=40 --session_timeout=14400 >"${PHASE_OUT}/env_server.log" 2>&1 &
ENV_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS --max-time 5 "${ENV_URL}/health" >"${PHASE_OUT}/health.json" 2>/dev/null; then break; fi
  kill -0 "${ENV_PID}" || { tail -100 "${PHASE_OUT}/env_server.log"; exit 4; }
  sleep 2
done
curl -fsS --max-time 5 "${ENV_URL}/health" >/dev/null
for split in base_train common_sense_train long_horizon_train base common_sense long_horizon complex_instruction visual_appearance; do
  timeout --signal=TERM --kill-after=10s 300s "${PY}" -m nimloth.environment.navigation.prewarm --env-url "${ENV_URL}" --eval-set "${split}" --seed 0 --timeout-seconds 300 --env-id "id183-prewarm-${split}-${SLURM_JOB_ID}" | tee "${PHASE_OUT}/prewarm_${split}.json"
done

nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 >"${PHASE_OUT}/nvidia_smi.csv" 2>"${PHASE_OUT}/nvidia_smi.err" &
NVIDIA_PID=$!

PHASE_OVERRIDES=()
if [[ "${PHASE}" == resume_to_10 ]]; then
  PHASE_OVERRIDES+=(
    joint_integration_gate.phase=resume_to_10
    trainer.total_training_steps=10
    trainer.total_epochs=10
    trainer.resume_mode=auto
    trainer.val_before_train=false
    trainer.test_freq=10
  )
fi
cd "${VAGEN}"
COMMAND=("${PY}" -m vagen.main_ppo --config-path="${VAGEN}/vagen/configs" --config-name=joint_id183_canary "hydra.run.dir=${PHASE_OUT}/hydra" hydra.job.chdir=false "${PHASE_OVERRIDES[@]}")
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

if [[ "${PHASE}" == train_to_5 ]]; then
  EXPECTED_STEP=5
  EXPECTED_SOURCE=781
  EXPECTED_VALIDATION_STEP=0
else
  EXPECTED_STEP=10
  EXPECTED_SOURCE=786
  EXPECTED_VALIDATION_STEP=10
fi
"${PY}" - "${CHECKPOINT_DIR}" "${EXPECTED_STEP}" "${EXPECTED_SOURCE}" "${PHASE}" "${PHASE_OUT}" "${RUN_OUT}/planning_snapshots" "${RUN_OUT}" "${EXPECTED_VALIDATION_STEP}" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
import torch
from vagen.joint_policy.canary import summarize_canary_validation_rows
from vagen.joint_policy.checkpoint import load_complete_joint_checkpoint
from nimloth.training.rl.joint_planner import load_frozen_planning_snapshot_file
root=Path(sys.argv[1]); step=int(sys.argv[2]); source=int(sys.argv[3]); phase=sys.argv[4]
out=Path(sys.argv[5]); snapshots=Path(sys.argv[6]); run_out=Path(sys.argv[7]); validation_step=int(sys.argv[8])
def atomic_json(path,payload):
 fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent)
 with os.fdopen(fd,'w',encoding='utf-8') as handle:
  json.dump(payload,handle,indent=2,allow_nan=False); handle.write('\n')
 os.replace(name,path)
def checkpoint_payload(value):
 payload=load_complete_joint_checkpoint(root/f'global_step_{value}')
 actor=payload['actor_critic']; owner=payload['frozen_q_owner']; active=owner['active_snapshot_state']
 assert payload['global_step']==value and payload['run_seed']==42179
 assert actor['schema']=='vagen_joint_k4_actor_planning_checkpoint_v1'
 assert actor['completed_updates']==value
 assert actor['source_step']==776+value and actor['score_dtype']=='float32'
 assert actor['planning_optimizer_state']['state']
 assert actor['planning_optimizer_fingerprint']
 assert owner['activation_version']==value
 assert active['schema']=='vagen_frozen_k4_planner_transport_v1'
 assert active['snapshot_id']==actor['snapshot_id']
 assert active['snapshot_source_step']==776+value
 assert actor['snapshot_transport']==active
 assert Path(active['transport_path']).is_file()
 return payload,actor,owner
payload,actor,owner=checkpoint_payload(step)
assert actor['source_step']==source
if phase=='resume_to_10':
 _,step5_actor,step5_owner=checkpoint_payload(5)
 assert step5_actor['source_step']==781 and step5_owner['activation_version']==5
 log=(out/'train.log').read_text()
 assert 'Setting global step to 5' in log
 assert 'ID183_K4_CANARY_RESUME_OK global_step=5' in log
expected_checkpoint_steps={5} if phase=='train_to_5' else {5,10}
actual_checkpoint_steps={
 int(path.name.removeprefix('global_step_'))
 for path in root.glob('global_step_*') if path.is_dir()
}
assert actual_checkpoint_steps==expected_checkpoint_steps
initial_path=snapshots/'source_step_776/frozen_k4_planner.pt'
updated_path=snapshots/f'source_step_{source}/frozen_k4_planner.pt'
assert initial_path.is_file() and updated_path.is_file()
initial=load_frozen_planning_snapshot_file(initial_path,device=torch.device('cpu'))
updated=load_frozen_planning_snapshot_file(updated_path,device=torch.device('cpu'))
assert initial.source_step==776 and updated.source_step==source
assert updated.snapshot_id==actor['snapshot_id']
assert initial.snapshot_id!=updated.snapshot_id
validation_path=run_out/'validation'/f'{validation_step}.jsonl'
rows=[json.loads(line) for line in validation_path.read_text().splitlines() if line]
expected_sources=(
 'navigation_base_val_id183',
 'navigation_common_sense_val_id183',
 'navigation_long_horizon_val_id183',
 'navigation_complex_instruction_val_id183',
 'navigation_visual_appearance_val_id183',
)
validation=summarize_canary_validation_rows(
 rows,expected_data_sources=expected_sources,expected_rows_per_source=8,
 expected_step=validation_step,
)
summary={
 'status':'ALL_OK','phase':phase,'global_step':step,'source_step':source,
 'initial_snapshot_id':initial.snapshot_id,'snapshot_id':actor['snapshot_id'],
 'planning_optimizer_fingerprint':actor['planning_optimizer_fingerprint'],
 'activation_version':owner['activation_version'],
 'checkpoint_steps':sorted(actual_checkpoint_steps),'validation':validation,
}
atomic_json(out/'validator.json',summary)
if phase=='train_to_5':
 atomic_json(run_out/'interim_status.json',{**summary,'status':'needs_resume_to_10'})
else:
 before_rows=[json.loads(line) for line in (run_out/'validation/0.jsonl').read_text().splitlines() if line]
 before=summarize_canary_validation_rows(
  before_rows,expected_data_sources=expected_sources,expected_rows_per_source=8,
  expected_step=0,
 )
 final={
  **summary,'status':'passed','validation_before':before,
  'success_rate_delta':validation['success_rate']-before['success_rate'],
  'reward_mean_delta':validation['reward_mean']-before['reward_mean'],
  'full_heldout_5x60_evaluation_started':False,
  'long_training_started':False,
 }
 atomic_json(run_out/'final_status.json',final)
print(json.dumps(summary,allow_nan=False))
PY

for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
