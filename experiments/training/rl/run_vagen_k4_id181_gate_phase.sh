#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
: "${PHASE:?PHASE must be update_1 or restore_only}"
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
PHASE_NAME=$([[ "${PHASE}" == update_1 ]] && echo phase1_update || echo phase2_fresh_restore_only)
PHASE_TAG=$([[ "${PHASE}" == update_1 ]] && echo p1 || echo p2)
PHASE_OUT=${RUN_OUT}/${PHASE_NAME}
ENV_PORT=$((18800 + SLURM_JOB_ID % 300 + ($([[ "${PHASE}" == update_1 ]] && echo 0 || echo 300))))
ENV_URL=http://127.0.0.1:${ENV_PORT}
RUNTIME_ROOT=/tmp/i180-${SLURM_JOB_ID}-${PHASE_TAG}
RAY_TMPDIR=${RUNTIME_ROOT}
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
TRAIN_PID=
NVIDIA_PID=
PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS:-$([[ "${PHASE}" == update_1 ]] && echo 4200 || echo 1800)}

[[ "${PHASE}" == update_1 || "${PHASE}" == restore_only ]]
[[ "${RUN_NAME}" == 181_gate_k4schemeb_jointupdate_dp8_tp8_train3x8_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095 ]]
[[ "${EXPECTED_VERL_COMMIT}" == 494f264494b2525f2c13595f63ac4912963e6d2f ]]
[[ "${SLURM_JOB_PARTITION:-}" == normal ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 64 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 262144 ]]
if [[ "${PHASE}" == update_1 ]]; then
  [[ "${PHASE_TIMEOUT_SECONDS}" == 4200 ]]
else
  [[ "${PHASE_TIMEOUT_SECONDS}" == 1800 ]]
fi
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
grep -q 'TimeLimit=02:00:00' <<<"${JOB_DETAILS}"
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

if [[ "${PHASE}" == update_1 ]]; then
  [[ ! -e "${RUN_OUT}" ]] || { echo "ID181 output already exists" >&2; exit 2; }
  mkdir -p "${RUN_OUT}" "${CHECKPOINT_DIR}"
else
  [[ -f "${CHECKPOINT_DIR}/global_step_1/joint_checkpoint_complete.json" ]]
  [[ ! -e "${CHECKPOINT_DIR}/global_step_2" ]]
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
export WANDB_RUN_ID=nimloth-id181-k4-single-update-restore-gate
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
  [[ "${RUNTIME_ROOT}" == /tmp/i180-* ]] && rm -rf -- "${RUNTIME_ROOT}"
  exit "${status}"
}
trap cleanup EXIT

"${PY}" - "${VAGEN}" "${PHASE_OUT}" "${ENV_URL}" <<'PY'
import hashlib, json, sys
from pathlib import Path
vagen=Path(sys.argv[1]); out=Path(sys.argv[2]); url=sys.argv[3]
for name in ('train','val'):
 src=vagen/'examples/train/navigation'/f'{name}_navigation_joint_id181.yaml'
 text=src.read_text()
 assert 'http://127.0.0.1:8000' in text
 for reward_field in (
  'per_turn_format_reward: 0.01',
  'format_reward: 0.0',
  'success_reward: 1.0',
 ):
  assert reward_field in text
 dst=out/f'{name}_navigation_joint_id181.yaml'
 dst.write_text(text.replace('http://127.0.0.1:8000',url))
expected={
 'base_train':'eb0aa69186604cedc6dc6c2a8874393beae09b7ac1dadae5458e87492b5e01e9',
 'common_sense_train':'dd74a0f02c48e59efda445a68dc717278ffe6fe828f0a431418f205eb67d403b',
 'long_horizon_train':'27d3c95fc0b73fd7f3b89fb6cbad6a93fd9dc91eb42b0ff636b78ddc1d2499e1',
}
assets={}
for name,digest in expected.items():
 asset=vagen/'vagen/envs/navigation/assets'/f'{name}.json'
 tasks=json.loads(asset.read_text())['tasks']
 assert len(tasks)==1200
 actual=hashlib.sha256(asset.read_bytes()).hexdigest()
 assert actual==digest
 assets[name]={'sha256':actual,'task_count':len(tasks)}
(out/'source_hashes.json').write_text(json.dumps({
 'train_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/train_navigation_joint_id181.yaml').read_bytes()).hexdigest(),
 'val_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/val_navigation_joint_id181.yaml').read_bytes()).hexdigest(),
 'assets':assets,
},indent=2)+'\n')
PY
export ID181_TRAIN_CONFIG=${PHASE_OUT}/train_navigation_joint_id181.yaml
export ID181_VAL_CONFIG=${PHASE_OUT}/val_navigation_joint_id181.yaml
export ID181_ACTOR_MODEL=${ACTOR_MODEL}
export ID181_PLANNING_CHECKPOINT=${PLANNING_MODEL}
export ID181_AGENT_CONFIG=${VAGEN}/vagen/configs/agent_no_concat.yaml
export ID181_RUN_NAME=${RUN_NAME}
export ID181_RUN_OUT=${RUN_OUT}

"${PY}" - "${ID181_TRAIN_CONFIG}" "${PHASE_OUT}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from vagen.gym_agent_dataset import AgenticDataset
config_path=Path(sys.argv[1]); out=Path(sys.argv[2])
dataset=AgenticDataset(data_files=str(config_path),config={'base_seed':0})
rows=[]
for index in range(len(dataset)):
 item=dataset[index]
 rows.append({
  'dataset_index':index,
  'data_source':str(item['data_source']),
  'seed':int(item['seed']),
  'rollout_sample_id':str(item['rollout_sample_id']),
 })
counts=Counter(row['data_source'] for row in rows)
assert len(rows)==24
assert counts=={
 'navigation_base_train_id181':8,
 'navigation_common_sense_train_id181':8,
 'navigation_long_horizon_train_id181':8,
}
assert len({row['rollout_sample_id'] for row in rows})==24
(out/'dataset_manifest.json').write_text(json.dumps({
 'base_seed':0,
 'seed_directive':'inclusive [0,8] sampled deterministically with replacement',
 'counts':dict(sorted(counts.items())),
 'rows':rows,
},indent=2)+'\n')
PY

if [[ "${PHASE}" == update_1 ]]; then
  cat >"${RUN_OUT}/README.md" <<EOF
# ID181 corrected single-update K4 Scheme-B integration gate

- retry provenance: ID180 was cancelled before rollout or update after the agent misclassified a caught optional JIT warning; it did not evaluate the corrected SIGReg update. ID181 keeps the same approved contract and corrected full-module device placement.
- project/run: vagen / ${RUN_NAME}
- approval: one optimizer update followed by fresh-runtime restore-only verification; no canary, validation rollout, second update, or long training. A caught optional JIT warning is not a fatal result; monitoring waits for an authoritative failed future/process, terminal Slurm state, or phase timeout.
- parent/VAGEN/VERL: ${EXPECTED_PARENT_COMMIT} / ${EXPECTED_VAGEN_COMMIT} / ${EXPECTED_VERL_COMMIT}
- data: base_train, common_sense_train, long_horizon_train, 8 deterministic instances per split from the inclusive seed directive [0,8]; 24 complete trajectories; max 20 real actions; per-turn format 0.01, terminal format 0, success 1.
- actor initialization: immutable completed ID176 repaired Qwen checkpoint.
- planning initialization: immutable corrected ID74 projector, horizon-4 predictor, and 8-action ValueHead at source step 776.
- behavior: K4/100 UCT/c1, alpha1, approved beta85.78297006578457, prior temperature1, float32, keyed sampling, CoT temperature0.7/top-p0.95, response cap512.
- update: actor lr1e-7; PPO clip0.2, one epoch, token KL0.01, guided entropy0.01; unified projector/predictor/ValueHead AdamW lr1e-4, state/DINO/SIGReg weights1/0.5/0.1, selected Huber delta1, gamma1, lambda0.95.
- checkpoint: phase1 atomically writes only global_step_1 and full planning snapshot source777; phase2 starts a fresh runtime, restores it exactly, performs zero updates, and must not create global_step_2.
- resources: normal, one node, 8 H800, 64 CPU, 256 GiB, two-hour batch allocation; excluded nodes dgx-13/23/32/37/51.
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

if [[ "${PHASE}" == update_1 ]]; then
  ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases "${AI2THOR_HOME_ROOT}/.ai2thor/releases"
  rm -f "${AI2THOR_HOME_ROOT}/.ai2thor/cuda-vulkan-mapping.json"
  source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh" > >(tee -a "${PHASE_OUT}/controller.log") 2>&1

  timeout --signal=TERM --kill-after=10s 150s "${PY}" -m nimloth.environment.navigation.direct_render_probe --gpu-device 0 | tee "${PHASE_OUT}/render_probe.json"
  cd "${VAGEN}"
  ! ss -ltnH "sport = :${ENV_PORT}" | grep -q .
  "${PY}" -m vagen.envs.navigation.serve --host=127.0.0.1 --port="${ENV_PORT}" --devices='[0]' --max_envs=24 --max_inflight=24 --thread_pool_size=24 --session_timeout=7200 >"${PHASE_OUT}/env_server.log" 2>&1 &
  ENV_PID=$!
  for _ in $(seq 1 90); do
    if curl -fsS --max-time 5 "${ENV_URL}/health" >"${PHASE_OUT}/health.json" 2>/dev/null; then break; fi
    kill -0 "${ENV_PID}" || { tail -100 "${PHASE_OUT}/env_server.log"; exit 4; }
    sleep 2
  done
  curl -fsS --max-time 5 "${ENV_URL}/health" >/dev/null
  for split in base_train common_sense_train long_horizon_train; do
    timeout --signal=TERM --kill-after=10s 300s "${PY}" -m nimloth.environment.navigation.prewarm --env-url "${ENV_URL}" --eval-set "${split}" --seed 0 --timeout-seconds 300 --env-id "id181-prewarm-${split}-${SLURM_JOB_ID}" | tee "${PHASE_OUT}/prewarm_${split}.json"
  done
fi

nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 >"${PHASE_OUT}/nvidia_smi.csv" 2>"${PHASE_OUT}/nvidia_smi.err" &
NVIDIA_PID=$!

PHASE_OVERRIDES=()
if [[ "${PHASE}" == restore_only ]]; then
  PHASE_OVERRIDES+=(joint_integration_gate.phase=restore_only trainer.resume_mode=auto)
fi
COMMAND=("${PY}" -m vagen.main_ppo --config-path="${VAGEN}/vagen/configs" --config-name=joint_id181_gate "hydra.run.dir=${PHASE_OUT}/hydra" hydra.job.chdir=false "${PHASE_OVERRIDES[@]}")
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

EXPECTED_STEP=1
EXPECTED_SOURCE=777
"${PY}" - "${CHECKPOINT_DIR}" "${EXPECTED_STEP}" "${EXPECTED_SOURCE}" "${PHASE}" "${PHASE_OUT}" "${RUN_OUT}/planning_snapshots" <<'PY'
import json, math, sys
from pathlib import Path
import torch
from vagen.joint_policy.checkpoint import load_complete_joint_checkpoint
from nimloth.training.rl.joint_planner import load_frozen_planning_snapshot_file
root=Path(sys.argv[1]); step=int(sys.argv[2]); source=int(sys.argv[3]); phase=sys.argv[4]; out=Path(sys.argv[5]); snapshots=Path(sys.argv[6])
folder=root/f'global_step_{step}'
payload=load_complete_joint_checkpoint(folder)
actor=payload['actor_critic']; owner=payload['frozen_q_owner']; active=owner['active_snapshot_state']
assert payload['global_step']==step
assert payload['run_seed']==42179
assert actor['schema']=='vagen_joint_k4_actor_planning_checkpoint_v1'
assert actor['completed_updates']==step
assert actor['source_step']==source
assert actor['score_dtype']=='float32'
assert actor['planning_optimizer_state']['state']
assert actor['planning_optimizer_fingerprint']
assert owner['activation_version']==step
assert active['schema']=='vagen_frozen_k4_planner_transport_v1'
assert active['snapshot_id']==actor['snapshot_id']
assert active['snapshot_source_step']==source
assert actor['snapshot_transport']==active
assert Path(active['transport_path']).is_file()
initial_path=snapshots/'source_step_776/frozen_k4_planner.pt'
updated_path=snapshots/'source_step_777/frozen_k4_planner.pt'
assert initial_path.is_file() and updated_path.is_file()
initial=load_frozen_planning_snapshot_file(initial_path,device=torch.device('cpu'))
updated=load_frozen_planning_snapshot_file(updated_path,device=torch.device('cpu'))
assert initial.source_step==776 and updated.source_step==777
assert updated.snapshot_id==actor['snapshot_id']
assert initial.snapshot_id!=updated.snapshot_id
assert not (root/'global_step_2').exists()
if phase=='restore_only':
 log=(out/'train.log').read_text()
 assert 'global_step_1' in log and 'Setting global step to 1' in log
 assert 'ID181_K4_FRESH_RESTORE_ONLY_ALL_OK global_step=1' in log
summary={
 'status':'ALL_OK','phase':phase,'global_step':step,'source_step':source,
 'initial_snapshot_id':initial.snapshot_id,'snapshot_id':actor['snapshot_id'],
 'planning_optimizer_fingerprint':actor['planning_optimizer_fingerprint'],
 'activation_version':owner['activation_version'],'global_step_2_exists':False,
}
(out/'validator.json').write_text(json.dumps(summary,indent=2)+'\n')
if phase=='restore_only':
 (out.parent/'final_status.json').write_text(json.dumps({**summary,'status':'passed'},indent=2)+'\n')
print(json.dumps(summary))
PY

for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm" "${REPO}/external/RCDM"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
