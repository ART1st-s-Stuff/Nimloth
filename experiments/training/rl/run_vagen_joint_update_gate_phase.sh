#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/peilab/atst/nimloth
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
: "${PHASE:?PHASE must be update_1 or resume_update_2}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${RUN_DATE:?RUN_DATE is required}"
VAGEN=${REPO}/external/VAGEN
VERL=${VAGEN}/verl
PY=${ROOT}/.venv-vagen-main/bin/python3
MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}
CHECKPOINT_DIR=${RUN_OUT}/checkpoints
PHASE_NAME=$([[ "${PHASE}" == update_1 ]] && echo phase1_update || echo phase2_resume_update)
PHASE_OUT=${RUN_OUT}/${PHASE_NAME}
ENV_PORT=$((18800 + SLURM_JOB_ID % 300 + ($([[ "${PHASE}" == update_1 ]] && echo 0 || echo 300))))
ENV_URL=http://127.0.0.1:${ENV_PORT}
RUNTIME_ROOT=/tmp/id165-${SLURM_JOB_ID}-${PHASE_NAME}
RAY_TMPDIR=${RUNTIME_ROOT}/ray
TMPDIR=${RUNTIME_ROOT}/tmp
AI2THOR_HOME_ROOT=${RUNTIME_ROOT}/ai2thor
ENV_PID=
TRAIN_PID=
NVIDIA_PID=
PHASE_TIMEOUT_SECONDS=${PHASE_TIMEOUT_SECONDS:-1600}

[[ "${PHASE}" == update_1 || "${PHASE}" == resume_update_2 ]]
[[ "${RUN_NAME}" == 165_smoke_vagenlite_jointupdate_dp8_tp8_base_train8_t2_a1b1_g099_l095_clip02_akl001_ent001 ]]
[[ "${EXPECTED_VERL_COMMIT}" == 42cb2f129357ffdd2c58f338d78da4dc91e3412e ]]
[[ "${SLURM_JOB_PARTITION:-}" == normal ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 64 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 262144 ]]
[[ "${PHASE_TIMEOUT_SECONDS}" == 1600 ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
IFS=, read -r -a VISIBLE_GPUS <<<"${CUDA_VISIBLE_DEVICES}"
(( ${#VISIBLE_GPUS[@]} == 8 ))
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == 8 ))
for name in "${GPU_NAMES[@]}"; do [[ "${name}" == *H800* ]]; done
[[ "$(hostname)" != dgx-51 ]]

JOB_DETAILS=$(scontrol show job -dd "${SLURM_JOB_ID}" -o)
grep -q 'Partition=normal' <<<"${JOB_DETAILS}"
grep -q 'TimeLimit=01:00:00' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*mem=256G' <<<"${JOB_DETAILS}"

[[ "$(git -C "${REPO}" rev-parse HEAD)" == "${EXPECTED_PARENT_COMMIT}" ]]
[[ "$(git -C "${VAGEN}" rev-parse HEAD)" == "${EXPECTED_VAGEN_COMMIT}" ]]
[[ "$(git -C "${VERL}" rev-parse HEAD)" == "${EXPECTED_VERL_COMMIT}" ]]
[[ "$(git -C "${REPO}/external/le-wm" rev-parse HEAD)" == 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac ]]
for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
[[ -x "${PY}" ]]
[[ -d "${MODEL}" ]]

if [[ "${PHASE}" == update_1 ]]; then
  [[ ! -e "${RUN_OUT}" ]] || { echo "ID165 output already exists" >&2; exit 2; }
  mkdir -p "${RUN_OUT}" "${CHECKPOINT_DIR}"
else
  [[ -f "${CHECKPOINT_DIR}/global_step_1/joint_checkpoint_complete.json" ]]
  [[ ! -e "${CHECKPOINT_DIR}/global_step_2/joint_checkpoint_complete.json" ]]
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
export WANDB_RUN_ID=nimloth-id165-joint-update-gate
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
  [[ "${RUNTIME_ROOT}" == /tmp/id165-* ]] && rm -rf -- "${RUNTIME_ROOT}"
  exit "${status}"
}
trap cleanup EXIT

"${PY}" - "${VAGEN}" "${PHASE_OUT}" "${ENV_URL}" <<'PY'
import hashlib, json, sys
from pathlib import Path
vagen=Path(sys.argv[1]); out=Path(sys.argv[2]); url=sys.argv[3]
for name in ('train','val'):
 src=vagen/'examples/train/navigation'/f'{name}_navigation_joint_id165.yaml'
 text=src.read_text()
 assert 'http://127.0.0.1:8000' in text
 dst=out/f'{name}_navigation_joint_id165.yaml'
 dst.write_text(text.replace('http://127.0.0.1:8000',url))
(out/'source_hashes.json').write_text(json.dumps({
 'train_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/train_navigation_joint_id165.yaml').read_bytes()).hexdigest(),
 'val_config_sha256':hashlib.sha256((vagen/'examples/train/navigation/val_navigation_joint_id165.yaml').read_bytes()).hexdigest(),
},indent=2)+'\n')
PY
export ID165_TRAIN_CONFIG=${PHASE_OUT}/train_navigation_joint_id165.yaml
export ID165_VAL_CONFIG=${PHASE_OUT}/val_navigation_joint_id165.yaml
export ID165_MODEL=${MODEL}
export ID165_AGENT_CONFIG=${VAGEN}/vagen/configs/agent_no_concat.yaml
export ID165_RUN_NAME=${RUN_NAME}
export ID165_RUN_OUT=${RUN_OUT}

if [[ "${PHASE}" == update_1 ]]; then
  cat >"${RUN_OUT}/README.md" <<EOF
# ID165 DP8 joint-update and exact-resume smoke

- project/run: vagen / ${RUN_NAME}
- purpose: non-production two-phase target-DP8 joint update and resume gate
- parent/VAGEN/VERL: ${EXPECTED_PARENT_COMMIT} / ${EXPECTED_VAGEN_COMMIT} / ${EXPECTED_VERL_COMMIT}
- data: Navigation base_train seeds 0..7, 8 trajectories, max turns 2; validation disabled
- initialization: corrected ID74 policy plus state projector and ValueHead at source step 776
- train: Qwen actor and DP8 replicated current projector+ValueHead critic
- frozen: vision tower, reference policy, rollout-time CPU frozen-Q snapshot
- test-only values: alpha/beta/priorT=1/1/1, float32, seed42001, gamma0.99, lambda0.95, clip0.2, actor lr1e-7, critic lr1e-4, KL/entropy0.01/0.01
- checkpoint: every complete global update; phase1 writes step1, phase2 must exact-resume step1 and write step2
- resources: normal one node, 8 H800, 64 CPU, 256 GiB, 60 minute hold; dgx-51 excluded
EOF
fi

"${PY}" - "${MODEL}" "${PHASE_OUT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
model=Path(sys.argv[1]); out=Path(sys.argv[2])
required=['config.json','model.safetensors.index.json','state_proj.pt','value_head/value_head.pt','training_state.pt']
result={}
for rel in required:
 p=model/rel
 assert p.is_file(), p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 result[rel]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(out/'checkpoint_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
PY

ln -s /project/peilab/atst/flower/.ai2thor-home/.ai2thor/releases "${AI2THOR_HOME_ROOT}/.ai2thor/releases"
rm -f "${AI2THOR_HOME_ROOT}/.ai2thor/cuda-vulkan-mapping.json"
source "${REPO}/experiments/training/baseline/setup_ai2thor_env.sh" > >(tee -a "${PHASE_OUT}/controller.log") 2>&1

timeout --signal=TERM --kill-after=10s 150s "${PY}" -m nimloth.environment.navigation.direct_render_probe --gpu-device 0 | tee "${PHASE_OUT}/render_probe.json"
cd "${VAGEN}"
! ss -ltnH "sport = :${ENV_PORT}" | grep -q .
"${PY}" -m vagen.envs.navigation.serve --host=127.0.0.1 --port="${ENV_PORT}" --devices='[0]' --max_envs=8 --max_inflight=8 --thread_pool_size=8 --session_timeout=1800 >"${PHASE_OUT}/env_server.log" 2>&1 &
ENV_PID=$!
for _ in $(seq 1 90); do
  if curl -fsS --max-time 5 "${ENV_URL}/health" >"${PHASE_OUT}/health.json" 2>/dev/null; then break; fi
  kill -0 "${ENV_PID}" || { tail -100 "${PHASE_OUT}/env_server.log"; exit 4; }
  sleep 2
done
curl -fsS --max-time 5 "${ENV_URL}/health" >/dev/null

timeout --signal=TERM --kill-after=10s 300s "${PY}" -m nimloth.environment.navigation.prewarm --env-url "${ENV_URL}" --eval-set base_train --seed 0 --timeout-seconds 300 --env-id "id165-prewarm-${PHASE}-${SLURM_JOB_ID}" | tee "${PHASE_OUT}/prewarm.json"

nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 1 >"${PHASE_OUT}/nvidia_smi.csv" 2>"${PHASE_OUT}/nvidia_smi.err" &
NVIDIA_PID=$!

PHASE_OVERRIDES=()
if [[ "${PHASE}" == resume_update_2 ]]; then
  PHASE_OVERRIDES+=(joint_integration_gate.phase=resume_update_2 trainer.total_training_steps=2 trainer.total_epochs=2 trainer.resume_mode=auto)
fi
COMMAND=("${PY}" -m vagen.main_ppo --config-path="${VAGEN}/vagen/configs" --config-name=joint_id165_gate "hydra.run.dir=${PHASE_OUT}/hydra" hydra.job.chdir=false "${PHASE_OVERRIDES[@]}")
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

EXPECTED_STEP=$([[ "${PHASE}" == update_1 ]] && echo 1 || echo 2)
EXPECTED_SOURCE=$((776 + EXPECTED_STEP))
"${PY}" - "${CHECKPOINT_DIR}" "${EXPECTED_STEP}" "${EXPECTED_SOURCE}" "${PHASE}" "${PHASE_OUT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
from vagen.joint_policy.checkpoint import load_complete_joint_checkpoint
root=Path(sys.argv[1]); step=int(sys.argv[2]); source=int(sys.argv[3]); phase=sys.argv[4]; out=Path(sys.argv[5])
folder=root/f'global_step_{step}'
payload=load_complete_joint_checkpoint(folder)
actor=payload['actor_critic']; owner=payload['frozen_q_owner']
assert payload['global_step']==step
assert payload['run_seed']==42001
assert actor['completed_updates']==step
assert actor['source_step']==source
assert owner['activation_version']==step
assert owner['active_snapshot_state']['snapshot_id']==actor['snapshot_id']
assert owner['active_snapshot_state']['source_step']==source
if phase=='resume_update_2':
 log=(out/'train.log').read_text()
 assert 'global_step_1' in log and 'Setting global step to 1' in log
 assert (root/'global_step_1/joint_checkpoint_complete.json').is_file()
summary={'status':'ALL_OK','phase':phase,'global_step':step,'source_step':source,'snapshot_id':actor['snapshot_id'],'optimizer_fingerprint':actor['critic_optimizer_fingerprint'],'activation_version':owner['activation_version']}
(out/'validator.json').write_text(json.dumps(summary,indent=2)+'\n')
if phase=='resume_update_2':
 (out.parent/'final_status.json').write_text(json.dumps({**summary,'status':'passed'},indent=2)+'\n')
print(json.dumps(summary))
PY

for source_repo in "${REPO}" "${VAGEN}" "${VERL}" "${REPO}/external/le-wm"; do
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]]
done
