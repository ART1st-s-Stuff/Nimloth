#!/usr/bin/env bash
# Launch ID184 from the complete ID183 step10 checkpoint on exact 4x2 H800.
set -euo pipefail

HOLD_JOB=${1:?usage: launch_vagen_k4_id184_continue_to20_on_hold.sh HOLD_JOB}
: "${REPO:?REPO is required}"
: "${EXPECTED_PARENT_COMMIT:?EXPECTED_PARENT_COMMIT is required}"
: "${EXPECTED_VAGEN_COMMIT:?EXPECTED_VAGEN_COMMIT is required}"
: "${EXPECTED_VERL_COMMIT:?EXPECTED_VERL_COMMIT is required}"
[[ "${EXPECTED_VERL_COMMIT}" == 494f264494b2525f2c13595f63ac4912963e6d2f ]]

ROOT=/project/peilab/atst/nimloth
PY=${ROOT}/.venv-vagen-main/bin/python3
SLURM_BIN_DIR=/cm/shared/apps/slurm/current/bin
SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
[[ -x "${SLURM_BIN_DIR}/scontrol" && -x "${SLURM_BIN_DIR}/srun" ]]
[[ -r "${SLURM_CONF}" ]]
RUN_NAME=184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8
RUN_DATE=2026-08-17
RUN_OUTPUT_SUFFIX=_retry1
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}${RUN_OUTPUT_SUFFIX}
RUNNER=${REPO}/experiments/training/rl/run_vagen_k4_id184_continue_to20.sh
PHASE_TAG=c20
PHASE_NAME=continue_step10_to20
PHASE_OUT=${RUN_OUT}/${PHASE_NAME}
RUNTIME_ROOT=/tmp/i184-${HOLD_JOB}-${PHASE_TAG}
RAY_PORT=$((22000 + HOLD_JOB % 10000))
RAY_CLUSTER_ROOT=/tmp/i184-ray-${HOLD_JOB}-${PHASE_TAG}
RAY_LOG_ROOT=${ROOT}/outputs/experiments/training/rl/slurm/id184-ray-${HOLD_JOB}-${PHASE_TAG}
RAY_PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl
mkdir -p "${ROOT}/outputs/experiments/training/rl/slurm"
exec 9>"${ROOT}/outputs/experiments/training/rl/slurm/.id184-${HOLD_JOB}-${PHASE_TAG}.launch.lock"
flock -n 9 || { echo "duplicate ID184 launcher for job ${HOLD_JOB}" >&2; exit 94; }
ACTOR_MODEL=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint
PLANNING_MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
ID184_SOURCE_CHECKPOINT=${ROOT}/outputs/experiments/training/rl/2026-08-16/183_canary_k4schemeb_jointupdate_dp8_tp8_u10_r5_train3x8_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8_retry2/checkpoints/global_step_10
source "${REPO}/experiments/training/rl/slurm_allocation.sh"
set -a
source /project/peilab/atst/flower/.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
WANDB_RESUME=never

[[ -x "${PY}" && -x "${RUNNER}" ]]
JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")
grep -q 'JobState=RUNNING' <<<"${JOB_DETAILS}"
grep -q 'Partition=normal' <<<"${JOB_DETAILS}"
grep -q 'NumNodes=4' <<<"${JOB_DETAILS}"
grep -q 'TimeLimit=05:00:00' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*mem=256G([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -q 'MinMemoryNode=64G' <<<"${JOB_DETAILS}"

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o '%N')")
[[ ${#NODES[@]} -eq 4 ]]
for node in "${NODES[@]}"; do
  for excluded in dgx-09 dgx-13 dgx-32 dgx-51; do
    [[ "${node}" != "${excluded}" ]]
  done
done
NAVIGATION_HEAD_EXCLUSIONS=(dgx-09 dgx-13 dgx-23 dgx-32 dgx-37 dgx-51)
HEAD_NODE=
for node in "${NODES[@]}"; do
  allowed=true
  for excluded in "${NAVIGATION_HEAD_EXCLUSIONS[@]}"; do
    if [[ "${node}" == "${excluded}" ]]; then allowed=false; break; fi
  done
  if [[ "${allowed}" == true ]]; then HEAD_NODE=${node}; break; fi
done
[[ -n "${HEAD_NODE}" ]]
WORKER_NODES=()
for node in "${NODES[@]}"; do
  [[ "${node}" == "${HEAD_NODE}" ]] || WORKER_NODES+=("${node}")
done
[[ ${#WORKER_NODES[@]} -eq 3 ]]
CLUSTER_NODES=("${HEAD_NODE}" "${WORKER_NODES[@]}")

declare -A GPU_COUNTS
nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
for node in "${NODES[@]}"; do
  [[ "${GPU_COUNTS[${node}]:-}" == 2 ]]
done

mapfile -t FABRIC_ROWS < <(
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --gpus=0 bash -lc '
      row=$(ip -o -4 addr show | awk '\''$4 ~ /^10\.23\./ {split($4,a,"/"); print a[1],$2; exit}'\'')
      [[ -n "${row}" ]]
      printf "%s %s\n" "$(hostname -s)" "${row}"
    '
)
[[ ${#FABRIC_ROWS[@]} -eq 4 ]]
declare -A NODE_IP NODE_IFACE
for row in "${FABRIC_ROWS[@]}"; do
  read -r node ip iface <<<"${row}"
  [[ -z "${NODE_IP[${node}]:-}" ]]
  NODE_IP[${node}]=${ip}
  NODE_IFACE[${node}]=${iface}
done
HEAD_IP=${NODE_IP[${HEAD_NODE}]}
FABRIC_IFACE=${NODE_IFACE[${HEAD_NODE}]}
EXPECTED_NODE_IPS=()
for node in "${CLUSTER_NODES[@]}"; do
  [[ -n "${NODE_IP[${node}]:-}" ]]
  [[ "${NODE_IFACE[${node}]}" == "${FABRIC_IFACE}" ]]
  EXPECTED_NODE_IPS+=("${NODE_IP[${node}]}")
done
RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}
RAY_EXPECTED_NODE_IPS=$(IFS=,; echo "${EXPECTED_NODE_IPS[*]}")
ID184_CLUSTER_NODES=$(IFS=,; echo "${CLUSTER_NODES[*]}")

timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
  --ntasks-per-node=1 --gres=gpu:2 bash -lc '
    set -euo pipefail
    mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ ${#names[@]} -eq 2 ]]
    for name in "${names[@]}"; do [[ "${name}" == *H800* ]]; done
    printf "%s gpu_count=%s names=%s\n" "$(hostname -s)" "${#names[@]}" "${names[*]}"
  '

timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
  --ntasks-per-node=1 --gpus=0 \
  env RAY_CLUSTER_ROOT="${RAY_CLUSTER_ROOT}" RUNTIME_ROOT="${RUNTIME_ROOT}" \
  bash -lc '
    [[ ! -e "${RAY_CLUSTER_ROOT}" && ! -e "${RUNTIME_ROOT}" ]]
    mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/ai2thor"
  '
timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" --gpus=0 env HEAD_IP="${HEAD_IP}" RAY_PORT="${RAY_PORT}" \
  "${PY}" - <<'PY'
import os, socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind((os.environ['HEAD_IP'], int(os.environ['RAY_PORT'])))
PY

mkdir -p "${RAY_LOG_ROOT}"
printf '%s\n' "${JOB_DETAILS}" >"${RAY_LOG_ROOT}/allocation.txt"
printf '%s\n' "${FABRIC_ROWS[@]}" >"${RAY_LOG_ROOT}/fabric.txt"
RAY_STEP_PIDS=()

node_runtime_processes() {
  local signal=${1:?signal required}
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --gpus=0 \
    env TARGET_ROOTS="${RAY_CLUSTER_ROOT},${RUNTIME_ROOT}" SIGNAL="${signal}" "${PY}" - <<'PY'
import os, signal
from pathlib import Path
roots=[value.encode() for value in os.environ['TARGET_ROOTS'].split(',')]
sig=getattr(signal,'SIG'+os.environ['SIGNAL'])
ancestors=set(); pid=os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try: pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
    except (FileNotFoundError,PermissionError,ValueError,IndexError): break
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try: evidence=(entry/'environ').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if any(root in evidence for root in roots):
        try: os.kill(int(entry.name),sig)
        except (ProcessLookupError,PermissionError): pass
PY
}

audit_node_runtime_empty() {
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --gpus=0 \
    env TARGET_ROOTS="${RAY_CLUSTER_ROOT},${RUNTIME_ROOT}" "${PY}" - <<'PY'
import os, socket
from pathlib import Path
roots=[value.encode() for value in os.environ['TARGET_ROOTS'].split(',')]
found=[]; ancestors=set(); pid=os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try: pid=int((Path('/proc')/str(pid)/'stat').read_text().split()[3])
    except (FileNotFoundError,PermissionError,ValueError,IndexError): break
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try: evidence=(entry/'environ').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if any(root in evidence for root in roots): found.append(int(entry.name))
print(f'{socket.gethostname()} owned_pids={found}')
if found: raise SystemExit(1)
PY
}

RAY_LOG_CAPTURE_SCRIPT=${RAY_LOG_ROOT}/capture_ray_session_logs.py
cat >"${RAY_LOG_CAPTURE_SCRIPT}" <<'PY'
import json, os, socket, tempfile
from pathlib import Path
stage=os.environ['CAPTURE_STAGE']
root=Path(os.environ['RAY_CLUSTER_ROOT'])
shared=Path(os.environ['SHARED_LOG_ROOT'])
host=socket.gethostname().split('.')[0]
destination=shared/'ray_internal'/stage/host
destination.mkdir(parents=True,exist_ok=True)
session=(root/'session_latest').resolve()
logs=session/'logs'
fixed=(
 'ray_process_exit.log','gcs_server.err','gcs_server.out',
 'raylet.err','raylet.out','monitor.err','monitor.out',
 'dashboard.err','dashboard.log','dashboard_agent.err','dashboard_agent.log',
 'runtime_env_agent.err','runtime_env_agent.log',
)
candidates=[]
for name in fixed:
 path=logs/name
 if path.is_file(): candidates.append(path)
if logs.is_dir():
 for path in sorted(logs.glob('*.err')):
  if path.is_file() and path not in candidates: candidates.append(path)
max_file_bytes=2*1024*1024
max_total_bytes=16*1024*1024
captured_total=0
files=[]
errors=[]
for source in candidates:
 if captured_total>=max_total_bytes: break
 try:
  original=source.stat().st_size
  take=min(original,max_file_bytes,max_total_bytes-captured_total)
  with source.open('rb') as handle:
   if original>take: handle.seek(original-take)
   payload=handle.read(take)
  target=destination/source.name
  target.write_bytes(payload)
 except OSError as exc:
  errors.append({'name':source.name,'error':repr(exc)})
  continue
 captured_total+=len(payload)
 files.append({
  'name':source.name,'original_bytes':original,
  'captured_bytes':len(payload),'tail_truncated':original>len(payload),
 })
manifest={
 'stage':stage,'host':host,'root_exists':root.exists(),
 'session':str(session),'session_exists':session.exists(),
 'logs_exists':logs.is_dir(),'captured_total_bytes':captured_total,
 'max_total_bytes':max_total_bytes,'files':files,'errors':errors,
}
fd,name=tempfile.mkstemp(prefix='.capture_manifest.',suffix='.tmp',dir=destination)
with os.fdopen(fd,'w',encoding='utf-8') as handle:
 json.dump(manifest,handle,indent=2); handle.write('\n')
os.replace(name,destination/'capture_manifest.json')
print(json.dumps(manifest,sort_keys=True),flush=True)
if errors: raise SystemExit(1)
PY
RAY_LOG_CAPTURE_COMPLETE=true

persist_ray_logs() {
  local stage=${1:?capture stage required}
  local capture_out=${RAY_LOG_ROOT}/ray_log_capture_${stage}.out
  local capture_err=${RAY_LOG_ROOT}/ray_log_capture_${stage}.err
  local failed=false
  if ! timeout 60s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --gpus=0 \
    env CAPTURE_STAGE="${stage}" RAY_CLUSTER_ROOT="${RAY_CLUSTER_ROOT}" \
    SHARED_LOG_ROOT="${RAY_LOG_ROOT}" "${PY}" "${RAY_LOG_CAPTURE_SCRIPT}" \
    >"${capture_out}" 2>"${capture_err}"
  then
    failed=true
  fi
  for node in "${NODES[@]}"; do
    manifest=${RAY_LOG_ROOT}/ray_internal/${stage}/${node}/capture_manifest.json
    [[ -f "${manifest}" ]] || failed=true
  done
  if [[ "${failed}" == true ]]; then
    printf 'Ray log capture failed or incomplete: stage=%s\n' "${stage}" \
      >"${RAY_LOG_ROOT}/ray_log_capture_${stage}.failed"
    return 1
  fi
}

cleanup_cluster() {
  local status=$?
  trap - EXIT
  set +e
  persist_ray_logs pre_cleanup || RAY_LOG_CAPTURE_COMPLETE=false
  for pid in "${RAY_STEP_PIDS[@]}"; do kill -TERM "${pid}" >/dev/null 2>&1 || true; done
  for _ in $(seq 1 30); do
    alive=false
    for pid in "${RAY_STEP_PIDS[@]}"; do kill -0 "${pid}" >/dev/null 2>&1 && alive=true; done
    [[ "${alive}" == false ]] && break
    sleep 1
  done
  for pid in "${RAY_STEP_PIDS[@]}"; do
    kill -KILL "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  done
  node_runtime_processes TERM >/dev/null 2>&1 || true
  sleep 5
  node_runtime_processes KILL >/dev/null 2>&1 || true
  sleep 2
  cleanup_empty=false
  for _ in $(seq 1 20); do
    if audit_node_runtime_empty >"${RAY_LOG_ROOT}/owned_processes_after.log" 2>&1; then
      cleanup_empty=true
      break
    fi
    node_runtime_processes KILL >/dev/null 2>&1 || true
    sleep 1
  done
  if [[ "${cleanup_empty}" != true && "${status}" -eq 0 ]]; then status=93; fi
  persist_ray_logs post_cleanup || RAY_LOG_CAPTURE_COMPLETE=false
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --gpus=0 \
    env RAY_CLUSTER_ROOT="${RAY_CLUSTER_ROOT}" RUNTIME_ROOT="${RUNTIME_ROOT}" \
    RAY_LOG_CAPTURE_COMPLETE="${RAY_LOG_CAPTURE_COMPLETE}" bash -lc '
      [[ "${RAY_CLUSTER_ROOT}" == /tmp/i184-ray-* ]]
      [[ "${RUNTIME_ROOT}" == /tmp/i184-* ]]
      rm -rf -- "${RUNTIME_ROOT}"
      if [[ "${RAY_LOG_CAPTURE_COMPLETE}" == true ]]; then
        rm -rf -- "${RAY_CLUSTER_ROOT}"
      fi
    ' \
    >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup_cluster EXIT

# These long-lived raylet and driver steps intentionally share the job's CPU,
# memory and GRES allocation. Slurm --overlap is required on every such step;
# Ray's own resource accounting keeps the eight worker actors within 2+2+2+2 GPUs.
COMMON_ENV=(
  PATH="${SLURM_BIN_DIR}:${ROOT}/.venv-vagen-main/bin:/usr/bin:/bin"
  SLURM_CONF="${SLURM_CONF}"
  PYTHONPATH="${RAY_PYTHONPATH}"
  PYTHONDONTWRITEBYTECODE=1
  RAY_TMPDIR="${RAY_CLUSTER_ROOT}"
  TMPDIR="${RUNTIME_ROOT}/tmp"
  AI2THOR_HOME_ROOT="${RUNTIME_ROOT}/ai2thor"
  HF_HOME=/project/peilab/atst/.cache/huggingface
  TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
  TORCH_HOME=/project/peilab/atst/flower/.cache/torch
  TOKENIZERS_PARALLELISM=true
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  VLLM_ALLREDUCE_USE_SYMM_MEM=0
  VLLM_USE_FLASHINFER_SAMPLER=0
  NIMLOTH_LATENT_TOKEN_COUNT=16
  NCCL_SOCKET_IFNAME="${FABRIC_IFACE}"
  GLOO_SOCKET_IFNAME="${FABRIC_IFACE}"
  WANDB_ENTITY=art2nd-hong-kong-university-of-science-and-technology
  WANDB_PROJECT=vagen
  WANDB_NAME="${RUN_NAME}"
  WANDB_RUN_ID=nimloth-id184-k4-continue-to20-retry1
  WANDB_RESUME="${WANDB_RESUME}"
  WANDB_DIR="${RAY_LOG_ROOT}/wandb"
  ID184_TRAIN_CONFIG="${PHASE_OUT}/train_navigation_joint_id184.yaml"
  ID184_VAL_CONFIG="${PHASE_OUT}/val_navigation_joint_id184.yaml"
  ID184_ACTOR_MODEL="${ACTOR_MODEL}"
  ID184_PLANNING_CHECKPOINT="${PLANNING_MODEL}"
  ID184_SOURCE_CHECKPOINT="${ID184_SOURCE_CHECKPOINT}"
  ID184_AGENT_CONFIG="${REPO}/external/VAGEN/vagen/configs/agent_no_concat.yaml"
  ID184_RUN_NAME="${RUN_NAME}"
  ID184_RUN_OUT="${RUN_OUT}"
  RAY_agent_register_timeout_ms=120000
)

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" --gres=gpu:2 --cpus-per-task=16 \
  env "${COMMON_ENV[@]}" VLLM_HOST_IP="${HEAD_IP}" \
  "${PY}" -m ray.scripts.scripts start --block --head \
    --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
    --temp-dir="${RAY_CLUSTER_ROOT}" --num-cpus=16 --num-gpus=2 \
    --object-store-memory=10000000000 \
    --system-config='{"agent_register_timeout_ms":120000}' \
    --include-dashboard=false --disable-usage-stats \
  >"${RAY_LOG_ROOT}/${HEAD_NODE}.log" 2>&1 &
RAY_STEP_PIDS+=("$!")
head_ready=false
for _ in $(seq 1 90); do
  if grep -q 'Ray runtime started' "${RAY_LOG_ROOT}/${HEAD_NODE}.log"; then head_ready=true; break; fi
  kill -0 "${RAY_STEP_PIDS[0]}" 2>/dev/null || { tail -200 "${RAY_LOG_ROOT}/${HEAD_NODE}.log"; exit 5; }
  sleep 2
done
[[ "${head_ready}" == true ]]

for node in "${WORKER_NODES[@]}"; do
  worker_ip=${NODE_IP[${node}]}
  srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
    -w "${node}" --gres=gpu:2 --cpus-per-task=16 \
    env "${COMMON_ENV[@]}" VLLM_HOST_IP="${worker_ip}" \
    "${PY}" -m ray.scripts.scripts start --block \
      --address="${RAY_ADDRESS}" --node-ip-address="${worker_ip}" \
      --temp-dir="${RAY_CLUSTER_ROOT}" --num-cpus=16 --num-gpus=2 \
      --object-store-memory=10000000000 --disable-usage-stats \
    >"${RAY_LOG_ROOT}/${node}.log" 2>&1 &
  RAY_STEP_PIDS+=("$!")
done

cluster_ready=false
for _ in $(seq 1 30); do
  if timeout 10s srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
    -w "${HEAD_NODE}" --gpus=0 env RAY_ADDRESS="${RAY_ADDRESS}" \
    RAY_EXPECTED_NODE_IPS="${RAY_EXPECTED_NODE_IPS}" "${PY}" - <<'PY' >/dev/null 2>&1
import os, ray
ray.init(address=os.environ['RAY_ADDRESS'])
alive=[node for node in ray.nodes() if node['Alive']]
counts=sorted(float(node['Resources'].get('GPU',0)) for node in alive)
addresses=sorted(str(node['NodeManagerAddress']) for node in alive)
ray.shutdown()
if counts != [2.0, 2.0, 2.0, 2.0] or addresses != sorted(os.environ['RAY_EXPECTED_NODE_IPS'].split(',')):
    raise RuntimeError((counts,addresses))
PY
  then cluster_ready=true; break; fi
  for pid in "${RAY_STEP_PIDS[@]}"; do kill -0 "${pid}" 2>/dev/null || { tail -200 "${RAY_LOG_ROOT}"/*.log; exit 6; }; done
  sleep 2
done
[[ "${cluster_ready}" == true ]]

timeout 120s srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" --gpus=0 env RAY_ADDRESS="${RAY_ADDRESS}" \
  RAY_EXPECTED_NODE_IPS="${RAY_EXPECTED_NODE_IPS}" \
  EXPECTED_WANDB_RESUME="${WANDB_RESUME}" "${PY}" - <<'PY' | tee "${RAY_LOG_ROOT}/cluster_probe.json"
import json, os, ray
ray.init(address=os.environ['RAY_ADDRESS'])
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
@ray.remote(num_cpus=0)
def probe_node():
    import nimloth, vagen
    from verl.utils.import_utils import load_extern_type
    dataset_type=load_extern_type(
        'pkg://vagen.gym_agent_dataset',
        'AgenticDataset',
    )
    address=ray.util.get_node_ip_address()
    return {
        'address':address,
        'vllm_host_ip':os.environ.get('VLLM_HOST_IP'),
        'nccl_socket_ifname':os.environ.get('NCCL_SOCKET_IFNAME'),
        'hf_home':os.environ.get('HF_HOME'),
        'torch_home':os.environ.get('TORCH_HOME'),
        'vllm_worker_multiproc_method':os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'),
        'id184_train_config':os.environ.get('ID184_TRAIN_CONFIG'),
        'id184_source_checkpoint':os.environ.get('ID184_SOURCE_CHECKPOINT'),
        'wandb_run_id':os.environ.get('WANDB_RUN_ID'),
        'wandb_resume':os.environ.get('WANDB_RESUME'),
        'wandb_api_key_present':bool(os.environ.get('WANDB_API_KEY')),
        'nimloth':str(nimloth.__file__),
        'vagen':str(vagen.__file__),
        'dataset_type':f'{dataset_type.__module__}.{dataset_type.__name__}',
    }
probes=ray.get([
    probe_node.options(resources={f"node:{row['address']}":0.001}).remote()
    for row in rows
])
print(json.dumps({
    'nodes':rows,
    'expected_addresses':sorted(os.environ['RAY_EXPECTED_NODE_IPS'].split(',')),
    'import_probes':sorted(probes,key=lambda row:row['address']),
}))
assert [row['gpus'] for row in rows] == [2.0,2.0,2.0,2.0]
assert [row['address'] for row in rows] == sorted(os.environ['RAY_EXPECTED_NODE_IPS'].split(','))
assert all(row['address']==row['vllm_host_ip'] for row in probes)
assert all(row['nccl_socket_ifname'] for row in probes)
assert all(row['hf_home']=='/project/peilab/atst/.cache/huggingface' for row in probes)
assert all(row['torch_home']=='/project/peilab/atst/flower/.cache/torch' for row in probes)
assert all(row['vllm_worker_multiproc_method']=='spawn' for row in probes)
assert all(row['id184_train_config'] for row in probes)
assert all(row['id184_source_checkpoint'].endswith('/global_step_10') for row in probes)
assert all(row['wandb_run_id']=='nimloth-id184-k4-continue-to20-retry1' for row in probes)
assert all(row['wandb_resume']==os.environ['EXPECTED_WANDB_RESUME'] for row in probes)
assert all(row['wandb_api_key_present'] for row in probes)
assert all(row['dataset_type']=='vagen.gym_agent_dataset.AgenticDataset' for row in probes)
ray.shutdown()
PY

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" --gres=gpu:2 --cpus-per-task=16 \
  env "${COMMON_ENV[@]}" RAY_ADDRESS="${RAY_ADDRESS}" \
    RAY_EXPECTED_NODE_IPS="${RAY_EXPECTED_NODE_IPS}" VLLM_HOST_IP="${HEAD_IP}" \
    ID184_HEAD_IP="${HEAD_IP}" ID184_EXPECTED_NNODES=4 \
    ID184_EXPECTED_GPUS_PER_NODE=2 ID184_CLUSTER_NODES="${ID184_CLUSTER_NODES}" \
    REPO="${REPO}" EXPECTED_PARENT_COMMIT="${EXPECTED_PARENT_COMMIT}" \
    EXPECTED_VAGEN_COMMIT="${EXPECTED_VAGEN_COMMIT}" \
    EXPECTED_VERL_COMMIT="${EXPECTED_VERL_COMMIT}" \
    RUN_NAME="${RUN_NAME}" RUN_DATE="${RUN_DATE}" \
    RUN_OUTPUT_SUFFIX="${RUN_OUTPUT_SUFFIX}" \
    "${RUNNER}"
