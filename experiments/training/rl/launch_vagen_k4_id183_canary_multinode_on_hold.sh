#!/usr/bin/env bash
# Launch one ID183 phase on an exact two-node, four-H800-per-node allocation.
set -euo pipefail

HOLD_JOB=${1:?usage: launch_vagen_k4_id183_canary_multinode_on_hold.sh HOLD_JOB PHASE}
PHASE=${2:?phase must be train_to_5 or resume_to_10}
[[ "${PHASE}" == train_to_5 || "${PHASE}" == resume_to_10 ]]
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
RUN_NAME=183_canary_k4schemeb_jointupdate_dp8_tp8_u10_r5_train3x8_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8
RUN_DATE=2026-08-16
RUN_OUT=${ROOT}/outputs/experiments/training/rl/${RUN_DATE}/${RUN_NAME}
RUNNER=${REPO}/experiments/training/rl/run_vagen_k4_id183_canary_phase.sh
PHASE_TAG=$([[ "${PHASE}" == train_to_5 ]] && echo p1 || echo p2)
PHASE_NAME=$([[ "${PHASE}" == train_to_5 ]] && echo phase1_train_to_5 || echo phase2_fresh_resume_to_10)
PHASE_OUT=${RUN_OUT}/${PHASE_NAME}
RUNTIME_ROOT=/tmp/i183-${HOLD_JOB}-${PHASE_TAG}
RAY_PORT=$((22000 + HOLD_JOB % 10000))
RAY_CLUSTER_ROOT=/tmp/i183-ray-${HOLD_JOB}-${PHASE_TAG}
RAY_LOG_ROOT=${ROOT}/outputs/experiments/training/rl/slurm/id183-ray-${HOLD_JOB}-${PHASE_TAG}
RAY_PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl
mkdir -p "${ROOT}/outputs/experiments/training/rl/slurm"
exec 9>"${ROOT}/outputs/experiments/training/rl/slurm/.id183-${HOLD_JOB}-${PHASE_TAG}.launch.lock"
flock -n 9 || { echo "duplicate ID183 launcher for job ${HOLD_JOB} phase ${PHASE}" >&2; exit 94; }
ACTOR_MODEL=${ROOT}/outputs/experiments/training/sft2/2026-08-15/176_id74_action_head_repair_balanced271x8_val40x8/checkpoint
PLANNING_MODEL=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001
source "${REPO}/experiments/training/rl/slurm_allocation.sh"
set -a
source /project/peilab/atst/flower/.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
WANDB_RESUME=$([[ "${PHASE}" == train_to_5 ]] && echo never || echo must)

[[ -x "${PY}" && -x "${RUNNER}" ]]
JOB_DETAILS=$(scontrol show job -dd "${HOLD_JOB}")
grep -q 'JobState=RUNNING' <<<"${JOB_DETAILS}"
grep -q 'Partition=normal' <<<"${JOB_DETAILS}"
grep -q 'NumNodes=2' <<<"${JOB_DETAILS}"
grep -q 'TimeLimit=05:00:00' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'ReqTRES=[^ ]*mem=256G([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*gres/gpu=8([, ]|$)' <<<"${JOB_DETAILS}"
grep -Eq 'AllocTRES=[^ ]*cpu=64([, ]|$)' <<<"${JOB_DETAILS}"
grep -q 'MinMemoryNode=128G' <<<"${JOB_DETAILS}"

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "${HOLD_JOB}" -o '%N')")
[[ ${#NODES[@]} -eq 2 ]]
HEAD_NODE=${NODES[0]}
WORKER_NODE=${NODES[1]}
[[ "${HEAD_NODE}" != "${WORKER_NODE}" ]]
for node in "${NODES[@]}"; do
  for excluded in dgx-13 dgx-23 dgx-32 dgx-37 dgx-51; do
    [[ "${node}" != "${excluded}" ]]
  done
done

declare -A GPU_COUNTS
nimloth_load_slurm_gpu_counts "${JOB_DETAILS}" GPU_COUNTS
for node in "${NODES[@]}"; do
  [[ "${GPU_COUNTS[${node}]:-}" == 4 ]]
done

mapfile -t FABRIC_ROWS < <(
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=2 --ntasks=2 \
    --ntasks-per-node=1 --gpus=0 bash -lc '
      row=$(ip -o -4 addr show | awk '\''$4 ~ /^10\.23\./ {split($4,a,"/"); print a[1],$2; exit}'\'')
      [[ -n "${row}" ]]
      printf "%s %s\n" "$(hostname -s)" "${row}"
    '
)
[[ ${#FABRIC_ROWS[@]} -eq 2 ]]
declare -A NODE_IP NODE_IFACE
for row in "${FABRIC_ROWS[@]}"; do
  read -r node ip iface <<<"${row}"
  [[ -z "${NODE_IP[${node}]:-}" ]]
  NODE_IP[${node}]=${ip}
  NODE_IFACE[${node}]=${iface}
done
[[ -n "${NODE_IP[${HEAD_NODE}]:-}" && -n "${NODE_IP[${WORKER_NODE}]:-}" ]]
[[ "${NODE_IFACE[${HEAD_NODE}]}" == "${NODE_IFACE[${WORKER_NODE}]}" ]]
HEAD_IP=${NODE_IP[${HEAD_NODE}]}
WORKER_IP=${NODE_IP[${WORKER_NODE}]}
FABRIC_IFACE=${NODE_IFACE[${HEAD_NODE}]}
RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}
RAY_EXPECTED_NODE_IPS=${HEAD_IP},${WORKER_IP}

timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=2 --ntasks=2 \
  --ntasks-per-node=1 --gres=gpu:4 bash -lc '
    set -euo pipefail
    mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ ${#names[@]} -eq 4 ]]
    for name in "${names[@]}"; do [[ "${name}" == *H800* ]]; done
    printf "%s gpu_count=%s names=%s\n" "$(hostname -s)" "${#names[@]}" "${names[*]}"
  '

timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=2 --ntasks=2 \
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
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=2 --ntasks=2 \
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
    try: evidence=(entry/'environ').read_bytes()+(entry/'cmdline').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if any(root in evidence for root in roots):
        try: os.kill(int(entry.name),sig)
        except (ProcessLookupError,PermissionError): pass
PY
}

audit_node_runtime_empty() {
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=2 --ntasks=2 \
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
    try: evidence=(entry/'environ').read_bytes()+(entry/'cmdline').read_bytes()
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    if any(root in evidence for root in roots): found.append(int(entry.name))
print(f'{socket.gethostname()} owned_pids={found}')
if found: raise SystemExit(1)
PY
}

cleanup_cluster() {
  local status=$?
  trap - EXIT
  set +e
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
  timeout 30s srun --jobid="${HOLD_JOB}" --overlap --nodes=2 --ntasks=2 \
    --ntasks-per-node=1 --gpus=0 \
    env RAY_CLUSTER_ROOT="${RAY_CLUSTER_ROOT}" RUNTIME_ROOT="${RUNTIME_ROOT}" \
    bash -lc '
      [[ "${RAY_CLUSTER_ROOT}" == /tmp/i183-ray-* ]]
      [[ "${RUNTIME_ROOT}" == /tmp/i183-* ]]
      rm -rf -- "${RAY_CLUSTER_ROOT}" "${RUNTIME_ROOT}"
    ' \
    >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup_cluster EXIT

# These long-lived raylet and driver steps intentionally share the job's CPU,
# memory and GRES allocation. Slurm --overlap is required on every such step;
# Ray's own resource accounting keeps the eight worker actors within 4+4 GPUs.
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
  WANDB_RUN_ID=nimloth-id183-k4-10update-canary
  WANDB_RESUME="${WANDB_RESUME}"
  WANDB_DIR="${RAY_LOG_ROOT}/wandb"
  ID183_TRAIN_CONFIG="${PHASE_OUT}/train_navigation_joint_id183.yaml"
  ID183_VAL_CONFIG="${PHASE_OUT}/val_navigation_joint_id183.yaml"
  ID183_ACTOR_MODEL="${ACTOR_MODEL}"
  ID183_PLANNING_CHECKPOINT="${PLANNING_MODEL}"
  ID183_AGENT_CONFIG="${REPO}/external/VAGEN/vagen/configs/agent_no_concat.yaml"
  ID183_RUN_NAME="${RUN_NAME}"
  ID183_RUN_OUT="${RUN_OUT}"
  RAY_agent_register_timeout_ms=120000
)

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" --gres=gpu:4 --cpus-per-task=32 \
  env "${COMMON_ENV[@]}" VLLM_HOST_IP="${HEAD_IP}" \
  "${PY}" -m ray.scripts.scripts start --block --head \
    --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
    --temp-dir="${RAY_CLUSTER_ROOT}" --num-cpus=32 --num-gpus=4 \
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

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${WORKER_NODE}" --gres=gpu:4 --cpus-per-task=32 \
  env "${COMMON_ENV[@]}" VLLM_HOST_IP="${WORKER_IP}" \
  "${PY}" -m ray.scripts.scripts start --block \
    --address="${RAY_ADDRESS}" --node-ip-address="${WORKER_IP}" \
    --temp-dir="${RAY_CLUSTER_ROOT}" --num-cpus=32 --num-gpus=4 \
    --object-store-memory=10000000000 --disable-usage-stats \
  >"${RAY_LOG_ROOT}/${WORKER_NODE}.log" 2>&1 &
RAY_STEP_PIDS+=("$!")

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
if counts != [4.0, 4.0] or addresses != sorted(os.environ['RAY_EXPECTED_NODE_IPS'].split(',')):
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
    address=ray.util.get_node_ip_address()
    return {
        'address':address,
        'vllm_host_ip':os.environ.get('VLLM_HOST_IP'),
        'nccl_socket_ifname':os.environ.get('NCCL_SOCKET_IFNAME'),
        'hf_home':os.environ.get('HF_HOME'),
        'torch_home':os.environ.get('TORCH_HOME'),
        'vllm_worker_multiproc_method':os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'),
        'id183_train_config':os.environ.get('ID183_TRAIN_CONFIG'),
        'wandb_run_id':os.environ.get('WANDB_RUN_ID'),
        'wandb_resume':os.environ.get('WANDB_RESUME'),
        'wandb_api_key_present':bool(os.environ.get('WANDB_API_KEY')),
        'nimloth':str(nimloth.__file__),
        'vagen':str(vagen.__file__),
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
assert [row['gpus'] for row in rows] == [4.0,4.0]
assert [row['address'] for row in rows] == sorted(os.environ['RAY_EXPECTED_NODE_IPS'].split(','))
assert all(row['address']==row['vllm_host_ip'] for row in probes)
assert all(row['nccl_socket_ifname'] for row in probes)
assert all(row['hf_home']=='/project/peilab/atst/.cache/huggingface' for row in probes)
assert all(row['torch_home']=='/project/peilab/atst/flower/.cache/torch' for row in probes)
assert all(row['vllm_worker_multiproc_method']=='spawn' for row in probes)
assert all(row['id183_train_config'] for row in probes)
assert all(row['wandb_run_id']=='nimloth-id183-k4-10update-canary' for row in probes)
assert all(row['wandb_resume']==os.environ['EXPECTED_WANDB_RESUME'] for row in probes)
assert all(row['wandb_api_key_present'] for row in probes)
ray.shutdown()
PY

srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 \
  -w "${HEAD_NODE}" --gres=gpu:4 --cpus-per-task=32 \
  env "${COMMON_ENV[@]}" RAY_ADDRESS="${RAY_ADDRESS}" \
    RAY_EXPECTED_NODE_IPS="${RAY_EXPECTED_NODE_IPS}" VLLM_HOST_IP="${HEAD_IP}" \
    ID183_HEAD_IP="${HEAD_IP}" ID183_EXPECTED_NNODES=2 \
    ID183_EXPECTED_GPUS_PER_NODE=4 ID183_CLUSTER_NODES="${HEAD_NODE},${WORKER_NODE}" \
    REPO="${REPO}" EXPECTED_PARENT_COMMIT="${EXPECTED_PARENT_COMMIT}" \
    EXPECTED_VAGEN_COMMIT="${EXPECTED_VAGEN_COMMIT}" \
    EXPECTED_VERL_COMMIT="${EXPECTED_VERL_COMMIT}" \
    RUN_NAME="${RUN_NAME}" RUN_DATE="${RUN_DATE}" PHASE="${PHASE}" \
    "${RUNNER}"
