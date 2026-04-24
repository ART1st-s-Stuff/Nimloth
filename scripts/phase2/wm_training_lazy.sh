#!/usr/bin/env bash
#
# WM DDP 训练 + Lazy Encoding 启动脚本（分块模式）
#
# 用法:
#   ./scripts/phase2/wm_training_lazy.sh wm=cfm_dinov2m
#   ./scripts/phase2/wm_training_lazy.sh wm=cfm_dinov2m --new
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/phase2/wm_training_lazy.sh
#
# 架构:
#   GPU 0  → encoder_server.py（分块编码，先编码 episode 0 再与训练并行）
#   GPU 1..N → torchrun DDP 训练（lazy 模式，等待 episode 就绪后读取分块 cache）
#
# 分块模式：
#   - encoder 先编码 episode 0，完成后写入 episode_0000.ready
#   - 训练进程等待 episode 0 就绪后开始
#   - encoder 继续编码剩余 episode，与训练并行
#
# 要求:
#   - torchrun（PyTorch >= 1.9）已在 PATH 中
#   - 至少 2 块 GPU（1 块 encoder + 至少 1 块训练）
#

set -euo pipefail

# ---- NCCL 环境健壮性检查 ---------------------------------------------------------
# 某些集群会预置 NCCL_TOPO_FILE，但文件并非 NCCL XML（例如 nvidia-smi topo 文本表格），
# 会在 dist.barrier 时触发 "XML Parse error" 并导致 DDP 初始化失败。
if [[ -n "${NCCL_TOPO_FILE:-}" ]]; then
  if [[ ! -f "${NCCL_TOPO_FILE}" ]]; then
    echo "[warn] NCCL_TOPO_FILE 不存在，已忽略: ${NCCL_TOPO_FILE}"
    unset NCCL_TOPO_FILE
  elif ! python - "${NCCL_TOPO_FILE}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="ignore")
stripped = text.lstrip()
sys.exit(0 if stripped.startswith("<") else 1)
PY
  then
    echo "[warn] NCCL_TOPO_FILE 不是 XML，已自动禁用以避免 NCCL 解析失败: ${NCCL_TOPO_FILE}"
    unset NCCL_TOPO_FILE
  fi
fi

# ---- 参数解析 ----------------------------------------------------------------
force_new_run="false"
hydra_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--new" ]]; then
    force_new_run="true"
  elif [[ "${arg}" == pipeline.train.rollout_steps=* ]]; then
    echo "[compat] 检测到废弃参数 ${arg}，自动映射为 pipeline.train.temporal_stride=${arg#*=}"
    hydra_args+=("pipeline.train.temporal_stride=${arg#*=}")
  else
    hydra_args+=("${arg}")
  fi
done

# ---- GPU 检测 ----------------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[info] CUDA_VISIBLE_DEVICES 未设置，检测所有可用 GPU..."
  num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l)
  if [[ "${num_gpus}" -eq 0 ]]; then
    echo "[error] 未检测到可用 GPU"
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus - 1)))
  echo "[info] 自动设置 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

IFS=',' read -ra GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
num_total_gpus=${#GPU_DEVICES[@]}

if [[ "${num_total_gpus}" -lt 2 ]]; then
  echo "[error] lazy encoding 模式至少需要 2 块 GPU（1 块 encoder + 1 块训练）"
  echo "        当前可用 GPU: ${num_total_gpus} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  exit 1
fi

num_train_gpus=$((num_total_gpus - 1))
encoder_gpu=${GPU_DEVICES[0]}
train_gpu_list=("${GPU_DEVICES[@]:1}")
train_gpus=$(IFS=,; echo "${train_gpu_list[*]}")

echo "=========================================="
echo "WM Training (DDP + Lazy Encoding - Chunk Mode)"
echo "=========================================="
echo "  总 GPU:        ${num_total_gpus}"
echo "  Encoder GPU:   ${encoder_gpu}  (encoder_server.py, chunk mode)"
echo "  Training GPUs: ${num_train_gpus}  (torchrun, local_rank 0..$((num_train_gpus - 1)))"
echo "  Training IDs:  ${train_gpus}"
echo "  Cache:         <分块模式，按 episode 组织>"
echo "=========================================="

# ---- Manifest 路径解析 --------------------------------------------------------
# 优先使用 WM_MANIFEST_PATH 环境变量指定的路径
manifest_path="${WM_MANIFEST_PATH:-}"

if [[ -z "${manifest_path}" ]]; then
  # 自动从 train split 目录解析（使用 datasets/ai2thor/train 而非 latest）
  train_dir="datasets/ai2thor/train"
  manifest_path="$(
    python - <<'PY'
import sys
from pathlib import Path
import json

train_dir = Path("datasets/ai2thor/train")
meta = train_dir / "metadata.json"
manifest_path = ""

if meta.exists():
    latest = json.loads(meta.read_text(encoding="utf-8")).get("latest")
    if latest:
        path = train_dir / latest / "manifest.jsonl"
        if path.exists():
            manifest_path = str(path)

if not manifest_path:
    runs = sorted([p for p in train_dir.iterdir() if p.is_dir() and (p / "manifest.jsonl").exists()], reverse=True)
    if runs:
        manifest_path = str(runs[0] / "manifest.jsonl")

print(manifest_path)
PY
  )"
fi

if [[ -z "${manifest_path}" ]]; then
  echo "[error] 未找到可用 manifest，请先执行 scripts/phase1/wm_data_collection.sh"
  exit 1
fi

echo "[info] 使用 manifest: ${manifest_path}"

# 打印 manifest 中的 episode 信息
episode_info="$(python3 - "${manifest_path}" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

manifest_path = sys.argv[1]
episode_counts = {}
total = 0
for line in Path(manifest_path).read_text().splitlines():
    if line.strip():
        sample = json.loads(line)
        ep = int(sample.get("episode_id", 0))
        episode_counts[ep] = episode_counts.get(ep, 0) + 1
        total += 1

print(f"总样本数: {total}, Episode 数: {len(episode_counts)}")
for ep_id in sorted(episode_counts.keys()):
    print(f"  Episode {ep_id}: {episode_counts[ep_id]} 张图")
PY
)"
echo "[info] Manifest 信息:"
echo "${episode_info}" | sed 's/^/[info] /'

# ---- 准备日志目录 --------------------------------------------------------------
log_root="${WM_OUTPUTS_ROOT:-outputs}/dev/$(date +%Y%m%d_%H%M%S)_lazy_wm"
mkdir -p "${log_root}"
encoder_log="${log_root}/encoder_server.log"
train_log="${log_root}/train.log"
echo "[info] 日志目录: ${log_root}"

# ---- 清理历史残留 encoder_server -----------------------------------------------
echo "[step 0/3] 清理历史残留 encoder_server（同 manifest）..."
python - "${manifest_path}" <<'PY'
import os
import signal
import subprocess
import sys
import time

manifest = sys.argv[1]
me = os.getpid()
out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
targets = []
for raw in out.splitlines():
    line = raw.strip()
    if not line:
        continue
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        continue
    pid_s, cmd = parts
    try:
        pid = int(pid_s)
    except ValueError:
        continue
    if pid == me:
        continue
    if "python -m src.train.encoder_server" not in cmd:
        continue
    if f"dataset.manifests.train={manifest}" not in cmd:
        continue
    targets.append(pid)

if targets:
    print(f"[warn] 发现残留 encoder_server: {targets}")
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 8.0
    alive = set(targets)
    while alive and time.time() < deadline:
        time.sleep(0.2)
        for pid in list(alive):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive.remove(pid)
    for pid in list(alive):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"[info] 已清理残留 encoder_server: {targets}")
else:
    print("[info] 未发现同 manifest 的残留 encoder_server")
PY

# ---- 启动 encoder_server.py（GPU 0，后台）--------------------------------------
echo "[step 1/3] 启动 encoder_server.py on GPU ${encoder_gpu}（分块模式）..."
encoder_pid=""
monitor_pid=""
tail_pid=""
cleanup() {
  if [[ -n "${tail_pid}" ]]; then
    kill -TERM "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
  fi
  if [[ -n "${monitor_pid}" ]]; then
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -n "${encoder_pid}" ]]; then
    echo "[cleanup] 停止 encoder_server (PID=${encoder_pid})..."
    kill -TERM "${encoder_pid}" 2>/dev/null || true
    wait "${encoder_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# 构建 cache_dir 路径
cache_stem=$(basename "${manifest_path}" .jsonl)
dataset_parent=$(dirname "${manifest_path}")
# 从 hydra_args 中提取 wm 名称，默认使用配置文件中的值
# 支持 wm=cfm_dinov2m 或 pipeline.train.wm_name=xxx 格式
wm_name=""
for arg in "${hydra_args[@]}"; do
  if [[ "${arg}" == wm=* ]]; then
    wm_name="${arg#wm=}"
    break
  elif [[ "${arg}" == pipeline.train.wm_name=* ]]; then
    wm_name="${arg#pipeline.train.wm_name=}"
    break
  fi
done
# 如果没有从参数传入，使用默认值
if [[ -z "${wm_name}" ]]; then
  wm_name="cfm_dinov2m"  # 默认值
fi
cache_dir="${dataset_parent}/${cache_stem}.latents.${wm_name}"
first_episode_ready="${cache_dir}/episode_0000.ready"

# 清理可能存在的旧的 ready marker（如果从中断恢复，需要确认是否需要重新编码）
# 这里不清理，让 encoder_server 检测已完成 episode

CUDA_VISIBLE_DEVICES="${encoder_gpu}" \
  python -m src.train.encoder_server \
  dataset.manifests.train="${manifest_path}" \
  pipeline.train.encoder_batch_size=64 \
  pipeline.train.lazy_episode_chunk=true \
  pipeline.train.lazy_wait_first_episode=true \
  "${hydra_args[@]}" \
  > "${encoder_log}" 2>&1 &
encoder_pid=$!
echo "[info] encoder_server.py 启动 (PID=${encoder_pid})，日志: ${encoder_log}"

# 等待 encoder_server 初始化（约 10s，用于加载 DINOv2 模型）
echo "[info] 等待 encoder_server 初始化（约 10s）..."
sleep 10

# 检查 encoder_server 是否仍在运行
if ! kill -0 "${encoder_pid}" 2>/dev/null; then
  echo "[error] encoder_server.py 启动失败，查看日志: ${encoder_log}"
  cat "${encoder_log}"
  exit 1
fi
echo "[info] encoder_server.py 已就绪，开始等待 episode 0 编码完成..."

# 等待 episode 0 就绪（encoder 完成第一阶段预编码）
wait_timeout=300  # 最多等待 5 分钟
wait_interval=2
waited=0
while [[ ! -f "${first_episode_ready}" ]]; do
  sleep ${wait_interval}
  waited=$((waited + wait_interval))
  # 检查 encoder 是否还在运行
  if ! kill -0 "${encoder_pid}" 2>/dev/null; then
    echo "[error] encoder_server.py 在等待期间退出，查看日志: ${encoder_log}"
    cat "${encoder_log}"
    exit 1
  fi
  if [[ ${waited} -ge ${wait_timeout} ]]; then
    echo "[error] 等待 episode 0 就绪超时 (${wait_timeout}s)，查看 encoder 日志: ${encoder_log}"
    tail -30 "${encoder_log}"
    exit 1
  fi
  echo "  等待 episode 0 就绪... (${waited}s)"
done

echo "[step 1/3] Episode 0 就绪，cache_dir=${cache_dir}"

# ---- 启动统一监控仪表板（后台） --------------------------------------------
echo "[info] 启动统一训练仪表板..."
monitor_log="${log_root}/monitor.log"
python -m src.utils.training_monitor \
  --cache-dir "${cache_dir}" \
  --manifest "${manifest_path}" \
  > "${monitor_log}" 2>&1 &
monitor_pid=$!
echo "[info] 监控仪表板已启动 (PID=${monitor_pid})，日志: ${monitor_log}"

# ---- 启动 DDP 训练（GPU 1..N）------------------------------------------------
echo "[step 2/3] 启动 DDP 训练 on GPUs ${train_gpus}..."
echo "[info] 训练日志: ${train_log}"
echo "[info] 终端将实时显示训练日志（同时写入文件）"

# 先创建日志文件，再后台 tail，实现终端实时观察。
: > "${train_log}"
tail -f "${train_log}" &
tail_pid=$!

# torchrun 参数：
#   --nnodes=1            单机
#   --nproc_per_node=N   N 个训练进程（= num_train_gpus）
#   --master_addr/port    rank 0 的地址（默认本机）
# 注意：训练进程只看到训练卡（不含 encoder 卡），local_rank 与可见卡一一对应。
CUDA_VISIBLE_DEVICES="${train_gpus}" PYTHONUNBUFFERED=1 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${num_train_gpus}" \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  src/train/train_wm_ddp.py \
  dataset.manifests.train="${manifest_path}" \
  pipeline.train.operation.force_new_run="${force_new_run}" \
  pipeline.train.lazy_encoding=true \
  pipeline.train.lazy_episode_chunk=true \
  "${hydra_args[@]}" \
  > "${train_log}" 2>&1

train_exit_code=$?

# 主流程结束后主动执行一次清理（EXIT trap 仍会兜底）。
cleanup
trap - EXIT INT TERM

if [[ "${train_exit_code}" -ne 0 ]]; then
  echo "[error] 训练失败 (exit code=${train_exit_code})，查看日志: ${train_log}"
  tail -50 "${train_log}"
  exit "${train_exit_code}"
fi

echo "=========================================="
echo "训练完成"
echo "  训练日志: ${train_log}"
echo "  encoder 日志: ${encoder_log}"
echo "  监控日志: ${monitor_log}"
echo "=========================================="
