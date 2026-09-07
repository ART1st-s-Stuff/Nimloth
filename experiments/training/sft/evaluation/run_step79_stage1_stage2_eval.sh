#!/usr/bin/env bash
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=600G
#SBATCH --time=06:00:00
#SBATCH --job-name=sft12-step79-eval

set -euo pipefail

: "${REPO:?set REPO to the clean committed unified-SFT worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact Nimloth commit}"
: "${EXPECTED_VAGEN_COMMIT:?set EXPECTED_VAGEN_COMMIT}"
: "${EXPECTED_VERL_COMMIT:?set EXPECTED_VERL_COMMIT}"
: "${EXPECTED_LEWM_COMMIT:?set EXPECTED_LEWM_COMMIT}"
: "${SLURM_JOB_ID:?submit this script through Slurm}"

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
ROOT=/project/peilab/atst/nimloth
SOURCE_CHECKPOINT=${ROOT}/experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2/checkpoints/global_step_79/actor/huggingface
DATA_ROOT=${ROOT}/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/converted_strict_k8_b6c811c
TRAIN_JSONL=${DATA_ROOT}/train_success.jsonl
VAL_JSONL=${DATA_ROOT}/val_all.jsonl
DINO_CACHE=${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-20/sft2/cache/k16_all3217_px100352_bf16_dino4x4_f32_b8659fe
ASSET_ROOT=${REPO}/external/VAGEN/vagen/envs/navigation/assets
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${RUN_ROOT:-${ROOT}/outputs/experiments/training/sft/evaluation/${RUN_STAMP}_job${SLURM_JOB_ID}_${EXPECTED_COMMIT:0:8}}
CONTRACT_MODULE=experiments.training.sft.evaluation.pipeline_contract
ARM_SCRIPT=${REPO}/experiments/training/sft/evaluation/environment_arm.sh

[[ -x "${PYTHON}" ]] || { echo "missing fixed interpreter: ${PYTHON}" >&2; exit 2; }
[[ -f "${ARM_SCRIPT}" ]] || { echo "missing environment arm: ${ARM_SCRIPT}" >&2; exit 2; }
[[ "${SLURM_JOB_NUM_NODES:-0}" == 1 ]] || { echo "pipeline requires one node" >&2; exit 2; }
for pair in \
  "${REPO}:${EXPECTED_COMMIT}" \
  "${REPO}/external/VAGEN:${EXPECTED_VAGEN_COMMIT}" \
  "${REPO}/external/VAGEN/verl:${EXPECTED_VERL_COMMIT}" \
  "${REPO}/external/le-wm:${EXPECTED_LEWM_COMMIT}"; do
  source_repo=${pair%%:*}
  expected=${pair#*:}
  [[ "$(git -C "${source_repo}" rev-parse HEAD)" == "${expected}" ]] || {
    echo "commit mismatch: ${source_repo}" >&2; exit 2;
  }
  [[ -z "$(git -C "${source_repo}" status --porcelain --untracked-files=all)" ]] || {
    echo "source is dirty: ${source_repo}" >&2; exit 2;
  }
done
for input in "${SOURCE_CHECKPOINT}/config.json" "${TRAIN_JSONL}" "${VAL_JSONL}" \
  "${DINO_CACHE}/train/dino_grid4/manifest.json" \
  "${DINO_CACHE}/val/dino_grid4/manifest.json" \
  "${ASSET_ROOT}/base.json" "${ASSET_ROOT}/common_sense.json"; do
  [[ -s "${input}" ]] || { echo "missing required input: ${input}" >&2; exit 2; }
done
[[ ! -e "${RUN_ROOT}" ]] || { echo "fresh output already exists: ${RUN_ROOT}" >&2; exit 2; }

ALLOCATED_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
IFS=',' read -r -a GPU_TOKENS <<<"${ALLOCATED_CUDA_VISIBLE_DEVICES}"
(( ${#GPU_TOKENS[@]} == 8 )) || {
  echo "pipeline requires exactly eight allocated CUDA tokens; got ${#GPU_TOKENS[@]}" >&2
  exit 2
}
PORT_BASE=$((22000 + SLURM_JOB_ID % 8000))
TRAIN_PORT=${PORT_BASE}
ENV1_PORT=$((PORT_BASE + 1))
ENV2_PORT=$((PORT_BASE + 2))
for port in "${TRAIN_PORT}" "${ENV1_PORT}" "${ENV2_PORT}"; do
  if ss -ltnH "sport = :${port}" | grep -q .; then
    echo "required port is already occupied: ${port}" >&2; exit 2
  fi
done

mkdir -p "${RUN_ROOT}"
export PYTHONPATH=${REPO}/src:${REPO}:${REPO}/external/VAGEN:${REPO}/external/VAGEN/verl:${REPO}/external/le-wm
export PATH=/project/peilab/atst/nimloth/.venv-vagen-main/bin:${PATH}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}
[[ -d "${HF_HOME}" && -r "${HF_HOME}" ]] || { echo "verified HF cache is unavailable" >&2; exit 2; }
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NIMLOTH_PIPELINE_RUNTIME_ROOT=/tmp/nimloth-sft12-${SLURM_JOB_ID}-${EXPECTED_COMMIT:0:8}
mkdir -p "${NIMLOTH_PIPELINE_RUNTIME_ROOT}"

ENV1_PID=
ENV2_PID=
EVAL1_PID=
EVAL2_PID=
terminate_group() {
  local pid=${1:-}
  [[ -n "${pid}" ]] || return 0
  kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    kill -0 -- "-${pid}" >/dev/null 2>&1 || return 0
    sleep 1
  done
  kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
}
runtime_process_ids() {
  "${PYTHON}" - "${NIMLOTH_PIPELINE_RUNTIME_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

marker = sys.argv[1].encode()
ancestors = set()
pid = os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try:
        pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        break
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    try:
        environment = (entry / "environ").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if marker in environment:
        print(entry.name)
PY
}
terminate_runtime_processes() {
  local signal=$1 pid
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    kill -"${signal}" "${pid}" >/dev/null 2>&1 || true
  done < <(runtime_process_ids)
}
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  set +e
  terminate_group "${EVAL1_PID}"
  terminate_group "${EVAL2_PID}"
  terminate_group "${ENV1_PID}"
  terminate_group "${ENV2_PID}"
  terminate_runtime_processes TERM
  sleep 3
  terminate_runtime_processes KILL
  sleep 1
  runtime_process_ids >"${RUN_ROOT}/owned_processes_after.log" 2>/dev/null || true
  if [[ -s "${RUN_ROOT}/owned_processes_after.log" ]]; then status=91; fi
  for port in "${ENV1_PORT}" "${ENV2_PORT}"; do
    if ss -ltnH "sport = :${port}" | grep -q .; then status=92; fi
  done
  "${PYTHON}" -m "${CONTRACT_MODULE}" record-exit \
    --exit-code "${status}" --output "${RUN_ROOT}/final_status.json" || true
  rm -rf -- "${NIMLOTH_PIPELINE_RUNTIME_ROOT}"
  exit "${status}"
}
handle_termination() {
  trap - TERM INT
  exit 143
}
trap cleanup EXIT
trap handle_termination TERM INT

exec > >(tee -a "${RUN_ROOT}/controller.log") 2>&1
cat >"${RUN_ROOT}/README.md" <<EOF
# 统一 SFT1/SFT2 一轮训练与 direct 评估

- 用途：验证近期统一 SFT 代码；本结果不自动成为后续训练初始化。
- 代码：Nimloth ${EXPECTED_COMMIT}，VAGEN ${EXPECTED_VAGEN_COMMIT}，VERL ${EXPECTED_VERL_COMMIT}，LeWM ${EXPECTED_LEWM_COMMIT}。
- 初始化：历史 VAGEN step79 HF checkpoint：${SOURCE_CHECKPOINT}。
- 数据：train_success.jsonl（613 条轨迹、7309 个回答前缀）与 val_all.jsonl（355 条轨迹、6054 个回答前缀）。
- 离线验证边界：val_all 与 train_success 有1个任务重叠，因此只用于训练过程诊断，不称为独立 held-out；正式 Base/Common Sense 120 与训练任务及场景均无重叠。
- SFT1：format，K1 generate，LoRA r64/alpha128，world8，batch1，GA8，一轮。
- SFT2：从 SFT1 epoch_001/hf_merged 初始化；query，K16 inject，CE+DINO，其他训练规模相同，一轮。
- 计算单元：SFT1遍历613条完整轨迹（约10个optimizer step）；SFT2按回答前缀建立索引，完整遍历7309个回答（约115个optimizer step），每个样本保留该回答之前的全部真实历史，不截断或抽样回答。
- 可训练参数：基础权重冻结；LoRA后缀同时命中语言层与视觉块MLP，历史模型探针为698个可训练tensor、770,940,928个参数；embedding和lm_head完整训练，SFT2另训练共享slot projector，因此不将视觉分支描述为完全冻结。
- 评估：两阶段合并模型并发 direct rollout；Base/Common Sense 各 seeds 1..60，greedy，最多20步，512 tokens，TP1。
- 资源：normal 单节点八卡，训练顺序执行；评估各占一张环境卡和一张策略卡。
- W&B：禁用。
EOF

"${PYTHON}" -m "${CONTRACT_MODULE}" preflight-inputs \
  --source-checkpoint "${SOURCE_CHECKPOINT}" --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" --dino-cache-root "${DINO_CACHE}" \
  --asset-root "${ASSET_ROOT}" --output-root "${ROOT}" \
  --output "${RUN_ROOT}/input_preflight.json"

# controller.log keeps the fully expanded commands that actually ran.
export PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=8 \
  --master_port="${TRAIN_PORT}" -m "${CONTRACT_MODULE}" rank-map \
  --output "${RUN_ROOT}/rank_map.json"

STAGE1_OUT=${RUN_ROOT}/stage1
STAGE1_EPOCH=${STAGE1_OUT}/epoch_001
STAGE1_MERGED=${STAGE1_EPOCH}/hf_merged
STAGE2_OUT=${RUN_ROOT}/stage2
STAGE2_EPOCH=${STAGE2_OUT}/epoch_001
STAGE2_MERGED=${STAGE2_EPOCH}/hf_merged

CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=8 --master_port="${TRAIN_PORT}" \
  -m nimloth.training.sft.stage1 \
  --model "${SOURCE_CHECKPOINT}" --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output-dir "${STAGE1_OUT}" --epochs 1 --batch-size 1 --grad-accum 8 \
  --lr 1e-6 --embedding-lr 5e-6 --max-length 12000 --max-pixels 100352 \
  --latent-token-count 1 --latent-query-mode generate \
  --lora --lora-r 64 --lora-alpha 128 --no-cache --no-wandb \
  2>&1 | tee "${RUN_ROOT}/stage1_train.log"
"${PYTHON}" -m nimloth.training.sft.stage1.checkpoint_export \
  --base-model "${SOURCE_CHECKPOINT}" --adapter-dir "${STAGE1_EPOCH}" \
  --out-dir "${STAGE1_MERGED}" 2>&1 | tee "${RUN_ROOT}/stage1_merge.log"
"${PYTHON}" -m "${CONTRACT_MODULE}" validate-merged --stage format \
  --adapter-dir "${STAGE1_EPOCH}" --merged-dir "${STAGE1_MERGED}" \
  --output "${RUN_ROOT}/stage1_merge.json"

CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=8 --master_port="${TRAIN_PORT}" \
  -m nimloth.training.sft.stage2 \
  --model "${STAGE1_MERGED}" --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output-dir "${STAGE2_OUT}" --dino-cache-root "${DINO_CACHE}" \
  --epochs 1 --batch-size 1 --grad-accum 8 --max-length 12000 --max-pixels 100352 \
  --lr 1e-6 --embedding-lr 5e-6 --latent-token-count 16 --latent-query-mode inject \
  --grid-size 4 --lora --lora-r 64 --lora-alpha 128 --no-wandb \
  2>&1 | tee "${RUN_ROOT}/stage2_train.log"
"${PYTHON}" -m nimloth.training.sft.stage1.checkpoint_export \
  --base-model "${STAGE1_MERGED}" --adapter-dir "${STAGE2_EPOCH}" \
  --out-dir "${STAGE2_MERGED}" 2>&1 | tee "${RUN_ROOT}/stage2_merge.log"
"${PYTHON}" -m "${CONTRACT_MODULE}" validate-merged --stage query \
  --adapter-dir "${STAGE2_EPOCH}" --merged-dir "${STAGE2_MERGED}" \
  --output "${RUN_ROOT}/stage2_merge.json"

start_environment() {
  local arm=$1 gpu=$2 port=$3
  local arm_root=${NIMLOTH_PIPELINE_RUNTIME_ROOT}/${arm}
  local arm_output=${RUN_ROOT}/${arm}_environment
  mkdir -p "${arm_root}"
  setsid env REPO="${REPO}" PYTHON="${PYTHON}" ARM_ROOT="${arm_root}" \
    ARM_OUTPUT="${arm_output}" ENV_PORT="${port}" CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PYTHONPATH}" \
    NIMLOTH_PIPELINE_RUNTIME_ROOT="${NIMLOTH_PIPELINE_RUNTIME_ROOT}" \
    bash "${ARM_SCRIPT}" >"${RUN_ROOT}/${arm}_environment.log" 2>&1 &
  LAST_CHILD_PID=$!
}
LAST_CHILD_PID=
start_environment stage1 "${GPU_TOKENS[0]}" "${ENV1_PORT}"
ENV1_PID=${LAST_CHILD_PID}
start_environment stage2 "${GPU_TOKENS[2]}" "${ENV2_PORT}"
ENV2_PID=${LAST_CHILD_PID}
for spec in "${ENV1_PID}:${ENV1_PORT}" "${ENV2_PID}:${ENV2_PORT}"; do
  pid=${spec%%:*}; port=${spec#*:}
  for _ in $(seq 1 120); do
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
    kill -0 "${pid}" || { echo "environment process exited: ${pid}" >&2; exit 4; }
    sleep 2
  done
  curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null
done

for spec in "stage1:${ENV1_PORT}" "stage2:${ENV2_PORT}"; do
  arm=${spec%%:*}; port=${spec#*:}
  for dataset in base common_sense; do
    "${PYTHON}" -m nimloth.environment.navigation.prewarm \
      --env-url "http://127.0.0.1:${port}" --eval-set "${dataset}" --seed 1 \
      --timeout-seconds 300 --env-id "sft12-${arm}-${dataset}-${SLURM_JOB_ID}" \
      >"${RUN_ROOT}/${arm}_prewarm_${dataset}.json"
  done
done

run_eval() {
  local arm=$1 gpu=$2 port=$3 checkpoint=$4
  local out=${RUN_ROOT}/${arm}_direct_eval
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" NIMLOTH_PIPELINE_RUNTIME_ROOT="${NIMLOTH_PIPELINE_RUNTIME_ROOT}" \
    "${PYTHON}" -m nimloth.training.sft.evaluation \
    --mode direct --checkpoint "${checkpoint}" --env-url "http://127.0.0.1:${port}" \
    --output-dir "${out}" --eval-sets base common_sense --split test \
    --episodes-per-eval-set 60 --seed-offset 1 --max-steps 20 \
    --temperature 0 --top-p 1 --max-response-tokens 512 --tensor-parallel-size 1 \
    --max-pixels 100352 >"${RUN_ROOT}/${arm}_eval.log" 2>&1 &
  LAST_CHILD_PID=$!
}
run_eval stage1 "${GPU_TOKENS[1]}" "${ENV1_PORT}" "${STAGE1_MERGED}"
EVAL1_PID=${LAST_CHILD_PID}
run_eval stage2 "${GPU_TOKENS[3]}" "${ENV2_PORT}" "${STAGE2_MERGED}"
EVAL2_PID=${LAST_CHILD_PID}
wait "${EVAL1_PID}"; EVAL1_PID=
wait "${EVAL2_PID}"; EVAL2_PID=

"${PYTHON}" -m "${CONTRACT_MODULE}" finalize \
  --stage1-eval "${RUN_ROOT}/stage1_direct_eval" \
  --stage2-eval "${RUN_ROOT}/stage2_direct_eval" \
  --stage1-merge "${RUN_ROOT}/stage1_merge.json" \
  --stage2-merge "${RUN_ROOT}/stage2_merge.json" \
  --output "${RUN_ROOT}/final_status.json"
