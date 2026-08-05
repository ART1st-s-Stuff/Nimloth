#!/usr/bin/env bash
# Mechanics-only gate for the planner PPO ValueHead critic.
# Required exports: RUNTIME_REPO, EXPECTED_COMMIT, OUTPUT_DIR, SFT2_CHECKPOINT,
# TRAJECTORY_JSONL, FRESH_ROLLOUT_MANIFEST.
#SBATCH --job-name=ppo-value-gpu-gate
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x-%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/rl/slurm/%x-%j.err

set -euo pipefail

: "${RUNTIME_REPO:?RUNTIME_REPO is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${SFT2_CHECKPOINT:?SFT2_CHECKPOINT is required}"
: "${TRAJECTORY_JSONL:?TRAJECTORY_JSONL is required}"
: "${FRESH_ROLLOUT_MANIFEST:?FRESH_ROLLOUT_MANIFEST is required}"

PYTHON=/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3
CONFIG=${RUNTIME_REPO}/configs/training/rl/planner_greedy_h1_full.yaml
GATE=${RUNTIME_REPO}/experiments/training/rl/gpu_gate_ppo_value_critic.py
WM_CHECKPOINT=${SFT2_CHECKPOINT}/wm_predictor
STATE_PROJ_CHECKPOINT=${SFT2_CHECKPOINT}/state_proj.pt
VALUE_HEAD_CHECKPOINT=${SFT2_CHECKPOINT}/value_head
MASTER_PORT=$((21000 + SLURM_JOB_ID % 20000))

test "$(git -C "${RUNTIME_REPO}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${RUNTIME_REPO}" status --porcelain --untracked-files=no --ignore-submodules=dirty)"
test -x "${PYTHON}"
test -f "${CONFIG}"
test -f "${GATE}"
test -f "${SFT2_CHECKPOINT}/config.json"
test -f "${WM_CHECKPOINT}/predictor.pt"
test -f "${STATE_PROJ_CHECKPOINT}"
test -f "${VALUE_HEAD_CHECKPOINT}/value_head.pt"
test -s "${TRAJECTORY_JSONL}"
test -s "${FRESH_ROLLOUT_MANIFEST}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH=${RUNTIME_REPO}/src:${PYTHONPATH:-}
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled

{
  echo "start_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "runtime_repo=${RUNTIME_REPO}"
  echo "commit=${EXPECTED_COMMIT}"
  echo "checkpoint=${SFT2_CHECKPOINT}"
  echo "trajectory=${TRAJECTORY_JSONL}"
  echo "manifest=${FRESH_ROLLOUT_MANIFEST}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  "${PYTHON}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "${OUTPUT_DIR}/contract.log"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${GATE}" \
  --mode single_grad \
  --config "${CONFIG}" \
  --model "${SFT2_CHECKPOINT}" \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --state-proj-checkpoint "${STATE_PROJ_CHECKPOINT}" \
  --value-head-checkpoint "${VALUE_HEAD_CHECKPOINT}" \
  --trajectory-jsonl "${TRAJECTORY_JSONL}" \
  --fresh-rollout-manifest "${FRESH_ROLLOUT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus-per-rank 1 \
  2>&1 | tee "${OUTPUT_DIR}/single_grad.log"

NIMLOTH_DDP_GPU_STRIDE=2 CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON}" \
  -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  "${GATE}" \
  --mode ddp_step \
  --config "${CONFIG}" \
  --model "${SFT2_CHECKPOINT}" \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --state-proj-checkpoint "${STATE_PROJ_CHECKPOINT}" \
  --value-head-checkpoint "${VALUE_HEAD_CHECKPOINT}" \
  --trajectory-jsonl "${TRAJECTORY_JSONL}" \
  --fresh-rollout-manifest "${FRESH_ROLLOUT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus-per-rank 2 \
  2>&1 | tee "${OUTPUT_DIR}/ddp_step.log"

echo "end_utc=$(date -u +%FT%TZ)" | tee -a "${OUTPUT_DIR}/contract.log"
