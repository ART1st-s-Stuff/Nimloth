#!/bin/bash
#SBATCH --job-name=sft1-train-v79
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --requeue

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
SL=/cm/shared/apps/slurm/current/bin
SCRIPTDIR=${ROOT}/experiments/navigation_baseline
BASE_RUN_NAME=vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2
BASE_RUN_DIR=${SCRIPTDIR}/runs/${BASE_RUN_NAME}
ROLLOUT_STEP=0
INIT_HF=${BASE_RUN_DIR}/checkpoints/global_step_79/actor/huggingface
ROLLOUT_ROOT=${SCRIPTDIR}/runs/sft1_rollouts_vagen79_greedy_parallel/validation
RECORDS_ROOT=${SCRIPTDIR}/runs/sft1_sft_records_vagen79_nimloth_format
TRAIN_OUT=${SCRIPTDIR}/runs/sft1_train_vagen79_qwen25vl
LOG=${TRAIN_OUT}/sft1_train_vagen79.log
NGPUS_PER_NODE=4
NNODES=2
RDZV_PORT=29500

mkdir -p "${TRAIN_OUT}"

COMMON_ENV='export UV_CACHE_DIR=/project/peilab/atst/nimloth/.cache/uv; export HOME=/project/peilab/atst/nimloth/.home; export PATH=/project/peilab/atst/nimloth/.venv/bin:/project/peilab/atst/nimloth/.local/bin:$PATH; export HF_HOME=/project/peilab/atst/.cache/huggingface; export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface; export TORCH_HOME=/project/peilab/atst/flower/.cache/torch; source /project/peilab/atst/nimloth/.venv/bin/activate; if [ -f /project/peilab/atst/flower/.env ]; then set -a; source /project/peilab/atst/flower/.env; set +a; elif [ -f /project/peilab/atst/.env ]; then set -a; source /project/peilab/atst/.env; set +a; fi; export TOKENIZERS_PARALLELISM=true; export NCCL_DEBUG=WARN; export NCCL_IB_DISABLE=1'

{
  echo "=== SFT1 vagen79 train starting at $(date) ==="
  echo "Job ID: ${SLURM_JOB_ID} nodes: ${SLURM_JOB_NODELIST}"
  echo "Parallel: ${NNODES} nodes x ${NGPUS_PER_NODE} GPU = $((NNODES * NGPUS_PER_NODE)) DDP"
  echo "Rollout input: ${ROLLOUT_ROOT} (step ${ROLLOUT_STEP}.jsonl)"
  echo "Init HF: ${INIT_HF}"
  echo "Records: ${RECORDS_ROOT}"
  echo "Output: ${TRAIN_OUT}"
} | tee -a "${LOG}"

eval "${COMMON_ENV}"

if [ ! -d "${INIT_HF}" ]; then
  echo "ERROR missing init HF ${INIT_HF}" | tee -a "${LOG}"
  exit 3
fi

if [ ! -f "${RECORDS_ROOT}/train_success.jsonl" ] || [ ! -f "${RECORDS_ROOT}/val_all.jsonl" ]; then
  echo "=== Converting rollouts -> Nimloth records ===" | tee -a "${LOG}"
  PYTHONUNBUFFERED=1 python3 "${SCRIPTDIR}/convert_sft1_rollouts_to_nimloth.py" \
    --input-root "${ROLLOUT_ROOT}" \
    --output-root "${RECORDS_ROOT}" \
    --checkpoint-hf "${INIT_HF}" \
    --checkpoint-step "${ROLLOUT_STEP}" \
    --splits train val test \
    --force 2>&1 | tee -a "${LOG}"
fi

if [ ! -s "${RECORDS_ROOT}/train_success.jsonl" ]; then
  echo "ERROR empty train_success.jsonl" | tee -a "${LOG}"
  exit 4
fi

MODEL_PATH="${INIT_HF}"
RESUME_FLAG=()
if [ -f "${TRAIN_OUT}/best/training_state.pt" ]; then
  MODEL_PATH="${TRAIN_OUT}/best"
  RESUME_FLAG=(--resume)
  echo "=== Resume from ${TRAIN_OUT}/best ===" | tee -a "${LOG}"
fi

mapfile -t NODES < <($SL/scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}')
if [ -z "$HEAD_IP" ]; then
  HEAD_IP=$($SL/srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
fi
echo "torchrun rdzv: ${HEAD_IP}:${RDZV_PORT} nodes=${NODES[*]}" | tee -a "${LOG}"

export NCCL_SOCKET_IFNAME="^lo,docker0,virbr0"

TRAIN_PY="${SCRIPTDIR}/train_sft1_qwen25vl.py"
RESUME_ARGS=""
if [ ${#RESUME_FLAG[@]} -gt 0 ]; then
  RESUME_ARGS="--resume"
fi

echo "=== Launch 2-node DDP via srun (8 GPU) ===" | tee -a "${LOG}"
set +e
$SL/srun --overlap --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
  --cpus-per-task=${SLURM_CPUS_PER_TASK} \
  bash -lc "
    set -euo pipefail
    ${COMMON_ENV}
    export NCCL_SOCKET_IFNAME='^lo,docker0,virbr0'
    cd ${SCRIPTDIR}
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=${NGPUS_PER_NODE} \
      --node_rank=\${SLURM_PROCID} \
      --rdzv_id=${SLURM_JOB_ID} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${HEAD_IP}:${RDZV_PORT} \
      ${TRAIN_PY} \
      --model ${MODEL_PATH} \
      --train-jsonl ${RECORDS_ROOT}/train_success.jsonl \
      --val-jsonl ${RECORDS_ROOT}/val_all.jsonl \
      --output-dir ${TRAIN_OUT} \
      --epochs 20 \
      --batch-size 1 \
      --grad-accum 4 \
      --lr 1e-6 \
      --max-length 20000 \
      ${RESUME_ARGS}
  " 2>&1 | tee -a "${LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
if [ "$TRAIN_EXIT" -ne 0 ]; then
  echo "ERROR multinode SFT failed rc=${TRAIN_EXIT}" | tee -a "${LOG}"
  exit 1
fi

echo "=== SFT1 vagen79 training finished at $(date) ===" | tee -a "${LOG}"
