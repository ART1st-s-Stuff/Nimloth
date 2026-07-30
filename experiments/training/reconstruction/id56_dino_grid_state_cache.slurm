#!/usr/bin/env bash
#SBATCH --job-name=id56-grid-cache
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT=/project/peilab/atst/nimloth
PYTHON=${ROOT}/.venv-vagen-main/bin/python3
SLURM_BIN=/cm/shared/apps/slurm/current/bin
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

: "${WORKTREE:?set WORKTREE to the clean remote worktree}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the exact cache-builder commit}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the run-owned resumable cache root}"

SFT2_CHECKPOINT=${SFT2_CHECKPOINT:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-29/sft2/56_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16_px100352/train_ws16/epoch_002}
DATA_ROOT=${DATA_ROOT:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/52_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/data}
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_ROOT}/train_terminal_cot_migrated.jsonl}
VAL_JSONL=${VAL_JSONL:-${DATA_ROOT}/val_terminal_cot_migrated.jsonl}
PREPROCESS_CACHE=${PREPROCESS_CACHE:-${ROOT}/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-28/sft2/53_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga8_ws8_px100352/cache/preprocess}
PROCESSOR_SOURCE=${PROCESSOR_SOURCE:-${ROOT}/outputs/experiments/sft1_checkpoint_merge_fix/2026-07-24/3_k16_ep5_untied_lm_head_restore/hf_merged}

actual_commit=$(git -C "${WORKTREE}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "commit mismatch: expected=${EXPECTED_COMMIT}, actual=${actual_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${WORKTREE}" status --porcelain)" ]]; then
  echo "remote worktree is dirty" >&2
  git -C "${WORKTREE}" status --short >&2
  exit 2
fi
if [[ ! -f "${WORKTREE}/external/le-wm/module.py" ]]; then
  echo "external/le-wm is not initialized in ${WORKTREE}" >&2
  exit 2
fi
for path in \
  "${PYTHON}" \
  "${SFT2_CHECKPOINT}/model.safetensors.index.json" \
  "${SFT2_CHECKPOINT}/state_proj.pt" \
  "${SFT2_CHECKPOINT}/wm_predictor/config.json" \
  "${SFT2_CHECKPOINT}/wm_predictor/predictor.pt" \
  "${TRAIN_JSONL}" \
  "${VAL_JSONL}" \
  "${PREPROCESS_CACHE}/train/manifest.json" \
  "${PREPROCESS_CACHE}/val/manifest.json" \
  "${PROCESSOR_SOURCE}/config.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 2
  fi
done

export PYTHONPATH=${WORKTREE}/src:${WORKTREE}/external/le-wm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker0,virbr0'
export OMP_NUM_THREADS=4

MASTER_ADDR=$("${SLURM_BIN}/scontrol" show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=8

echo "cache launch commit=${EXPECTED_COMMIT} job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} world=$((NNODES * NPROC_PER_NODE)) output=${OUTPUT_DIR}"

"${SLURM_BIN}/srun" \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task=64 \
  bash -lc '
    set -euo pipefail
    gpu_count=$('"${PYTHON}"' -c "import torch; print(torch.cuda.device_count())")
    if [[ "${gpu_count}" != "${NPROC_PER_NODE}" ]]; then
      echo "expected ${NPROC_PER_NODE} visible GPUs on $(hostname), got ${gpu_count}" >&2
      exit 2
    fi
    '"${PYTHON}"' -c "import torch; p=torch.cuda.get_device_properties(0); assert p.total_memory >= 75*1024**3, p.total_memory; print(torch.cuda.get_device_name(0), round(p.total_memory/1024**3,2))"
    exec '"${PYTHON}"' -u -m torch.distributed.run \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${SLURM_PROCID}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      -m nimloth.training.reconstruction.cache_dino_grid_states \
      --sft2-checkpoint '"${SFT2_CHECKPOINT}"' \
      --train-jsonl '"${TRAIN_JSONL}"' \
      --val-jsonl '"${VAL_JSONL}"' \
      --preprocess-cache-dir '"${PREPROCESS_CACHE}"' \
      --preprocess-cache-processor-source '"${PROCESSOR_SOURCE}"' \
      --output-dir '"${OUTPUT_DIR}"' \
      --git-commit '"${EXPECTED_COMMIT}"' \
      --max-length 12000 \
      --max-pixels 100352 \
      --latent-token-count 16 \
      --attn-implementation flash_attention_2 \
      --encode-batch-size 1 \
      --state-cache-shard-size 128
  '

if [[ ! -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "cache command returned without a completed summary: ${OUTPUT_DIR}" >&2
  exit 2
fi
