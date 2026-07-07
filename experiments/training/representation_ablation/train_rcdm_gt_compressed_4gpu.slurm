#!/bin/bash
#SBATCH --job-name=rcdm-gtc-4g
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/exp-latent-repr-ablation}
PY_VENV=${PY_VENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
PYTHON_BIN=${PYTHON_BIN:-${PY_VENV}/bin/python}
DATE_TAG=${DATE_TAG:-$(date +%F)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/${DATE_TAG}}
RUN_NAME=${RUN_NAME:-rcdm_gt_compressed_rollout4_b128_gt_current_5ep_4gpu}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${SLURM_JOB_ID:-local}}
QWEN_CHECKPOINT=${QWEN_CHECKPOINT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_lejepa_align_fn1_dgx56/ckpt_step1000_preserved}
RECORDS_ROOT=${RECORDS_ROOT:-/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format}
TRAIN_JSONL=${TRAIN_JSONL:-${RECORDS_ROOT}/train_all.jsonl}
VAL_JSONL=${VAL_JSONL:-${RECORDS_ROOT}/val_all.jsonl}
STATE_CACHE_DIR=${STATE_CACHE_DIR:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/cache/rcdm_gt_compressed_rollout4_b128_current}
EPOCHS=${EPOCHS:-5}
PER_RANK_BATCH_SIZE=${PER_RANK_BATCH_SIZE:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
MAX_VAL_BATCHES=${MAX_VAL_BATCHES:-32}
RESUME=${RESUME:-0}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
WANDB_ENTITY=${WANDB_ENTITY:-art2nd-hong-kong-university-of-science-and-technology}
WANDB_PROJECT=${WANDB_PROJECT:-nimloth}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}_${SLURM_JOB_ID:-local}}

cd "${REPO}"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p "${OUT_DIR}" /project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm
LOG=${OUT_DIR}/train_4gpu.log

export HOME=/project/peilab/atst/nimloth/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${REPO}/src:${REPO}/external/le-wm
export PATH=${PY_VENV}/bin:${PATH}
export WANDB_DIR=${OUT_DIR}/wandb
export WANDB_ENTITY=${WANDB_ENTITY}
if [[ -f /project/peilab/atst/flower/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /project/peilab/atst/flower/.env
  set +a
  export WANDB_ENTITY=${WANDB_ENTITY}
fi
export OMP_NUM_THREADS=8

cat > "${OUT_DIR}/README.md" <<README
# 4GPU RCDM GT compressed-token reconstruction

- purpose: train post-hoc RCDM on GT compressed Qwen vision-token conditions with 4 GPUs, using the existing compressed-token cache.
- git_commit: ${COMMIT}
- condition: flatten(AttentionTokenCompressor(Qwen vision tokens of current image)), K=16, D=512, cond_dim=8192.
- target: current image reconstruction from GT current compressed tokens.
- state_cache_dir: ${STATE_CACHE_DIR}
- train_jsonl: ${TRAIN_JSONL}
- val_jsonl: ${VAL_JSONL}
- split_semantics: train_all is training split; val_all is held-out validation split per records README/manifest.
- trainable: RCDM UNet only.
- frozen: Qwen/compressor are not used in training path; cache contains frozen GT compressed-token conditions.
- epochs: ${EPOCHS}
- per_rank_batch_size: ${PER_RANK_BATCH_SIZE}
- effective_batch_size: $((PER_RANK_BATCH_SIZE * 4))
- lr: 1e-4
- RCDM flags: upstream 128px flags from RCDM README.
- output_dir: ${OUT_DIR}
- wandb_entity: ${WANDB_ENTITY}
- wandb_project: ${WANDB_PROJECT}
- wandb_run_name: ${WANDB_RUN_NAME}
- resume: use \`RESUME=1 OUT_DIR=${OUT_DIR}\` or \`RESUME_CHECKPOINT=${OUT_DIR}/training_state_*.pt OUT_DIR=${OUT_DIR}\` with the same cache.
README

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

{
  echo "=== RCDM 4GPU GT compressed train $(date) ==="
  echo "job=${SLURM_JOB_ID:-local} node=${SLURM_JOB_NODELIST:-local} commit=${COMMIT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "out=${OUT_DIR}"
  echo "state_cache=${STATE_CACHE_DIR}"
  echo "epochs=${EPOCHS} per_rank_batch=${PER_RANK_BATCH_SIZE} effective_batch=$((PER_RANK_BATCH_SIZE * 4)) save_interval=${SAVE_INTERVAL}"
  echo "resume=${RESUME} resume_checkpoint=${RESUME_CHECKPOINT}"
  echo "wandb_entity=${WANDB_ENTITY} wandb_project=${WANDB_PROJECT}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m nimloth.training.reconstruction.rcdm_sft2 \
  --model "${QWEN_CHECKPOINT}" \
  --state-proj-checkpoint "${QWEN_CHECKPOINT}/state_proj.pt" \
  --wm-checkpoint "${QWEN_CHECKPOINT}/wm_predictor" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUT_DIR}" \
  --state-cache-dir "${STATE_CACHE_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${PER_RANK_BATCH_SIZE}" \
  --lr 1.0e-4 \
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --save-interval "${SAVE_INTERVAL}" \
  "${RESUME_ARGS[@]}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" 2>&1 | tee -a "${LOG}"

echo "=== done $(date) output=${OUT_DIR} ===" | tee -a "${LOG}"
