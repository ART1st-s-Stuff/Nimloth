#!/usr/bin/env bash
#SBATCH --job-name=sft1-dino-grid-ws2-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:30:00

set -euo pipefail
: "${REPO:?}" "${INIT_HF:?}" "${TRAIN_JSONL:?}" "${VAL_JSONL:?}" "${OUTPUT:?}" "${WANDB_RUN_NAME:?}"
ROOT=/project/peilab/atst/nimloth
PY="${ROOT}/.venv-vagen-main/bin/python3"
mkdir -p "${OUTPUT}"
set -a; source /project/peilab/atst/flower/.env; set +a
export WANDB_PROJECT=nimloth-sft1 WANDB_RUN_NAME
export HF_HOME=/project/peilab/atst/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" OMP_NUM_THREADS=8
cat >"${OUTPUT}/README.md" <<EOF
# DINO grid SFT1 world2 smoke

- Status: RUNNING
- Git commit: $(git -C "${REPO}" rev-parse HEAD)
- W&B: nimloth-sft1 / ${WANDB_RUN_NAME}
- Purpose: prove DDP with language LoRA + query adapter + shared projector and fail-closed frozen vision
- Data: success train max1 trajectory; val max1 trajectory; prefix-expanded
- Budget: 2 GPUs, 1 epoch, batch1/GA1, 30m
- Init: ${INIT_HF}
EOF
cd "${REPO}"
srun "${PY}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/training/sft1/train_dino_grid.py \
  --model "${INIT_HF}" --train-jsonl "${TRAIN_JSONL}" --val-jsonl "${VAL_JSONL}" \
  --output "${OUTPUT}" --dino-model facebook/dinov2-large \
  --epochs 1 --batch-size 1 --grad-accum 1 --max-train-records 1 --max-val-records 1 \
  --max-length 12000 --max-pixels 602112 --lr 2e-4 --query-lr 5e-4 \
  --projector-lr 2e-4 --dino-loss-weight 1.0 --wandb-run-name "${WANDB_RUN_NAME}"
