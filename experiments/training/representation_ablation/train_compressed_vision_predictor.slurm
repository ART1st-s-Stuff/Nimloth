#!/bin/bash
#SBATCH --job-name=repr-compress-pred
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth/.worktree/exp-latent-repr-ablation}
# Adapter-backed SFT2 checkpoints need the newer transformers stack in .venv-vagen-main;
# the repository code is still provided through PYTHONPATH below.  Use PY_VENV
# rather than VENV because some cluster shells export VENV already.
PY_VENV=${PY_VENV:-/project/peilab/atst/nimloth/.venv-vagen-main}
RUN_KIND=${RUN_KIND:-rollout4}  # one of: 1step, rollout4
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-manual}
DATE_TAG=${DATE_TAG:-$(date +%F)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/${DATE_TAG}}
QWEN_CHECKPOINT=${QWEN_CHECKPOINT:-/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-07-01/sft2_lejepa_align_fn1_dgx56/ckpt_step1000_preserved}
RECORDS_ROOT=${RECORDS_ROOT:-/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_sft_records_vagen79_nimloth_format}
TRAIN_JSONL=${TRAIN_JSONL:-${RECORDS_ROOT}/train_all.jsonl}
VAL_JSONL=${VAL_JSONL:-${RECORDS_ROOT}/val_all.jsonl}
MAX_RECORDS=${MAX_RECORDS:--1}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1.0e-4}
LAMBDA_SIGREG=${LAMBDA_SIGREG:-0.1}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
RUN_EVAL=${RUN_EVAL:-1}
EVAL_MAX_RECORDS=${EVAL_MAX_RECORDS:-360}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}

case "${RUN_KIND}" in
  1step)
    TEMPLATE=${REPO}/configs/training/representation_ablation/e_compressed_qwen_vision_tokens_predictor_1step.yaml
    EVAL_TEMPLATE=${REPO}/configs/eval/representation_ablation/e_compressed_qwen_vision_tokens_predictor_1step.yaml
    ;;
  rollout4)
    TEMPLATE=${REPO}/configs/training/representation_ablation/e_compressed_qwen_vision_tokens_predictor_rollout4.yaml
    EVAL_TEMPLATE=${REPO}/configs/eval/representation_ablation/e_compressed_qwen_vision_tokens_predictor_rollout4.yaml
    ;;
  *)
    echo "ERROR unknown RUN_KIND=${RUN_KIND}; expected 1step or rollout4" >&2
    exit 2
    ;;
esac

mkdir -p /project/peilab/atst/nimloth/outputs/experiments/representation_ablation/slurm
cd "${REPO}"
COMMIT=$(git rev-parse --short HEAD)
JOB_TAG=${SLURM_JOB_ID:-local}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/compressed_vision_${RUN_KIND}_${OUTPUT_SUFFIX}_${COMMIT}_${JOB_TAG}}
mkdir -p "${OUT_DIR}"
TRAIN_CONFIG=${OUT_DIR}/config.train.yaml
EVAL_CONFIG=${OUT_DIR}/config.eval.yaml
LOG=${OUT_DIR}/train.log

cat > "${OUT_DIR}/README.md" <<README
# compressed vision-token predictor train

- purpose: predictor capacity diagnostic in compressed direct Qwen vision-token space.
- git_commit: ${COMMIT}
- run_kind: ${RUN_KIND}
- code_entry: python -m nimloth.training.representation_ablation.train
- train_config: ${TRAIN_CONFIG}
- eval_config: ${EVAL_CONFIG}
- qwen_checkpoint: ${QWEN_CHECKPOINT}
- train_jsonl: ${TRAIN_JSONL}
- val_jsonl: ${VAL_JSONL}
- split_semantics: train_all is training split; val_all is held-out validation split, per sft1_sft_records_vagen79_nimloth_format/README.md and manifest.json.
- trainable_modules: AttentionTokenCompressor, TokenSetWMPredictor
- frozen_modules: Qwen2.5-VL checkpoint including vision encoder
- loss: predictor rollout MSE + lambda_sigreg * SIGReg(compressed tokens); no reconstruction loss
- resume_strategy: periodic module checkpoints every SAVE_INTERVAL steps; rerun with init.compressor_checkpoint/init.wm_predictor_checkpoint if manual resume is needed
- output_dir: ${OUT_DIR}
- wandb: disabled
- monitor: train_step_log.csv loss, predictor_rollout_mse, sigreg_loss; optional eval summary predictor MSE/cosine and rollout metrics
README

export HOME=/project/peilab/atst/nimloth/.home
export HF_HOME=/project/peilab/atst/.cache/huggingface
export TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export TORCH_HOME=/project/peilab/atst/flower/.cache/torch
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${REPO}/src:${REPO}/external/le-wm
export PATH=${PY_VENV}/bin:${PATH}
# shellcheck disable=SC1091
source "${PY_VENV}/bin/activate"

python - <<PY
from pathlib import Path
import yaml

template = Path("${TEMPLATE}")
out = Path("${TRAIN_CONFIG}")
cfg = yaml.safe_load(template.read_text())
cfg["experiment"]["output_dir"] = "${OUT_DIR}"
cfg["init"]["qwen_checkpoint"] = "${QWEN_CHECKPOINT}"
cfg["data"]["train_jsonl"] = "${TRAIN_JSONL}"
cfg["data"]["val_jsonl"] = "${VAL_JSONL}"
cfg["data"]["max_records"] = int("${MAX_RECORDS}")
cfg["train"]["batch_size"] = int("${BATCH_SIZE}")
cfg["train"]["epochs"] = int("${EPOCHS}")
cfg["train"]["lr"] = float("${LR}")
cfg["train"]["lambda_sigreg"] = float("${LAMBDA_SIGREG}")
cfg["train"]["save_interval"] = int("${SAVE_INTERVAL}")
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

et = Path("${EVAL_TEMPLATE}")
eout = Path("${EVAL_CONFIG}")
ecfg = yaml.safe_load(et.read_text())
ecfg["experiment"]["name"] = f"eval_{cfg['experiment']['name']}"
ecfg["init"]["qwen_checkpoint"] = "${QWEN_CHECKPOINT}"
ecfg["init"]["compressor_checkpoint"] = str(Path("${OUT_DIR}") / "compressor")
ecfg["init"]["wm_predictor_checkpoint"] = str(Path("${OUT_DIR}") / "wm_predictor")
ecfg["data"]["val_jsonl"] = "${VAL_JSONL}"
ecfg["data"]["max_records"] = int("${EVAL_MAX_RECORDS}")
ecfg["eval"]["batch_size"] = int("${EVAL_BATCH_SIZE}")
ecfg["experiment"]["output_dir"] = str(Path("${OUT_DIR}") / "eval")
eout.write_text(yaml.safe_dump(ecfg, sort_keys=False), encoding="utf-8")
PY

{
  echo "=== compressed vision predictor train $(date) ==="
  echo "job=${SLURM_JOB_ID:-local} node=${SLURM_JOB_NODELIST:-local}"
  echo "repo=${REPO} commit=${COMMIT}"
  echo "python=$(which python) py_venv=${PY_VENV}"
  echo "template=${TEMPLATE}"
  echo "train_config=${TRAIN_CONFIG}"
  echo "output=${OUT_DIR}"
  echo "qwen_checkpoint=${QWEN_CHECKPOINT}"
  echo "train_jsonl=${TRAIN_JSONL} max_records=${MAX_RECORDS} batch_size=${BATCH_SIZE} epochs=${EPOCHS}"
  echo "lambda_sigreg=${LAMBDA_SIGREG} run_eval=${RUN_EVAL}"
} | tee -a "${LOG}"

python -m nimloth.training.representation_ablation.train \
  --config "${TRAIN_CONFIG}" \
  --output-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG}"

if [ "${RUN_EVAL}" = "1" ]; then
  echo "=== eval $(date) ===" | tee -a "${LOG}"
  python -m nimloth.eval.representation_ablation \
    --config "${EVAL_CONFIG}" \
    --output-dir "${OUT_DIR}/eval" 2>&1 | tee -a "${LOG}"
fi

echo "=== done $(date) output=${OUT_DIR} ===" | tee -a "${LOG}"
