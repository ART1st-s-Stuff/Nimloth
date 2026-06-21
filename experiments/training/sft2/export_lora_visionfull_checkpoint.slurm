#!/bin/bash
#SBATCH --job-name=sft2-export-vfull
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
set -euo pipefail
: "${CHECKPOINT:?CHECKPOINT required}"
: "${OUT_DIR:?OUT_DIR required}"
REPO=/project/peilab/atst/nimloth
export PATH="${REPO}/.venv/bin:${PATH}" PYTHONPATH="${REPO}/src"
export HF_HOME=/project/peilab/atst/.cache/huggingface TRANSFORMERS_CACHE=/project/peilab/atst/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${REPO}/.venv/bin/activate"
mkdir -p "${OUT_DIR}"
python3 "${REPO}/experiments/training/sft2/export_lora_visionfull_checkpoint.py" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_DIR}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  ${REQUIRE_VISION_FULL_STATE:+--require-vision-full-state} \
  ${NO_REQUIRE_VISION_FULL_STATE:+--no-require-vision-full-state}
python3 - <<PY
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
p = "${OUT_DIR}"
processor = AutoProcessor.from_pretrained(p, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(p, torch_dtype="auto", trust_remote_code=True, device_map="cpu")
print({"loaded": p, "tokenizer_len": len(processor.tokenizer), "vocab_size": model.config.vocab_size})
PY
