#!/bin/bash
#SBATCH --job-name=vllm-nimloth-probe
#SBATCH --output=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth/outputs/experiments/navigation_baseline/slurm/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail
ROOT=/project/peilab/atst/nimloth
export HOME=${ROOT}/.home
export XDG_CACHE_HOME=${ROOT}/.cache
export VLLM_CACHE_ROOT=${ROOT}/.cache/vllm
mkdir -p "$HOME" "$VLLM_CACHE_ROOT"
source ${ROOT}/.venv/bin/activate

python3 <<'PY'
import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams

model = "/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_train_vagen79_qwen25vl/epoch_017"
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt = (
    "<|im_start|>system\nYou are a home robot.\n"
    "Respond in this format:\n"
    "<think>...</think><|latent_state|><|action_start|><|action_(idx)|><|action_end|>\n"
    "\n"
    "<|im_start|>user\n[Initial Observation]:\nHuman Instruction: go to sink\n"
    "Decide your next action(s).\n"
    "<|im_start|>assistant\n"
)
prompt_ids = tok.encode(prompt, add_special_tokens=False)
print("hf_vocab", len(tok))
print("action0_id", tok.convert_tokens_to_ids("<|action_(0)|>"))

# HF greedy generate (text-only probe)
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)
with torch.no_grad():
    out = m.generate(
        input_ids=torch.tensor([prompt_ids], device="cuda"),
        max_new_tokens=64,
        do_sample=False,
    )
hf_new = out[0, len(prompt_ids):].tolist()
print("hf_generated_ids", hf_new)
print("hf_decode", repr(tok.decode(hf_new, skip_special_tokens=False)))

# vLLM greedy generate
llm = LLM(
    model=model,
    trust_remote_code=True,
    max_model_len=4096,
    enforce_eager=True,
    gpu_memory_utilization=0.45,
    limit_mm_per_prompt={"image": 0},
)
vtok = llm.get_tokenizer()
print("vllm_vocab", len(vtok))
sp = SamplingParams(max_tokens=64, temperature=0.0, detokenize=False)
vout = llm.generate(prompts=[{"prompt_token_ids": prompt_ids}], sampling_params=sp)[0]
vids = list(vout.outputs[0].token_ids)
print("vllm_generated_ids", vids)
print("vllm_decode_hf_tok", repr(tok.decode(vids, skip_special_tokens=False)))
PY
