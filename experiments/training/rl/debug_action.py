#!/usr/bin/env python3
"""Debug script: test Nimloth action selection prompt format."""
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = "/project/peilab/atst/nimloth/outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2/export_best_hf"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.image_processor.min_pixels = 3136
processor.image_processor.max_pixels = 3136

from nimloth.latent.extraction import LatentActionTokens, special_token_ids, add_special_tokens
add_special_tokens(processor.tokenizer)
tokens = LatentActionTokens()
token_ids = special_token_ids(processor.tokenizer, tokens)
action_token_ids = [token_ids[t] for t in tokens.action_tokens]

print("action_tokens:", list(tokens.action_tokens))
print("action_token_ids:", action_token_ids)
for i, aid in enumerate(action_token_ids):
    print(f"  [{i}] {processor.tokenizer.decode([aid])!r} -> id={aid}")

img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

# --- Test 1: the exact format from _select_action_nimloth ---
print("\n=== Test 1: current _select_action_nimloth format ===")
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a navigation agent."}]},
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Observe the scene."},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "<think>What should I do?</think><|latent_state|><|action_start|>"},
    ]},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
enc = processor(text=[text], images=[img], return_tensors="pt")
input_ids = enc["input_ids"][0]
last_20 = input_ids[-20:].tolist()
print("last_20:", [processor.tokenizer.decode([t]) for t in last_20])
pos = (input_ids == token_ids[tokens.action_start]).nonzero(as_tuple=True)
print("action_start at:", pos[0].tolist() if pos[0].numel() > 0 else "NOT FOUND")

# Load model and check logits
print("\n=== Model forward ===")
from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True,
)
model.resize_token_embeddings(len(processor.tokenizer))
model.eval()
model.cuda()

with torch.no_grad():
    outputs = model(**{k: v.cuda() for k, v in enc.items()})
logits = outputs.logits[0, -1, :]
action_logits = logits[action_token_ids]
probs = torch.softmax(action_logits.float(), dim=-1)
print("action_logits:", {ACTION_NAMES[i]: f"{action_logits[i].item():.3f} ({probs[i].item():.3f})" for i in range(8)})
best = int(action_logits.argmax().item())
print(f"BEST: {ACTION_NAMES[best]} (idx={best})")

# --- Test 2: proper history format ---
print("\n=== Test 2: Nimloth history format (like SFT2 training) ===")
ACTION_NAMES = ["moveahead", "moveback", "moveright", "moveleft", "rotateright", "rotateleft", "lookup", "lookdown"]
history = ["moveahead"]  # one history action
img2 = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
img_past = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

messages2 = [
    {"role": "system", "content": [{"type": "text", "text": "You are a navigation agent."}]},
    {"role": "user", "content": [
        {"type": "image", "image": img_past},
        {"type": "text", "text": "Observe the scene."},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": f"<think>Previous step.</think><|latent_state|><|action_start|><|action_(0)|><|action_end|>"},
    ]},
    {"role": "user", "content": [
        {"type": "image", "image": img2},
        {"type": "text", "text": "Observe the scene after moveahead."},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "<think>Next action.</think><|latent_state|><|action_start|>"},
    ]},
]
text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=False)
enc2 = processor(text=[text2], images=[img_past, img2], return_tensors="pt")

with torch.no_grad():
    outputs2 = model(**{k: v.cuda() for k, v in enc2.items()})
logits2 = outputs2.logits[0, -1, :]
action_logits2 = logits2[action_token_ids]
probs2 = torch.softmax(action_logits2.float(), dim=-1)
print("action_logits:", {ACTION_NAMES[i]: f"{action_logits2[i].item():.3f} ({probs2[i].item():.3f})" for i in range(8)})
best2 = int(action_logits2.argmax().item())
print(f"BEST: {ACTION_NAMES[best2]} (idx={best2})")
