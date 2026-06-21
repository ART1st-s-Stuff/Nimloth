#!/usr/bin/env python3
"""Diagnose P4 prefix vs full latent mismatch (token alignment vs forward semantics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import (
    add_special_tokens,
    extract_latent_state,
    find_all_latent_state_indices,
    find_last_latent_state_index,
    special_token_ids,
)
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import forward_qwen_last_hidden
from nimloth.training.sft2.trajectory_forward import _batch_enc, _prefix_latent
from nimloth.wm.collate import messages_with_image_paths, prefix_messages_with_images
from nimloth.wm.dataset import expand_record_transitions, load_jsonl_records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    return ap.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device("cuda")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    record = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
    transitions = expand_record_transitions(record)
    image_paths = [str(p) for p in record.get("image_paths", [])]
    full_messages = messages_with_image_paths(list(record.get("messages", [])), image_paths)
    full_enc = encode_qwen_item(full_messages, processor, args.max_length, include_labels=False)
    full_ids = full_enc["input_ids"].tolist()
    full_indices = find_all_latent_state_indices(full_enc["input_ids"], token_id_map)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()
    full_hidden = forward_qwen_last_hidden(model, _batch_enc(full_enc), device)

    steps = []
    for sample in transitions[:8]:
        step = sample.step_index
        prefix_msgs = prefix_messages_with_images(sample)
        prefix_enc = encode_qwen_item(prefix_msgs, processor, args.max_length, include_labels=False)
        prefix_ids = prefix_enc["input_ids"].tolist()
        prefix_latent_pos = find_last_latent_state_index(prefix_enc["input_ids"], token_id_map)
        prefix_latent = _prefix_latent(model, sample, processor, token_id_map, device, args.max_length)
        full_latent_aligned = extract_latent_state(full_hidden[0], prefix_latent_pos)
        wrong_idx = full_indices[step] if step < len(full_indices) else None
        full_latent_wrong = (
            extract_latent_state(full_hidden[0], wrong_idx) if wrong_idx is not None else None
        )
        steps.append(
            {
                "step_index": step,
                "prefix_len": len(prefix_ids),
                "aligned_full_latent_idx": prefix_latent_pos,
                "wrong_full_latent_idx_step": wrong_idx,
                "prefix_ids_eq_full_prefix": full_ids[: len(prefix_ids)] == prefix_ids,
                "max_latent_abs_diff_aligned": float((prefix_latent - full_latent_aligned).abs().max().item()),
                "max_latent_abs_diff_wrong_idx": (
                    float((prefix_latent - full_latent_wrong).abs().max().item())
                    if full_latent_wrong is not None
                    else None
                ),
            }
        )

    print(
        json.dumps(
            {
                "record_id": record.get("id"),
                "num_transitions": len(transitions),
                "num_full_latent_tokens": len(full_indices),
                "steps": steps,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
