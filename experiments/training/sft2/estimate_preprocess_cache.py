#!/usr/bin/env python3
"""Estimate SFT2 preprocess cache disk usage (transition vs trajectory)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.training.sft2.preprocess_cache import encode_transition_item, encode_trajectory_record
from nimloth.wm.collate import transition_collate_for_qwen
from nimloth.wm.dataset import expand_record_transitions, load_jsonl_records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    return ap.parse_args()


def tensor_bytes(obj) -> int:
    if hasattr(obj, "nbytes"):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    return 0


def main() -> int:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)

    records = load_jsonl_records(args.train_jsonl, max_records=args.max_records)
    transition_bytes = 0
    trajectory_bytes = 0
    transitions = 0
    for record in records:
        steps = expand_record_transitions(record)
        transitions += len(steps)
        for sample in steps:
            item = transition_collate_for_qwen([sample])[0]
            enc = encode_transition_item(item, processor, args.max_length)
            transition_bytes += tensor_bytes(enc)
        traj = encode_trajectory_record(record, processor, args.max_length)
        trajectory_bytes += tensor_bytes(traj)

    per_transition = transition_bytes / max(transitions, 1)
    per_trajectory = trajectory_bytes / max(len(records), 1)
    report = {
        "sampled_records": len(records),
        "sampled_transitions": transitions,
        "transition_cache_bytes_sampled": transition_bytes,
        "trajectory_cache_bytes_sampled": trajectory_bytes,
        "avg_bytes_per_transition": per_transition,
        "avg_bytes_per_trajectory_record": per_trajectory,
        "estimated_transition_cache_gb_full": per_transition * transitions / (1024**3),
        "estimated_trajectory_cache_gb_full": per_trajectory * len(records) / (1024**3),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
