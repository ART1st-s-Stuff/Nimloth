#!/usr/bin/env python3
"""Collect RL-compatible navigation trajectories with the Nimloth policy.

This is the rollout producer for distributed JSONL training.  It reuses
``EnvRolloutCollector`` so online and two-stage rollout use the same Nimloth
special-token action policy and emit the same trajectory schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


_NAV_DATASETS = (
    "base",
    "common_sense",
    "complex_instruction",
    "visual_appearance",
    "long_horizon",
    "base_train",
    "common_sense_train",
    "long_horizon_train",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect Nimloth RL trajectories")
    ap.add_argument("--model", type=Path, required=True, help="Full HF policy checkpoint")
    ap.add_argument("--env-url", required=True, help="VAGEN env server base URL")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--eval-set", choices=_NAV_DATASETS, required=True)
    ap.add_argument("--split", choices=("train", "val", "test", "eval"), required=True)
    ap.add_argument("--seed-offset", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--max-pixels", type=int, default=3136)
    return ap.parse_args(argv)


def load_qwen(model_path: Path, attn_implementation: str, max_pixels: int):
    """Load the rollout policy and its processor on the current CUDA device."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from nimloth.latent import add_special_tokens
    from nimloth.training.rl.rollout import validate_rl_policy_protocol

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = max_pixels
    n_added = add_special_tokens(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if n_added:
        model.resize_token_embeddings(len(processor.tokenizer))
    validate_rl_policy_protocol(model.config)
    model.eval().cuda()
    return model, processor


def validate_split(eval_set: str, split: str) -> None:
    """Prevent evaluation assets from being mislabeled as training data."""
    if split == "train" and not eval_set.endswith("_train"):
        raise ValueError(f"refusing to label eval dataset {eval_set!r} as training data")
    if split != "train" and eval_set.endswith("_train"):
        raise ValueError(f"training dataset {eval_set!r} must use --split train")


def validate_trajectories(records) -> None:
    """Reject incomplete records before they can enter FSDP training."""
    from nimloth.training.rl.rollout import validate_rollout_trajectory

    if not records:
        raise RuntimeError("rollout produced no complete trajectories")
    for record in records:
        if record.split == "train" and not record.image_paths:
            raise RuntimeError(f"training trajectory {record.record_id} has no images")
        try:
            validate_rollout_trajectory(record)
        except ValueError as error:
            raise RuntimeError(str(error)) from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split(args.eval_set, args.split)
    if not torch.cuda.is_available():
        raise RuntimeError("rollout_env requires CUDA")

    repo_root = Path(__file__).resolve().parents[3]
    for path in (repo_root / "src", repo_root / "external" / "VAGEN"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from nimloth.training.rl.rollout import EnvRolloutCollector

    model, processor = load_qwen(
        args.model, args.attn_implementation, args.max_pixels
    )
    collector = EnvRolloutCollector(
        qwen_model=model,
        processor=processor,
        env_url=args.env_url,
        device=torch.device("cuda"),
        seed_offset=args.seed_offset,
        temperature=args.temperature,
        top_p=args.top_p,
        eval_sets=(args.eval_set,),
        split=args.split,
    )
    trajectories = collector.collect(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir,
    )
    validate_trajectories(trajectories)
    print(json.dumps({
        "status": "ALL_OK",
        "num_trajectories": len(trajectories),
        "num_transitions": sum(t.num_steps for t in trajectories),
        "jsonl": str(args.output_dir / "trajectories.jsonl"),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
