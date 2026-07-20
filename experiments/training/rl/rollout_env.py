#!/usr/bin/env python3
"""Collect RL-compatible navigation trajectories with the Nimloth policy.

This is the rollout producer for distributed JSONL training.  It reuses
``EnvRolloutCollector`` so online and two-stage rollout use the same Nimloth
special-token action policy and emit the same trajectory schema.
"""

from __future__ import annotations

import argparse
import json
import math
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
    ap.add_argument("--policy", choices=("qwen", "wm_value"), default="qwen")
    ap.add_argument(
        "--wm-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint root containing state_proj.pt, wm_predictor/, value_head/",
    )
    ap.add_argument("--fast-path-horizon", type=int, default=2)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--max-pixels", type=int, default=3136)
    return ap.parse_args(argv)


def load_qwen(model_path: Path, attn_implementation: str, max_pixels: int):
    """Load the rollout policy and its processor on the current CUDA device."""
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from nimloth.latent import add_special_tokens
    from nimloth.training.rl.rollout import (
        qwen_hidden_size_from_config,
        validate_rl_policy_protocol,
    )

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    protocol = validate_rl_policy_protocol(model_config)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = max_pixels
    n_added = add_special_tokens(
        processor.tokenizer, latent_token_count=protocol.latent_token_count
    )
    if n_added:
        raise ValueError(
            "metadata-bearing rollout checkpoint processor is missing "
            f"{n_added} required Nimloth special tokens"
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if model.get_input_embeddings().weight.shape[0] != len(processor.tokenizer):
        raise ValueError(
            "rollout checkpoint model/tokenizer vocabulary mismatch: "
            f"model={model.get_input_embeddings().weight.shape[0]}, "
            f"tokenizer={len(processor.tokenizer)}"
        )
    loaded_protocol = validate_rl_policy_protocol(model.config)
    if loaded_protocol != protocol:
        raise ValueError(
            f"AutoConfig/model protocol mismatch: {protocol} vs {loaded_protocol}"
        )
    model.eval().cuda()
    return model, processor, protocol, qwen_hidden_size_from_config(model.config)


def load_wm_modules(checkpoint: Path, *, qwen_hidden_dim: int, latent_token_count: int):
    """Load the frozen projector/predictor/value modules for wm_value rollout."""

    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead

    predictor = LatentWMPredictor.load_checkpoint(checkpoint / "wm_predictor")
    state_proj = StateProjector(
        qwen_hidden_dim=qwen_hidden_dim,
        lewm_emb_dim=predictor.emb_dim,
        latent_token_count=latent_token_count,
    )
    state_proj.load_state_dict(
        torch.load(checkpoint / "state_proj.pt", map_location="cpu", weights_only=True)
    )
    value_head = ValueHead.load_checkpoint(
        checkpoint / "value_head", emb_dim=predictor.emb_dim
    )
    for module in (state_proj, predictor, value_head):
        module.eval().cuda()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return state_proj, predictor, value_head


def validate_split(eval_set: str, split: str) -> None:
    """Prevent evaluation assets from being mislabeled as training data."""
    if split == "train" and not eval_set.endswith("_train"):
        raise ValueError(f"refusing to label eval dataset {eval_set!r} as training data")
    if split != "train" and eval_set.endswith("_train"):
        raise ValueError(f"training dataset {eval_set!r} must use --split train")


def validate_trajectories(records) -> None:
    """Reject incomplete records before they can enter FSDP training."""
    if not records:
        raise RuntimeError("rollout produced no complete trajectories")
    for record in records:
        if record.split == "train" and not record.image_paths:
            raise RuntimeError(f"training trajectory {record.record_id} has no images")
        if len(record.image_paths) != record.num_steps + 1:
            raise RuntimeError(
                f"trajectory {record.record_id}: images={len(record.image_paths)} "
                f"but actions={record.num_steps}"
            )
        if len(record.action_log_probs) != record.num_steps:
            raise RuntimeError(
                f"trajectory {record.record_id}: action_log_probs="
                f"{len(record.action_log_probs)} but actions={record.num_steps}"
            )
        if len(record.policy_sources) not in (0, record.num_steps):
            raise RuntimeError(f"trajectory {record.record_id} has misaligned policy_sources")
        if len(record.state_sources) not in (0, record.num_steps):
            raise RuntimeError(f"trajectory {record.record_id} has misaligned state_sources")
        if len(record.fast_path_steps) not in (0, record.num_steps):
            raise RuntimeError(f"trajectory {record.record_id} has misaligned fast_path_steps")
        sources = record.policy_sources or ["qwen"] * record.num_steps
        state_sources = record.state_sources or ["qwen_gt"] * record.num_steps
        fast_steps = record.fast_path_steps or [0] * record.num_steps
        if record.rollout_policy == "wm_value":
            if record.fast_path_horizon < 1 or any(
                source != "wm_value" for source in sources
            ):
                raise RuntimeError(
                    f"trajectory {record.record_id} has invalid wm_value policy metadata"
                )
            for step, (state_source, fast_step) in enumerate(
                zip(state_sources, fast_steps, strict=True)
            ):
                expected_fast_step = step % record.fast_path_horizon
                expected_state_source = (
                    "qwen_gt" if expected_fast_step == 0 else "wm_predicted"
                )
                if fast_step != expected_fast_step or state_source != expected_state_source:
                    raise RuntimeError(
                        f"trajectory {record.record_id} step {step} has invalid fast-path "
                        f"metadata: state={state_source}, fast_step={fast_step}"
                    )
        elif record.rollout_policy == "qwen":
            if record.action_log_prob_semantics != "sampling_distribution_v1":
                raise RuntimeError(
                    f"trajectory {record.record_id} lacks exact Qwen behavior-logprob semantics"
                )
            if any(source != "qwen" for source in sources):
                raise RuntimeError(
                    f"trajectory {record.record_id} mixes non-Qwen behavior into qwen policy"
                )
        else:
            raise RuntimeError(
                f"trajectory {record.record_id} has unknown rollout_policy "
                f"{record.rollout_policy!r}"
            )
        for step, (source, log_probs) in enumerate(
            zip(sources, record.action_log_probs, strict=True)
        ):
            if source == "qwen":
                chosen = record.action_indices[step]
                if (
                    log_probs is None
                    or len(log_probs) != 8
                    or log_probs[chosen] is None
                    or not math.isfinite(float(log_probs[chosen]))
                    or any(
                        value is not None and not math.isfinite(float(value))
                        for value in log_probs
                    )
                ):
                    raise RuntimeError(
                        f"trajectory {record.record_id} has invalid Qwen behavior log-probs"
                    )
            elif source == "wm_value":
                if log_probs is not None:
                    raise RuntimeError(
                        f"trajectory {record.record_id}: wm_value action must not carry "
                        "Qwen behavior log-probs"
                    )
            else:
                raise RuntimeError(
                    f"trajectory {record.record_id} has unknown policy source {source!r}"
                )
        if not record.nav_instruction:
            raise RuntimeError(f"trajectory {record.record_id} has no navigation instruction")


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

    model, processor, protocol, qwen_hidden_dim = load_qwen(
        args.model, args.attn_implementation, args.max_pixels
    )
    state_proj = predictor = value_head = None
    if args.policy == "wm_value":
        checkpoint = args.wm_checkpoint or args.model
        state_proj, predictor, value_head = load_wm_modules(
            checkpoint,
            qwen_hidden_dim=qwen_hidden_dim,
            latent_token_count=protocol.latent_token_count,
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
        rollout_policy=args.policy,
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=value_head,
        fast_path_horizon=args.fast_path_horizon,
        latent_token_count=protocol.latent_token_count,
        latent_query_mode=protocol.latent_query_mode,
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
        "policy": args.policy,
        "latent_token_count": protocol.latent_token_count,
        "jsonl": str(args.output_dir / "trajectories.jsonl"),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
