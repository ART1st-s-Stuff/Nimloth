#!/usr/bin/env python3
"""Real TP vLLM gate for batched multimodal PlannerPolicyHead states.

The gate replays the first policy prefix from distinct persisted trajectories,
generates them in one vLLM batch, and checks that each captured action-boundary
logit vector reproduces that same request's behavior action log-probabilities.
With temperature/top-p fixed to 1, this is a direct request-identity check: a
swapped state is compared against the wrong behavior distribution and fails.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-response-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    return parser.parse_args()


def _load_gate_prompts(path: Path, count: int) -> tuple[Any, ...]:
    from nimloth.agent import AgentPrompt, PromptTemplateSpec

    if count < 2:
        raise ValueError("identity gate requires at least two requests")
    prompts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            policy_messages = record.get("policy_messages")
            image_paths = record.get("image_paths")
            if not policy_messages or not image_paths:
                raise ValueError("gate trajectory has no first policy prefix/image")
            messages = policy_messages[0]
            if messages[-1].get("content") != "<think>":
                raise ValueError("gate policy prefix does not end with '<think>'")
            image_path = Path(image_paths[0])
            with Image.open(image_path) as image:
                copied_image = image.convert("RGB").copy()
            prompts.append(
                AgentPrompt(
                    messages=tuple(dict(message) for message in messages),
                    images=(copied_image,),
                    template=PromptTemplateSpec.from_record(record["prompt_template"]),
                )
            )
            if len(prompts) == count:
                break
    if len(prompts) != count:
        raise ValueError(f"gate requested {count} prompts but loaded {len(prompts)}")
    return tuple(prompts)


def _validate_generations(
    generations: tuple[Any, ...],
    *,
    expected_count: int,
    latent_token_count: int,
    action_count: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(generations) != expected_count:
        raise RuntimeError(
            f"vLLM gate returned {len(generations)} requests, expected {expected_count}"
        )
    rows = []
    for index, generated in enumerate(generations):
        decision = generated.qwen_decision
        state = generated.policy_state
        if state.latent_hidden.ndim != 2 or state.latent_hidden.shape[0] != latent_token_count:
            raise RuntimeError(
                f"request {index} latent shape is {tuple(state.latent_hidden.shape)}"
            )
        if state.action_logits.shape != (action_count,):
            raise RuntimeError(
                f"request {index} action-logit shape is {tuple(state.action_logits.shape)}"
            )
        if not torch.isfinite(state.latent_hidden).all() or not torch.isfinite(
            state.action_logits
        ).all():
            raise RuntimeError(f"request {index} captured non-finite state")
        behavior = torch.tensor(decision.action_log_probs, dtype=torch.float32)
        if behavior.shape != (action_count,) or not torch.isfinite(behavior).all():
            raise RuntimeError(f"request {index} has invalid behavior action log-probs")
        captured = torch.log_softmax(state.action_logits.float(), dim=-1)
        if not torch.allclose(captured, behavior, atol=atol, rtol=rtol):
            max_error = float((captured - behavior).abs().max().item())
            raise RuntimeError(
                "captured action logits do not reproduce request behavior "
                f"log-probs: request={index}, max_abs_error={max_error}"
            )
        rows.append(
            {
                "index": index,
                "action_index": int(decision.action_index),
                "response_chars": len(decision.response or ""),
                "latent_shape": list(state.latent_hidden.shape),
                "action_logit_shape": list(state.action_logits.shape),
                "max_log_prob_error": float((captured - behavior).abs().max().item()),
            }
        )
    pairwise_logit_delta = float(
        (generations[0].policy_state.action_logits - generations[1].policy_state.action_logits)
        .abs()
        .max()
        .item()
    )
    pairwise_latent_delta = float(
        (generations[0].policy_state.latent_hidden - generations[1].policy_state.latent_hidden)
        .abs()
        .max()
        .item()
    )
    if not math.isfinite(pairwise_logit_delta + pairwise_latent_delta) or max(
        pairwise_logit_delta, pairwise_latent_delta
    ) <= 1e-5:
        raise RuntimeError("distinct multimodal requests produced indistinguishable states")
    return {
        "status": "ALL_OK",
        "requests": rows,
        "pairwise_action_logit_max_abs": pairwise_logit_delta,
        "pairwise_latent_max_abs": pairwise_latent_delta,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least two")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be positive")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("comparison tolerances must be non-negative")

    from transformers import AutoConfig

    from nimloth.backbone.qwen25vl.loading import load_qwen_processor
    from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
    from nimloth.backbone.qwen25vl.vllm_policy import QwenVLLMAgentPolicy

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    latent_token_count = validate_agent_policy_protocol(config)
    processor = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=latent_token_count,
    ).processor
    prompts = _load_gate_prompts(args.trajectories, args.batch_size)
    policy = QwenVLLMAgentPolicy.from_model(
        str(args.model),
        processor=processor,
        max_pixels=args.max_pixels,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=1.0,
        top_p=1.0,
        max_model_len=args.max_model_len,
        max_images=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        latent_token_count=latent_token_count,
        credit_assignment="token",
        max_response_tokens=args.max_response_tokens,
        distributed_executor_backend="mp",
        enforce_eager=True,
        capture_policy_state=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0.0,
    )
    generations = policy.select_responses_with_state(prompts)
    result = _validate_generations(
        generations,
        expected_count=args.batch_size,
        latent_token_count=latent_token_count,
        action_count=len(policy.action_token_ids),
        atol=args.atol,
        rtol=args.rtol,
    )
    result.update(
        {
            "model": str(args.model.resolve()),
            "trajectories": str(args.trajectories.resolve()),
            "tensor_parallel_size": args.tensor_parallel_size,
            "latent_token_count": latent_token_count,
        }
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
