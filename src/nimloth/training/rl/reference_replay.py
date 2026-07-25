"""Persist frozen-reference CoT log-probs between vLLM rollout and PPO."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from nimloth.agent import PolicyReplayInput
from nimloth.backbone.qwen25vl.factory import load_backbone
from nimloth.backbone.qwen25vl.policy import (
    replay_policy_token_log_probs,
    validate_agent_policy_protocol,
)
from nimloth.rollout import (
    FreshRolloutManifest,
    load_trajectories,
    save_trajectories,
)
from nimloth.util.module import evaluating


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich fresh rollout trajectories with frozen reference logits"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-parallel-size", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-pixels", type=int, default=3136)
    return parser.parse_args(argv)


def _load_reference(args: argparse.Namespace, latent_token_count: int):
    load_args = argparse.Namespace(
        model=args.reference_model,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=False,
        llm_tune="freeze",
        vision_tune="freeze",
        lora=False,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        query_tune="freeze",
        resume=False,
    )
    return load_backbone(
        load_args,
        device=torch.device("cuda:0"),
        latent_token_count=latent_token_count,
        model_parallel_size=args.model_parallel_size,
    )


def _replay_input(trajectory, step: int) -> PolicyReplayInput:  # type: ignore[no-untyped-def]
    return PolicyReplayInput(
        prompt=trajectory.build_policy_prompt(step),
        action_index=trajectory.action_indices[step],
        sampling_temperature=trajectory.sampling_temperature,
        sampling_top_p=trajectory.sampling_top_p,
        latent_token_count=trajectory.resolved_latent_token_count(),
        credit_assignment=trajectory.policy_credit_assignment,
        token_trace=trajectory.policy_token_trace(step),
        assistant_response=trajectory.assistant_responses[step],
        planner_trace=trajectory.planner_policy_trace(step),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("reference replay requires CUDA")
    if args.model_parallel_size not in {1, 2}:
        raise ValueError("reference model_parallel_size currently supports 1 or 2")
    manifest = FreshRolloutManifest.read(args.manifest)
    if manifest.reference_policy_fingerprint is not None:
        raise ValueError("manifest is already reference-enriched")
    trajectories = load_trajectories(Path(manifest.trajectory_path))
    if len(trajectories) != manifest.num_trajectories:
        raise ValueError("manifest trajectory count does not match JSONL")
    latent_counts = {
        trajectory.resolved_latent_token_count() for trajectory in trajectories
    }
    if len(latent_counts) != 1:
        raise ValueError("reference replay cannot mix latent token counts")
    latent_token_count = latent_counts.pop()
    configured_count = validate_agent_policy_protocol(
        AutoConfig.from_pretrained(args.reference_model, trust_remote_code=True)
    )
    if configured_count != latent_token_count:
        raise ValueError("reference model latent token count does not match rollout")

    loaded = _load_reference(args, latent_token_count)
    model = loaded.backbone.model
    enriched = []
    with evaluating(model), torch.no_grad():
        for trajectory in trajectories:
            reference_rows: list[list[float | None]] = []
            for step in range(trajectory.num_steps):
                sample = _replay_input(trajectory, step)
                trace = sample.token_trace
                assert trace is not None
                output = replay_policy_token_log_probs(
                    samples=(sample,),
                    model=model,
                    processor=loaded.processor,
                    token_id_map=loaded.token_id_map,
                    device=torch.device("cuda:0"),
                    token_value_head=None,
                    compute_token_values=False,
                )
                if output.selected_full_log_probs is None:
                    raise RuntimeError("reference replay produced no CoT log-probs")
                values = iter(
                    float(value)
                    for value in output.selected_full_log_probs.cpu().tolist()
                )
                row: list[float | None] = []
                for selected, role in zip(
                    trace.loss_mask,
                    trace.token_roles,
                    strict=True,
                ):
                    row.append(next(values) if selected and role == "reasoning" else None)
                try:
                    next(values)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError("reference log-prob count exceeds trace")
                reference_rows.append(row)
            enriched.append(
                replace(
                    trajectory,
                    policy_reference_token_log_probs=reference_rows,
                )
            )

    trajectory_path = save_trajectories(enriched, args.output_dir)
    updated = manifest.with_reference(
        reference_policy_path=args.reference_model,
        trajectory_path=trajectory_path,
    )
    updated.write(args.manifest)
    print(
        json.dumps(
            {
                "status": "ALL_OK",
                "reference_policy_fingerprint": (
                    updated.reference_policy_fingerprint
                ),
                "num_trajectories": len(enriched),
                "trajectory_path": str(trajectory_path.resolve()),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
