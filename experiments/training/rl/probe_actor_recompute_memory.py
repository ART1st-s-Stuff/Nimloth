#!/usr/bin/env python3
"""Run one real long-history actor recompute/backward and record CUDA peak memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen_tuning import configure_qwen_tuning
from nimloth.latent.extraction import add_special_tokens, special_token_ids
from nimloth.training.rl.rollout import RolloutTrajectory, validate_rl_policy_protocol
from nimloth.training.rl.trainer import (
    _temporary_deterministic_train,
    compute_policy_token_stats_for_batch,
    normalize_policy_parameter_dtype,
)
from nimloth.training.rl.vagen_protocol import thought_from_assistant_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def load_longest_trajectory(path: Path) -> RolloutTrajectory:
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    trajectories = [RolloutTrajectory.from_record(record) for record in records]
    if not trajectories:
        raise ValueError(f"no trajectories in {path}")
    trajectory = max(trajectories, key=lambda item: item.num_steps)
    if trajectory.num_steps < 20:
        raise ValueError(
            f"memory gate requires a20-turn trajectory, got {trajectory.num_steps}"
        )
    return trajectory


def main() -> int:
    args = parse_args()
    rank = int(__import__("os").environ["RANK"])
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    latent_token_count = validate_rl_policy_protocol(model.config)
    add_special_tokens(processor.tokenizer, latent_token_count=latent_token_count)
    token_id_map = special_token_ids(
        processor.tokenizer, latent_token_count=latent_token_count
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    tune_args = SimpleNamespace(
        lora=False,
        llm_tune="lora",
        vision_tune="freeze",
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        gradient_checkpointing=True,
    )
    model = configure_qwen_tuning(model, tune_args)
    normalize_policy_parameter_dtype(model, dtype=torch.bfloat16)
    model.to(device)

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )

    embedding = model.get_input_embeddings()
    if getattr(embedding, "padding_idx", None) is not None:
        embedding.padding_idx = None
    model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
    )

    trajectory = load_longest_trajectory(args.trajectories)
    step = trajectory.num_steps - 1
    item = {
        "image_history_paths": trajectory.image_paths[: step + 1],
        "system_prompt": trajectory.system_prompt,
        "observation_texts": trajectory.observation_texts[: step + 1],
        "assistant_responses": trajectory.assistant_responses[:step],
        "current_thought": thought_from_assistant_response(
            trajectory.assistant_responses[step]
        ),
        "thought_token_ids": trajectory.thought_token_ids[step],
        "taken_action_idx": trajectory.action_indices[step],
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    with _temporary_deterministic_train(model):
        log_probs, entropies, token_counts = compute_policy_token_stats_for_batch(
            [item],
            model,
            processor,
            token_id_map,
            device,
            history_window=112,
            temperature=args.temperature,
            latent_token_count=latent_token_count,
        )
        loss = -log_probs.mean() - 0.01 * entropies.mean()
    loss.backward()
    torch.cuda.synchronize(device)
    result = {
        "rank": rank,
        "record_id": trajectory.record_id,
        "step": step,
        "history_images": len(item["image_history_paths"]),
        "policy_tokens": token_counts[0],
        "loss": float(loss.detach().item()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    (args.output_dir / f"rank_{rank:05d}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    if rank == 0:
        print(json.dumps({"actor_memory_probe": "passed", **result}), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
