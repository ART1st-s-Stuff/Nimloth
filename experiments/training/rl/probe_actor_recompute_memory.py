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
from nimloth.training.rl.fsdp import (
    count_fsdp_units,
    qwen25vl_transformer_auto_wrap_policy,
)
from nimloth.training.rl.token_critic import (
    compute_token_values_for_batch,
    load_independent_qwen_critic,
)
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
        auto_wrap_policy=qwen25vl_transformer_auto_wrap_policy(),
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

    actor_fsdp_units = count_fsdp_units(model)
    if actor_fsdp_units <= 1:
        raise RuntimeError("actor memory probe requires nested FSDP units")
    actor_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-6,
    )

    def actor_step() -> tuple[float, list[int]]:
        actor_optimizer.zero_grad(set_to_none=True)
        with _temporary_deterministic_train(model):
            log_probs, entropies, counts = compute_policy_token_stats_for_batch(
                [item],
                model,
                processor,
                token_id_map,
                device,
                history_window=112,
                temperature=args.temperature,
                latent_token_count=latent_token_count,
            )
            actor_loss = -log_probs.mean() - 0.01 * entropies.mean()
        actor_loss.backward()
        model.clip_grad_norm_(1.0)
        actor_optimizer.step()
        return float(actor_loss.detach().item()), counts

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    first_actor_loss, token_counts = actor_step()
    first_actor_peak = torch.cuda.max_memory_reserved(device) / 2**30

    critic = load_independent_qwen_critic(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        gradient_checkpointing=True,
    ).to(device)
    critic_embedding = critic.get_input_embeddings()
    if getattr(critic_embedding, "padding_idx", None) is not None:
        critic_embedding.padding_idx = None
    critic = FSDP(
        critic,
        auto_wrap_policy=qwen25vl_transformer_auto_wrap_policy(),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
    )
    critic_fsdp_units = count_fsdp_units(critic)
    if critic_fsdp_units <= 1:
        raise RuntimeError("critic memory probe requires nested FSDP units")
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-5)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    critic_optimizer.zero_grad(set_to_none=True)
    with _temporary_deterministic_train(critic):
        predicted_values, critic_counts = compute_token_values_for_batch(
            [item],
            critic,
            processor,
            token_id_map,
            device,
            history_window=112,
            latent_token_count=latent_token_count,
        )
        critic_loss = predicted_values.square().mean()
    critic_loss.backward()
    critic.clip_grad_norm_(1.0)
    critic_optimizer.step()
    critic_peak = torch.cuda.max_memory_reserved(device) / 2**30

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    steady_actor_loss, steady_counts = actor_step()
    torch.cuda.synchronize(device)
    result = {
        "rank": rank,
        "record_id": trajectory.record_id,
        "step": step,
        "history_images": len(item["image_history_paths"]),
        "policy_tokens": token_counts[0],
        "critic_tokens": critic_counts[0],
        "actor_fsdp_units": actor_fsdp_units,
        "critic_fsdp_units": critic_fsdp_units,
        "first_actor_loss": first_actor_loss,
        "steady_actor_loss": steady_actor_loss,
        "critic_loss": float(critic_loss.detach().item()),
        "first_actor_peak_reserved_gib": first_actor_peak,
        "critic_peak_reserved_gib": critic_peak,
        "steady_actor_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
        ),
        "steady_actor_peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
        ),
    }
    if critic_counts != token_counts or steady_counts != token_counts:
        raise RuntimeError(
            "actor/critic token counts changed during memory probe: "
            f"actor={token_counts}, critic={critic_counts}, steady={steady_counts}"
        )
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
