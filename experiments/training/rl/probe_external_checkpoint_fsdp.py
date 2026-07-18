#!/usr/bin/env python3
"""Tiny multimodal train-mode gate for external Qwen checkpoints plus FSDP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Qwen2_5_VLConfig

from nimloth.training.rl.fsdp import (
    apply_qwen_activation_checkpointing,
    count_activation_checkpoint_units,
    count_fsdp_units,
    qwen25vl_transformer_auto_wrap_policy,
)
from nimloth.training.rl.token_critic import IndependentQwenTokenCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tiny_config() -> Qwen2_5_VLConfig:
    return Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "attention_dropout": 0.0,
            "rope_scaling": {
                "rope_type": "default",
                "type": "default",
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "window_size": 4,
            "fullatt_block_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = IndependentQwenTokenCritic(tiny_config()).to(
        device=device, dtype=torch.bfloat16
    )
    activation_units = apply_qwen_activation_checkpointing(model)
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )

    model = FSDP(
        model,
        auto_wrap_policy=qwen25vl_transformer_auto_wrap_policy(),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
    )
    fsdp_units = count_fsdp_units(model)
    if activation_units != 2 or count_activation_checkpoint_units(model) != 2:
        raise RuntimeError("tiny Qwen must wrap one text and one vision block")
    if fsdp_units <= 1:
        raise RuntimeError("tiny Qwen external-checkpoint gate needs nested FSDP")
    if any(
        getattr(child, "gradient_checkpointing", False)
        for child in model.modules()
    ):
        raise RuntimeError("HF internal gradient checkpointing remained enabled")

    # Four 2x2 patches map to four image tokens.
    input_ids = torch.tensor(
        [[1, 62, 60, 60, 60, 60, 63, 2]], device=device
    )
    pixel_values = torch.randn(4, 12, device=device, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=False,
    )
    loss = output.logits.float().square().mean()
    loss.backward()
    model.clip_grad_norm_(1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    result = {
        "rank": rank,
        "loss": float(loss.detach().item()),
        "finite": bool(torch.isfinite(loss).item()),
        "activation_checkpoint_units": activation_units,
        "fsdp_units": fsdp_units,
        "image_tokens": 4,
        "optimizer_steps": 1,
    }
    (args.output_dir / f"rank_{rank:02d}.json").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
