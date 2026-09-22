"""FSDP ownership policy for direct-Qwen RL training."""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import CustomPolicy

STRATEGY_ID = "direct_qwen_block_full_shard_bf16_v1"


def qwen_block_wrap_policy(
    model: nn.Module,
) -> tuple[CustomPolicy, dict[str, int]]:
    """Shard Qwen decoder/vision blocks without changing parameter precision."""

    visual = getattr(model, "visual", None)
    if not isinstance(visual, nn.Module):
        raise TypeError("direct-Qwen RL FSDP requires Qwen's visual module")

    block_names = {"Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"}
    block_names.update(str(name) for name in (getattr(model, "_no_split_modules", ()) or ()))
    targets = {
        module
        for module in model.modules()
        if module.__class__.__name__ in block_names
    }
    targets.add(visual)

    counts = Counter(module.__class__.__name__ for module in targets)
    decoder_count = sum(
        count for name, count in counts.items() if "DecoderLayer" in name
    )
    vision_count = sum(
        count for name, count in counts.items() if "VisionBlock" in name
    )
    if decoder_count < 1 or vision_count < 1:
        raise RuntimeError(
            "direct-Qwen RL FSDP did not find both decoder and vision blocks: "
            f"decoder={decoder_count}, vision={vision_count}"
        )

    # Parameters are already loaded in the requested BF16 dtype.  Supplying an
    # FSDP MixedPrecision policy here would change the established RL optimizer
    # semantics; nested ownership alone fixes the whole-model all-gather peak.
    return CustomPolicy(lambda module: module in targets), dict(sorted(counts.items()))


def wrap_qwen_fsdp(
    model: nn.Module,
    *,
    device: torch.device,
) -> tuple[FSDP, dict[str, object]]:
    """Build a nested FULL_SHARD hierarchy and return auditable metadata."""

    if device.type != "cuda":
        raise ValueError("direct-Qwen RL FSDP requires a CUDA device")
    policy, target_counts = qwen_block_wrap_policy(model)
    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=False,
    )
    handle_count = sum(isinstance(module, FSDP) for module in wrapped.modules())
    if handle_count <= 1:
        raise RuntimeError("direct-Qwen RL FSDP produced no nested shard handles")
    return wrapped, {
        "strategy_id": STRATEGY_ID,
        "target_counts": target_counts,
        "fsdp_handle_count": handle_count,
        "limit_all_gathers": True,
        "parameter_dtype": "preserved",
    }


__all__ = ["STRATEGY_ID", "qwen_block_wrap_policy", "wrap_qwen_fsdp"]
