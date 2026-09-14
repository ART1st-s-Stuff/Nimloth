"""Stage3 Qwen full sharding, retaining the replicated world-model branch.

Frozen BF16 vocabulary tables are intentionally replicated. Trainable FP32
masters, including selected token rows and the full visual encoder, are sharded.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import CustomPolicy


STRATEGY_ID = "qwen_full_shard_orig_params_fp32_master_bf16_forward_v1"


def is_fsdp(model: nn.Module) -> bool:
    return isinstance(model, FSDP)


def qwen_wrap_policy(model: nn.Module):
    """Return the exact handle policy and replicated frozen parameters."""
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Stage3 FSDP requires Qwen's complete visual module")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise ValueError("Stage3 FSDP requires all trainable masters to be FP32")
    ignored = {parameter for parameter in model.parameters() if not parameter.requires_grad}
    # Ignored frozen tables may share a leaf with FP32 selected rows. Excluding
    # them from flattening avoids mixed-dtype handles without rounding masters.
    if any(parameter.dtype not in (torch.bfloat16, torch.float32) for parameter in ignored):
        raise ValueError("unsupported frozen parameter dtype")
    owners = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad:
                owners[id(parameter)] = owners.get(id(parameter), 0) + 1
    if any(count > 1 for count in owners.values()):
        raise ValueError("shared trainable parameters need an explicit FSDP ownership policy")
    block_names = {"Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"}
    block_names.update(getattr(model, "_no_split_modules", ()) or ())
    targets = {module for module in model.modules()
               if module.__class__.__name__ in block_names or isinstance(module, (nn.Linear, nn.Embedding))}
    targets.add(visual)
    visual_modules = set(visual.modules())
    common = dict(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                  buffer_dtype=None, keep_low_precision_grads=False)
    language_precision = MixedPrecision(**common, cast_forward_inputs=True)
    # Qwen deliberately calculates rotary cos/sin in FP32 inside vision. Do not
    # cast these locally generated arguments merely because a child is wrapped.
    visual_precision = MixedPrecision(**common, cast_forward_inputs=False, cast_root_forward_inputs=False)
    def policy(module):
        if module not in targets:
            return False
        return {"mixed_precision": visual_precision if module in visual_modules else language_precision}
    return CustomPolicy(policy), ignored, language_precision


def wrap_qwen_fsdp(model: nn.Module, device: torch.device) -> FSDP:
    if device.type != "cuda" or not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("Stage3 FSDP requires initialized multi-rank CUDA")
    policy, ignored, precision = qwen_wrap_policy(model)
    wrapped = FSDP(model, auto_wrap_policy=policy, ignored_states=ignored,
                mixed_precision=precision, sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True, device_id=device, sync_module_states=True,
                limit_all_gathers=True)
    if not isinstance(wrapped.module.visual, FSDP):
        raise RuntimeError("vision must have a nested FSDP owner for shard-local EMA")
    return wrapped


@torch.no_grad()
def clip_mixed_grad_norm(qwen: nn.Module, replicated_modules, max_norm: float) -> torch.Tensor:
    """Global Qwen-shard norm plus one copy of already DDP-averaged WM grads."""
    if not is_fsdp(qwen):
        raise TypeError("mixed gradient clipping requires a Qwen FSDP root")
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    sharded = [p for p in qwen.parameters() if p.requires_grad and p.grad is not None]
    reference = next(qwen.parameters())
    squared = torch.zeros((), device=reference.device, dtype=torch.float32)
    for parameter in sharded:
        squared.add_(parameter.grad.float().square().sum())
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    seen = {id(p) for p in qwen.parameters()}
    replicated = []
    for module in replicated_modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.requires_grad and parameter.grad is not None:
                replicated.append(parameter)
                squared.add_(parameter.grad.float().square().sum())
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite mixed FSDP/DDP gradient norm")
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
    for parameter in (*sharded, *replicated):
        parameter.grad.mul_(scale.to(parameter.grad.dtype))
    return norm
