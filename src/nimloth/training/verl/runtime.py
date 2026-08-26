"""Complete-root device, FSDP, optimizer, and clipping assembly."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nimloth.training.verl.source import (
    require_pinned_verl_import,
    verify_pinned_vagen_verl_source,
)


@dataclass(frozen=True)
class MixedPrecisionConfig:
    param_dtype: torch.dtype
    reduce_dtype: torch.dtype
    buffer_dtype: torch.dtype


@dataclass(frozen=True)
class TrainingAssembly:
    root: nn.Module
    optimizer: torch.optim.Optimizer


def assert_complete_module_device(module: nn.Module, device: torch.device) -> None:
    """Require every parameter and buffer on the selected rank device."""

    mismatches = []
    expected_index = device.index
    for kind, values in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in values:
            actual = value.device
            same_type = actual.type == device.type
            same_index = expected_index is None or actual.index == expected_index
            if not same_type or not same_index:
                mismatches.append(f"{kind} {name}: {actual} != {device}")
    if mismatches:
        raise RuntimeError("complete module device mismatch: " + "; ".join(mismatches))


def assemble_training_root(
    module: nn.Module,
    *,
    device: torch.device,
    wrap: Callable[[nn.Module], nn.Module],
    optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
) -> TrainingAssembly:
    """Move the complete module, wrap once, then construct its optimizer."""

    module.train()
    module.to(device)
    assert_complete_module_device(module, device)
    wrapped = wrap(module)
    optimizer = optimizer_factory(wrapped.parameters())
    return TrainingAssembly(root=wrapped, optimizer=optimizer)


def wrap_complete_fsdp(
    module: nn.Module,
    *,
    device: torch.device,
    wrap_policy: Mapping[str, Any],
    mixed_precision: MixedPrecisionConfig,
    repo_root: Path,
) -> nn.Module:
    """Wrap one complete objective root with official FULL_SHARD FSDP."""

    verify_pinned_vagen_verl_source(repo_root)
    require_pinned_verl_import(repo_root)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("complete-root FSDP requires an available CUDA rank device")
    if not torch.distributed.is_initialized():
        raise RuntimeError("complete-root FSDP requires an initialized process group")
    if not wrap_policy or bool(wrap_policy.get("disable", False)):
        raise ValueError("complete-root FSDP requires an explicit enabled wrap policy")

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.nn.parallel import DistributedDataParallel as DDP
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    for child in module.modules():
        if child is module:
            continue
        if isinstance(child, (FSDP, DDP)):
            raise ValueError("complete objective must be unwrapped before FSDP assembly")
    policy = get_fsdp_wrap_policy(module=module, config=dict(wrap_policy))
    return FSDP(
        module,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=policy,
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision.param_dtype,
            reduce_dtype=mixed_precision.reduce_dtype,
            buffer_dtype=mixed_precision.buffer_dtype,
        ),
        forward_prefetch=False,
    )


def adamw_factory(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    epsilon: float,
) -> Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]:
    if learning_rate <= 0.0 or weight_decay < 0.0 or epsilon <= 0.0:
        raise ValueError("AdamW learning rate/weight decay/epsilon are invalid")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError("AdamW betas must lie in [0,1)")

    def build(parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise ValueError("complete objective has no trainable parameters")
        return torch.optim.AdamW(
            trainable,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=betas,
            eps=float(epsilon),
        )

    return build


def clip_complete_fsdp_grad_norm_(root: nn.Module, max_norm: float) -> torch.Tensor:
    """Use the framework root's global clipping implementation; never all-reduce manually."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    clip = getattr(root, "clip_grad_norm_", None)
    if not callable(clip):
        raise TypeError("wrapped FSDP root must provide clip_grad_norm_()")
    value = clip(float(max_norm))
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value):
        raise RuntimeError("FSDP gradient norm is invalid")
    return value


__all__ = [
    "MixedPrecisionConfig",
    "TrainingAssembly",
    "adamw_factory",
    "assemble_training_root",
    "assert_complete_module_device",
    "clip_complete_fsdp_grad_norm_",
    "wrap_complete_fsdp",
]
