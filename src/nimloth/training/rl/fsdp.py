"""FSDP policies shared by the online actor, token critic, and memory gates."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch


FSDP_WRAP_PROTOCOL = "qwen25vl-decoder-and-vision-block-v1"
ACTIVATION_CHECKPOINT_PROTOCOL = "qwen25vl-external-nonreentrant-block-wrapper-v1"


def _qwen_block_classes() -> set[type[torch.nn.Module]]:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VLVisionBlock,
    )

    return {Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock}


def qwen25vl_transformer_auto_wrap_policy() -> Callable:
    """Wrap every Qwen text/vision transformer block as its own FSDP unit.

    Root-only FSDP reshards the complete model before non-reentrant activation
    checkpointing replays individual layers, which changes parameter metadata.
    Layer-wise units keep each replayed block's full parameters available for
    exactly that block's forward/backward lifecycle.
    """

    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=_qwen_block_classes(),
    )


def count_activation_checkpoint_units(module: torch.nn.Module) -> int:
    """Count PyTorch checkpoint wrappers without relying on private names."""

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )

    return sum(isinstance(child, CheckpointWrapper) for child in module.modules())


def apply_qwen_activation_checkpointing(module: torch.nn.Module) -> int:
    """Checkpoint complete Qwen blocks outside their future FSDP units.

    Transformers 4.55 checkpoints ``super().__call__`` from inside each Qwen
    block. That recomputation bypasses the wrapper-level boundary expected by
    FSDP. We first disable that internal mechanism, then replace each raw block
    with ``CheckpointWrapper(block)``. Applying FSDP auto-wrap afterwards wraps
    the raw block nested inside the checkpoint wrapper, so recomputation calls
    the nested FSDP unit again, matching VAGEN's Transformers 4.49 boundary.
    """

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    if count_activation_checkpoint_units(module):
        raise RuntimeError("Qwen activation checkpoint wrappers already applied")
    disable = getattr(module, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    internal_enabled = [
        child.__class__.__name__
        for child in module.modules()
        if getattr(child, "gradient_checkpointing", False)
    ]
    if internal_enabled:
        raise RuntimeError(
            "failed to disable Hugging Face internal gradient checkpointing: "
            + ", ".join(internal_enabled[:4])
        )

    wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=True,
    )
    block_classes = _qwen_block_classes()
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda child: type(child) in block_classes,
    )
    count = count_activation_checkpoint_units(module)
    if count <= 0:
        raise RuntimeError("Qwen activation-checkpoint policy wrapped no blocks")
    return count


def count_fsdp_units(module: torch.nn.Module) -> int:
    """Return the number of root+nested FSDP units for runtime gates."""

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return sum(isinstance(child, FSDP) for child in module.modules())
