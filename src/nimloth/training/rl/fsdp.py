"""FSDP policies shared by the online actor, token critic, and memory gates."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch


FSDP_WRAP_PROTOCOL = "qwen25vl-decoder-and-vision-block-v1"


def qwen25vl_transformer_auto_wrap_policy() -> Callable:
    """Wrap every Qwen text/vision transformer block as its own FSDP unit.

    Root-only FSDP reshards the complete model before non-reentrant activation
    checkpointing replays individual layers, which changes parameter metadata.
    Layer-wise units keep each replayed block's full parameters available for
    exactly that block's forward/backward lifecycle.
    """

    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VLVisionBlock,
    )

    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock},
    )


def count_fsdp_units(module: torch.nn.Module) -> int:
    """Return the number of root+nested FSDP units for runtime gates."""

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return sum(isinstance(child, FSDP) for child in module.modules())
