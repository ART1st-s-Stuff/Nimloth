"""Shared activation-storage context for training autograd graphs."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def saved_activation_context(enabled: bool):
    """Optionally save autograd tensors in pinned CPU memory.

    Live parameters, optimizer state, forward computation, and gradient
    reduction remain on their original devices. PyTorch restores each saved
    tensor to its source device when backward consumes it.
    """

    if enabled:
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    return nullcontext()
