"""Optional CPU storage for tensors saved by Stage3 forward autograd graphs."""
from __future__ import annotations

from contextlib import nullcontext

import torch


def saved_activation_context(enabled: bool):
    """Preserve tensor values/dtypes and restore their device during backward.

    Parameters, optimizer state, computation and gradient reduction stay on their
    original devices. PyTorch may save parameter views as well as activations;
    their saved copies use CPU storage without moving the live parameters.
    """
    if enabled:
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    return nullcontext()
