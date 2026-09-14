"""Exact window-weighted LM loss with bounded vocabulary intermediates.

The temporary hook lives inside the FSDP output-head wrapper. Its scalar output
keeps that wrapper's backward unshard hook active before checkpoint recomputation
reads the head's current parameter views. Never call the wrapped head recursively.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _linear_leaf(head: nn.Module) -> nn.Linear:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    while True:
        if isinstance(head, FSDP):
            head = head.module
            continue
        copies = getattr(head, "modules_to_save", None)
        if copies is not None:
            active = getattr(head, "active_adapter", "default")
            if not isinstance(active, str) or active not in copies:
                raise ValueError("weighted LM requires one active saved output head")
            head = copies[active]
            continue
        break
    # Functional projection must exactly implement the actual head, never skip
    # an arbitrary subclass forward (e.g. a LoRA or quantized linear operator).
    if type(head) is not nn.Linear or head.bias is not None:
        raise TypeError("bounded weighted LM requires an unbiased nn.Linear output head")
    return head


def _token_loss(head: nn.Linear, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Fetch every parameter here, including during checkpoint recomputation;
    # capturing an old FSDP parameter view would break fresh backward unsharding.
    scores = F.linear(hidden, head.weight)
    if getattr(head, "_nimloth_selected_token_rows", False):
        for ids, rows in ((head.nimloth_query_ids, head.nimloth_query_rows),
                          (head.nimloth_protocol_ids, head.nimloth_protocol_rows)):
            selected = F.linear(hidden, rows.to(hidden.dtype))
            scores.index_copy_(-1, ids.to(scores.device), selected.to(scores.dtype))
    return F.cross_entropy(scores.float(), targets, reduction="sum")


@contextmanager
def window_lm_projection(head: nn.Module, labels: torch.Tensor, weights: torch.Tensor):
    """Replace one ordinary head result with the exact scalar window LM loss."""
    if labels.ndim != 2 or weights.shape != (labels.shape[0],):
        raise ValueError("weighted LM labels and row weights have incompatible shapes")
    if not torch.all((weights == 0) | (weights == 1)):
        raise ValueError("LM row weights must be zero or one for each input row")
    positions = [(row[1:] != -100).nonzero(as_tuple=True)[0] for row in labels]
    if any(position.numel() == 0 for position in positions):
        raise ValueError("LM window has no supervised answer tokens")
    leaf = _linear_leaf(head)
    captured = {}

    def trim_projection(_module, args):
        captured["hidden"] = args[0]
        return (args[0][:, -1:, :], *args[1:])

    def compute_loss(module, _inputs, _output):
        hidden = captured.pop("hidden")
        if hidden.shape[:2] != labels.shape:
            raise ValueError("weighted LM hidden and label sequence shapes disagree")
        row_losses = []
        # The closure captures the module, not any transient parameter tensors.
        def token_loss(chunk_hidden, chunk_targets):
            return _token_loss(module, chunk_hidden, chunk_targets)
        for row, position in enumerate(positions):
            total = hidden.new_zeros((), dtype=torch.float32)
            for chunk in position.split(128):
                chunk_hidden = hidden[row, chunk]
                targets = labels[row, chunk + 1]
                total = total + (
                    checkpoint(token_loss, chunk_hidden, targets, use_reentrant=False)
                    if torch.is_grad_enabled() else token_loss(chunk_hidden, targets)
                )
            row_losses.append(total / position.numel())
        return (torch.stack(row_losses) * weights).sum() / weights.sum().clamp_min(1)

    pre = leaf.register_forward_pre_hook(trim_projection)
    post = leaf.register_forward_hook(compute_loss)
    try:
        yield
    finally:
        pre.remove()
        post.remove()
        captured.clear()
