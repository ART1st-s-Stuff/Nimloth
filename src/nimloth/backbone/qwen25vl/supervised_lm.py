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
def window_lm_projection(
    head: nn.Module, labels: torch.Tensor, weights: torch.Tensor, *,
    source_rows: torch.Tensor | None = None, return_rows: bool = False,
):
    """Replace one ordinary head result with the exact scalar window LM loss."""
    if labels.ndim != 2 or weights.shape != (labels.shape[0],):
        raise ValueError("weighted LM labels and row weights have incompatible shapes")
    if not torch.all((weights == 0) | (weights == 1)):
        raise ValueError("LM row weights must be zero or one for each input row")
    if source_rows is not None:
        if source_rows.dtype != torch.long or source_rows.shape != (labels.shape[0],):
            raise ValueError("LM source rows must be an integer index for each window")
    positions = [(row[1:] != -100).nonzero(as_tuple=True)[0] for row in labels]
    if any(position.numel() == 0 for position in positions):
        raise ValueError("LM window has no supervised answer tokens")
    # Resolve the binary mask once, before any full-vocabulary CE projection.
    # All rows still pass label validation, including failure/padding rows.
    active_rows = weights.nonzero(as_tuple=True)[0].tolist()
    leaf = _linear_leaf(head)
    captured = {}

    def trim_projection(_module, args):
        captured["hidden"] = args[0]
        return (args[0][:, -1:, :], *args[1:])

    def compute_loss(module, _inputs, _output):
        hidden = captured.pop("hidden")
        if hidden.shape[1] != labels.shape[1]:
            raise ValueError("weighted LM hidden and label sequence shapes disagree")
        sources = (list(range(labels.shape[0])) if source_rows is None
                   else source_rows.tolist())
        if (source_rows is None and hidden.shape[0] != labels.shape[0]) or any(
                row < 0 or row >= hidden.shape[0] for row in sources):
            raise ValueError("LM source row is outside the encoded batch")
        if not active_rows:
            # Keep the ordinary one-token head output connected. This gives
            # dense/selected head rows and hidden states explicit zero gradients
            # and retains the FSDP wrapper backward hook on ranks with no LM.
            # No CE or checkpointed projection is needed on these ranks.
            zero = _output.float().sum() * 0.0
            return zero.expand(labels.shape[0]) if return_rows else zero
        row_losses = {}
        # The closure captures the module, not any transient parameter tensors.
        def token_loss(chunk_hidden, chunk_targets):
            return _token_loss(module, chunk_hidden, chunk_targets)
        for row in active_rows:
            position = positions[row]
            total = hidden.new_zeros((), dtype=torch.float32)
            for chunk in position.split(128):
                chunk_hidden = hidden[sources[row], chunk]
                targets = labels[row, chunk + 1]
                total = total + (
                    checkpoint(token_loss, chunk_hidden, targets, use_reentrant=False)
                    if torch.is_grad_enabled() else token_loss(chunk_hidden, targets)
                )
            row_losses[row] = total / position.numel()
        if return_rows:
            zero = _output.float().sum() * 0.0
            return torch.stack([row_losses.get(row, zero) for row in range(labels.shape[0])])
        return torch.stack(list(row_losses.values())).sum() / weights.sum().clamp_min(1)

    pre = leaf.register_forward_pre_hook(trim_projection)
    post = leaf.register_forward_hook(compute_loss)
    try:
        yield
    finally:
        pre.remove()
        post.remove()
        captured.clear()
