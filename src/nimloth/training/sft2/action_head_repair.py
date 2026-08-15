"""Low-cost repair of ID74's eight frozen action-token LM-head rows.

The Qwen transformer, world model, projector, and ValueHead remain frozen.  A
small FP32 delta is trained against the final hidden state at ``action_start``
and can then be merged into only the configured rows of the standard LM head.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def balanced_action_sample_indices(
    samples: Sequence[Any],
    *,
    action_count: int,
    examples_per_action: int,
    seed: int,
) -> tuple[int, ...]:
    """Select an order-independent, deterministic equal count for every action."""

    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 1:
        raise ValueError("action_count must be a positive int")
    if (
        isinstance(examples_per_action, bool)
        or not isinstance(examples_per_action, int)
        or examples_per_action < 1
    ):
        raise ValueError("examples_per_action must be a positive int")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    grouped: list[list[tuple[bytes, str, int, int]]] = [
        [] for _ in range(action_count)
    ]
    identities: set[tuple[str, int]] = set()
    for index, sample in enumerate(samples):
        record_id = getattr(sample, "record_id", None)
        step_index = getattr(sample, "step_index", None)
        action_index = getattr(sample, "action_index", None)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("action sample record_id must be a non-empty string")
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index < 0
        ):
            raise ValueError("action sample step_index must be a non-negative int")
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 0 <= action_index < action_count
        ):
            raise ValueError("action sample action_index is outside the action table")
        identity = (record_id, step_index)
        if identity in identities:
            raise ValueError(f"duplicate action sample identity: {identity!r}")
        identities.add(identity)
        payload = (
            f"nimloth-action-head-repair-v1\0{seed}\0{action_index}\0"
            f"{record_id}\0{step_index}"
        ).encode("utf-8")
        grouped[action_index].append(
            (hashlib.sha256(payload).digest(), record_id, step_index, index)
        )
    selected: list[int] = []
    for action_index, candidates in enumerate(grouped):
        if len(candidates) < examples_per_action:
            raise ValueError(
                f"action {action_index} has {len(candidates)} examples; "
                f"requires {examples_per_action}"
            )
        candidates.sort()
        selected.extend(row[-1] for row in candidates[:examples_per_action])
    return tuple(selected)


class ActionTokenRowDelta(nn.Module):
    """FP32 trainable delta over frozen action-token LM-head rows only."""

    def __init__(self, base_action_rows: torch.Tensor) -> None:
        super().__init__()
        if (
            not isinstance(base_action_rows, torch.Tensor)
            or base_action_rows.ndim != 2
            or min(base_action_rows.shape) < 1
            or not torch.isfinite(base_action_rows).all()
        ):
            raise ValueError("base action rows must be a finite rank-2 tensor")
        base = base_action_rows.detach().to(dtype=torch.float32).contiguous()
        self.register_buffer("base_action_rows", base)
        self.delta = nn.Parameter(torch.zeros_like(base))

    @property
    def action_count(self) -> int:
        return int(self.base_action_rows.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.base_action_rows.shape[1])

    def forward(self, action_boundary_hidden: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(action_boundary_hidden, torch.Tensor)
            or action_boundary_hidden.ndim != 2
            or action_boundary_hidden.shape[1] != self.hidden_dim
            or not torch.isfinite(action_boundary_hidden).all()
        ):
            raise ValueError(
                "action boundary hidden must be finite with shape (B, hidden_dim)"
            )
        rows = self.base_action_rows + self.delta
        return F.linear(action_boundary_hidden.detach().to(dtype=torch.float32), rows)

    def merged_rows(self) -> torch.Tensor:
        rows = self.base_action_rows + self.delta
        if not torch.isfinite(rows).all():
            raise RuntimeError("repaired action rows are non-finite")
        return rows


@dataclass(frozen=True)
class ActionHeadRepairFit:
    """Best held-out result of one deterministic action-row repair fit."""

    delta: torch.Tensor
    best_epoch: int
    epochs_run: int
    training_nll_after: float
    validation_nll_before: float
    validation_nll_after: float
    validation_logits_after: torch.Tensor


def _validated_balanced_features(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    action_count: int,
    field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 2
        or hidden.shape[0] < action_count
        or hidden.shape[1] < 1
        or not torch.isfinite(hidden).all()
    ):
        raise ValueError(f"{field} hidden must be a finite rank-2 tensor")
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dtype != torch.long
        or tuple(targets.shape) != (hidden.shape[0],)
        or torch.any(targets < 0)
        or torch.any(targets >= action_count)
    ):
        raise ValueError(f"{field} targets must be valid torch.long action IDs")
    counts = torch.bincount(targets.cpu(), minlength=action_count)
    if torch.any(counts < 1) or not torch.equal(counts, counts[0].expand_as(counts)):
        raise ValueError(f"{field} targets must have equal per-action counts")
    return hidden.detach().to(dtype=torch.float32), targets.detach()


def fit_action_token_row_delta(
    *,
    base_action_rows: torch.Tensor,
    train_hidden: torch.Tensor,
    train_targets: torch.Tensor,
    validation_hidden: torch.Tensor,
    validation_targets: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    early_stopping_patience: int,
    minimum_validation_improvement: float,
    device: torch.device,
) -> ActionHeadRepairFit:
    """Fit only the eight FP32 row deltas using balanced restricted CE."""

    module = ActionTokenRowDelta(base_action_rows).to(device)
    action_count = module.action_count
    train_x, train_y = _validated_balanced_features(
        train_hidden,
        train_targets,
        action_count=action_count,
        field="training",
    )
    val_x, val_y = _validated_balanced_features(
        validation_hidden,
        validation_targets,
        action_count=action_count,
        field="validation",
    )
    if train_x.shape[1] != module.hidden_dim or val_x.shape[1] != module.hidden_dim:
        raise ValueError("repair hidden dimension does not match LM-head rows")
    for field, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
        ("minimum_validation_improvement", minimum_validation_improvement),
    ):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{field} must be a finite non-negative real")
        normalized = float(value)
        if not torch.isfinite(torch.tensor(normalized)) or normalized < 0.0:
            raise ValueError(f"{field} must be a finite non-negative real")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    for field, value in (
        ("max_epochs", max_epochs),
        ("early_stopping_patience", early_stopping_patience),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive int")

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    with torch.no_grad():
        validation_nll_before = float(
            restricted_action_cross_entropy(module(val_x), val_y).item()
        )
    optimizer = torch.optim.AdamW(
        [module.delta],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_delta = module.delta.detach().clone()
    best_epoch = 0
    best_validation = validation_nll_before
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = restricted_action_cross_entropy(module(train_x), train_y)
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation = float(
                restricted_action_cross_entropy(module(val_x), val_y).item()
            )
        epochs_run = epoch
        if validation < best_validation - 1e-12:
            best_validation = validation
            best_delta = module.delta.detach().clone()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
    with torch.no_grad():
        module.delta.copy_(best_delta)
        training_nll_after = float(
            restricted_action_cross_entropy(module(train_x), train_y).item()
        )
        validation_logits_after = module(val_x).cpu()
        validation_nll_after = float(
            restricted_action_cross_entropy(
                validation_logits_after,
                val_y.cpu(),
            ).item()
        )
    if (
        validation_nll_before - validation_nll_after
        < float(minimum_validation_improvement)
    ):
        raise RuntimeError(
            "action-head repair did not meet minimum validation NLL improvement: "
            f"before={validation_nll_before}, after={validation_nll_after}, "
            f"required={minimum_validation_improvement}"
        )
    return ActionHeadRepairFit(
        delta=best_delta.cpu(),
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        training_nll_after=training_nll_after,
        validation_nll_before=validation_nll_before,
        validation_nll_after=validation_nll_after,
        validation_logits_after=validation_logits_after,
    )


def restricted_action_cross_entropy(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
) -> torch.Tensor:
    """Compute action-only CE on the exact restricted action table."""

    if (
        not isinstance(action_logits, torch.Tensor)
        or action_logits.ndim != 2
        or min(action_logits.shape) < 1
        or not torch.isfinite(action_logits).all()
    ):
        raise ValueError("restricted action logits must be finite with shape (B, A)")
    if (
        not isinstance(action_targets, torch.Tensor)
        or action_targets.dtype != torch.long
        or tuple(action_targets.shape) != (action_logits.shape[0],)
        or torch.any(action_targets < 0)
        or torch.any(action_targets >= action_logits.shape[1])
    ):
        raise ValueError("restricted action targets must be valid torch.long IDs")
    loss = F.cross_entropy(action_logits, action_targets)
    if not torch.isfinite(loss):
        raise RuntimeError("restricted action cross entropy is non-finite")
    return loss


def apply_action_row_delta_(
    lm_head_weight: torch.Tensor,
    *,
    action_token_ids: Sequence[int],
    delta: torch.Tensor,
) -> torch.Tensor:
    """Merge an FP32 delta into only selected rows of an existing LM head."""

    if (
        not isinstance(lm_head_weight, torch.Tensor)
        or lm_head_weight.ndim != 2
        or min(lm_head_weight.shape) < 1
        or not torch.isfinite(lm_head_weight).all()
    ):
        raise ValueError("LM-head weight must be a finite rank-2 tensor")
    token_ids = tuple(action_token_ids)
    if not token_ids or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < lm_head_weight.shape[0]
        for value in token_ids
    ):
        raise ValueError("action token IDs must be valid non-negative ints")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("action token IDs must be unique")
    if (
        not isinstance(delta, torch.Tensor)
        or tuple(delta.shape) != (len(token_ids), lm_head_weight.shape[1])
        or not torch.isfinite(delta).all()
    ):
        raise ValueError("action row delta must be finite and align with selected rows")
    indices = torch.tensor(token_ids, dtype=torch.long, device=lm_head_weight.device)
    with torch.no_grad():
        base = lm_head_weight.index_select(0, indices).to(dtype=torch.float32)
        merged = (base + delta.detach().to(device=base.device, dtype=torch.float32)).to(
            dtype=lm_head_weight.dtype
        )
        lm_head_weight.index_copy_(0, indices, merged)
    return lm_head_weight


def population_action_spread(action_logits: torch.Tensor) -> torch.Tensor:
    """Return finite per-row population standard deviation for audit gates."""

    if (
        not isinstance(action_logits, torch.Tensor)
        or action_logits.ndim != 2
        or action_logits.shape[1] < 2
        or not torch.isfinite(action_logits).all()
    ):
        raise ValueError("action spread requires finite logits with shape (B, A>=2)")
    spread = action_logits.to(dtype=torch.float32).std(dim=-1, correction=0)
    if not torch.isfinite(spread).all():
        raise RuntimeError("action spread is non-finite")
    return spread


__all__ = [
    "ActionHeadRepairFit",
    "ActionTokenRowDelta",
    "apply_action_row_delta_",
    "balanced_action_sample_indices",
    "fit_action_token_row_delta",
    "population_action_spread",
    "restricted_action_cross_entropy",
]
