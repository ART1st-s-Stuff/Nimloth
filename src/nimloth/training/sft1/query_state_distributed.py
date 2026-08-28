"""Complete-root distributed update transaction for Query-State SFT1.

Only detached count/metric collectives are implemented here.  Parameter
synchronization, gradient averaging, and global clipping remain owned by the
official complete-root FSDP wrapper.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nimloth.backbone.base import BackboneInputBuilder
from nimloth.training.sft1.query_state import QueryStateNormalization
from nimloth.training.sft1.query_state_adapter import query_state_update_inputs
from nimloth.training.sft1.query_state_runtime import QueryStateWorkerAssembly
from nimloth.training.verl.runtime import clip_complete_fsdp_grad_norm_


@dataclass(frozen=True)
class QueryStateUpdateResult:
    metrics: dict[str, float]
    gradient_norm: float
    micro_batch_count: int


@dataclass(frozen=True)
class QueryStateDistributedWorkerAssembly:
    core: "QueryStateUpdateCore"
    root: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any | None


def _distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _global_sum_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(int(value), device=device, dtype=torch.long)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _global_sum_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    result = value.detach().to(device=device).clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def _global_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(int(value), device=device, dtype=torch.long)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def query_state_micro_batches(
    data: Any,
    *,
    max_padded_tokens: int,
    max_rows: int,
) -> tuple[Any, ...]:
    """Deterministic first-fit packing by exact padded-token cost."""

    if max_padded_tokens < 1 or max_rows < 1:
        raise ValueError("Query-State token and row budgets must be positive")
    counts = tuple(int(value) for value in data.batch["token_counts"].tolist())
    if not counts or any(value < 1 for value in counts):
        raise ValueError("Query-State token counts must be positive")
    oversized = [
        (index, value)
        for index, value in enumerate(counts)
        if value > max_padded_tokens
    ]
    if oversized:
        index, value = oversized[0]
        raise ValueError(
            f"Query-State row exceeds max_padded_tokens: row{index}={value}"
        )
    groups: list[list[int]] = []
    for index in sorted(range(len(counts)), key=lambda item: (-counts[item], item)):
        for group in groups:
            candidate = (*group, index)
            cost = max(counts[item] for item in candidate) * len(candidate)
            if len(candidate) <= max_rows and cost <= max_padded_tokens:
                group.append(index)
                break
        else:
            groups.append([index])
    return tuple(
        data[torch.tensor(group, dtype=torch.long)] for group in groups
    )


def _padding_micro_batch(data: Any) -> Any:
    if len(data) < 1:
        raise ValueError("Query-State update requires an explicit local schedule row")
    padding = data[torch.tensor([0], dtype=torch.long)]
    padding.batch["row_valid"] = torch.zeros_like(
        padding.batch["row_valid"], dtype=torch.bool
    )
    return padding


def _local_lm_valid_token_count(data: Any) -> int:
    row_valid = data.batch.get("row_valid")
    encoded_rows = data.non_tensor_batch.get("encoded_rows")
    if (
        not isinstance(row_valid, torch.Tensor)
        or row_valid.dtype != torch.bool
        or row_valid.shape != (len(data),)
        or encoded_rows is None
        or len(encoded_rows) != len(data)
    ):
        raise ValueError("Query-State update row-valid/encoded-row alignment is invalid")
    total = 0
    for valid, encoded in zip(row_valid.tolist(), encoded_rows, strict=True):
        if not isinstance(encoded, dict):
            raise ValueError("Query-State update encoded row must be a mapping")
        labels = encoded.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError("Query-State update encoded row has invalid labels")
        if valid:
            total += int((labels != -100).sum().item())
    return total


def build_query_state_distributed_worker(
    *,
    worker: QueryStateWorkerAssembly,
    input_builder: BackboneInputBuilder,
    device: torch.device,
    max_padded_tokens: int,
    max_rows: int,
    max_grad_norm: float,
    scheduler: Any | None = None,
) -> QueryStateDistributedWorkerAssembly:
    """Attach the update transaction to an already assembled complete root."""

    core = QueryStateUpdateCore(
        root=worker.root,
        optimizer=worker.optimizer,
        input_builder=input_builder,
        device=device,
        max_padded_tokens=max_padded_tokens,
        max_rows=max_rows,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
    )
    return QueryStateDistributedWorkerAssembly(
        core=core,
        root=worker.root,
        optimizer=worker.optimizer,
        scheduler=scheduler,
    )


class QueryStateUpdateCore:
    """One optimizer step with equal complete-root forward/backward ordering."""

    def __init__(
        self,
        *,
        root: nn.Module,
        optimizer: torch.optim.Optimizer,
        input_builder: BackboneInputBuilder,
        device: torch.device,
        max_padded_tokens: int,
        max_rows: int,
        max_grad_norm: float,
        scheduler: Any | None = None,
    ) -> None:
        self.root = root
        self.optimizer = optimizer
        self.input_builder = input_builder
        self.device = device
        self.max_padded_tokens = int(max_padded_tokens)
        self.max_rows = int(max_rows)
        self.max_grad_norm = float(max_grad_norm)
        self.scheduler = scheduler
        if self.max_padded_tokens < 1 or self.max_rows < 1:
            raise ValueError("Query-State update token/row budgets must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("Query-State max_grad_norm must be positive")

    def update(self, data: Any) -> QueryStateUpdateResult:
        """Compute denominators first, then run exactly one optimizer transaction."""

        if len(data) < 1:
            raise ValueError("Query-State worker batch must not be empty")
        row_valid = data.batch.get("row_valid")
        if (
            not isinstance(row_valid, torch.Tensor)
            or row_valid.dtype != torch.bool
            or row_valid.shape != (len(data),)
        ):
            raise ValueError("Query-State update row_valid is invalid")

        # Both denominators cover the entire rank update and are globally reduced
        # before any rank begins a micro-batch forward.
        local_state_count = int(row_valid.sum().item()) * 16 * 1024
        local_lm_count = _local_lm_valid_token_count(data)
        global_state_count = _global_sum_int(local_state_count, self.device)
        global_lm_count = _global_sum_int(local_lm_count, self.device)
        if global_state_count < 1:
            raise ValueError("Query-State update has no globally valid state element")
        if global_lm_count < 1:
            raise ValueError("Query-State update has no globally valid LM token")
        world_size = _distributed_world_size()
        normalization = QueryStateNormalization(
            global_state_valid_element_count=global_state_count,
            global_lm_valid_token_count=global_lm_count,
            gradient_average_world_size=world_size,
        )

        micro_batches = list(
            query_state_micro_batches(
                data,
                max_padded_tokens=self.max_padded_tokens,
                max_rows=self.max_rows,
            )
        )
        global_micro_count = _global_max_int(len(micro_batches), self.device)
        while len(micro_batches) < global_micro_count:
            micro_batches.append(_padding_micro_batch(data))
        if global_micro_count < 1:
            raise RuntimeError("Query-State update produced no global micro-batch")

        self.optimizer.zero_grad(set_to_none=True)
        local_sums: dict[str, torch.Tensor] = {}
        local_counts: dict[str, int] = {}
        for index, micro_batch in enumerate(micro_batches):
            inputs = query_state_update_inputs(
                micro_batch,
                input_builder=self.input_builder,
            )
            no_sync = getattr(self.root, "no_sync", None)
            sync_context = (
                no_sync()
                if index + 1 < global_micro_count and callable(no_sync)
                else contextlib.nullcontext()
            )
            with sync_context:
                output = self.root(
                    inputs.student_batch,
                    inputs.targets,
                    normalization,
                )
                if output.total_loss.ndim != 0 or not torch.isfinite(output.total_loss):
                    raise RuntimeError("Query-State update total loss is non-finite")
                output.total_loss.backward()
            for name, value in output.loss_sums.items():
                local_sums[name] = local_sums.get(
                    name, torch.zeros_like(value)
                ) + value.detach()
            for name, value in output.local_valid_counts.items():
                local_counts[name] = local_counts.get(name, 0) + int(value)

        # FSDP owns global gradient norm aggregation and clipping.  There is no
        # manual parameter-gradient collective in this module.
        gradient_norm = clip_complete_fsdp_grad_norm_(
            self.root, self.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics: dict[str, float] = {}
        for name in sorted(local_sums):
            total_sum = _global_sum_tensor(local_sums[name], self.device)
            total_count = _global_sum_int(local_counts[name], self.device)
            metrics[f"loss/{name}"] = (
                float(total_sum.item()) / total_count if total_count else 0.0
            )
            metrics[f"count/{name}"] = float(total_count)
        return QueryStateUpdateResult(
            metrics=metrics,
            gradient_norm=float(gradient_norm.detach().item()),
            micro_batch_count=global_micro_count,
        )


__all__ = [
    "QueryStateDistributedWorkerAssembly",
    "QueryStateUpdateCore",
    "QueryStateUpdateResult",
    "build_query_state_distributed_worker",
    "query_state_micro_batches",
]
