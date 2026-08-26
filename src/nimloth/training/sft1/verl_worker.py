"""Worker-local SFT1-v2 update lifecycle over one complete FSDP root."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from nimloth.backbone.base import BackboneInputBuilder
from nimloth.training.sft1.objective import (
    OBSERVED_MOVEMENT_ACTION_INDICES,
    SFT1V2Normalization,
)
from nimloth.training.sft1.verl_adapter import (
    sft1_v2_micro_batches,
    sft1_v2_update_inputs,
)
from nimloth.training.verl.runtime import (
    MixedPrecisionConfig,
    adamw_factory,
    assemble_training_root,
    clip_complete_fsdp_grad_norm_,
    wrap_complete_fsdp,
)
from nimloth.training.verl.source import verify_pinned_vagen_verl_source


@dataclass(frozen=True)
class SFT1V2WorkerAssembly:
    core: "SFT1V2UpdateCore"
    root: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any | None


@dataclass(frozen=True)
class SFT1V2UpdateResult:
    metrics: dict[str, float]
    gradient_norm: float
    micro_batch_count: int


def _distributed_world_size(device: torch.device) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _global_sum(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def _global_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def _padding_micro_batch(data: Any) -> Any:
    if len(data) < 1:
        raise ValueError("worker requires at least one real or explicit padding row")
    padding = data[torch.tensor([0], dtype=torch.long)]
    padding.batch["row_valid"] = torch.zeros_like(padding.batch["row_valid"])
    padding.batch["feasibility_label_valid"] = torch.zeros_like(
        padding.batch["feasibility_label_valid"]
    )
    return padding


def build_sft1_v2_fsdp_worker(
    *,
    objective_root: nn.Module,
    input_builder: BackboneInputBuilder,
    device: torch.device,
    repo_root: Path,
    wrap_policy: Mapping[str, Any],
    mixed_precision: MixedPrecisionConfig,
    learning_rate: float,
    weight_decay: float,
    adam_betas: tuple[float, float],
    adam_epsilon: float,
    max_padded_tokens: int,
    max_rows: int,
    max_grad_norm: float,
    scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
) -> SFT1V2WorkerAssembly:
    """Build the production SFT worker through one complete official FSDP root."""

    verify_pinned_vagen_verl_source(repo_root)
    contract = getattr(objective_root, "assert_trainable_contract", None)
    if not callable(contract):
        raise TypeError("SFT1-v2 objective root must expose its trainable contract")
    contract()
    assembly = assemble_training_root(
        objective_root,
        device=device,
        wrap=lambda module: wrap_complete_fsdp(
            module,
            device=device,
            wrap_policy=wrap_policy,
            mixed_precision=mixed_precision,
            repo_root=repo_root,
        ),
        optimizer_factory=adamw_factory(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=adam_betas,
            epsilon=adam_epsilon,
        ),
    )
    scheduler = (
        scheduler_factory(assembly.optimizer)
        if scheduler_factory is not None
        else None
    )
    core = SFT1V2UpdateCore(
        root=assembly.root,
        optimizer=assembly.optimizer,
        input_builder=input_builder,
        device=device,
        max_padded_tokens=max_padded_tokens,
        max_rows=max_rows,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
    )
    return SFT1V2WorkerAssembly(
        core=core,
        root=assembly.root,
        optimizer=assembly.optimizer,
        scheduler=scheduler,
    )


class SFT1V2UpdateCore:
    """One optimizer transaction with equal FSDP forward/backward order on every rank."""

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
            raise ValueError("worker token/row budgets must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("worker max_grad_norm must be positive")

    def update(self, data: Any) -> SFT1V2UpdateResult:
        """Run one complete update; count reductions never synchronize gradients manually."""

        if len(data) < 1:
            raise ValueError("SFT1-v2 worker batch must not be empty")
        row_valid = data.batch["row_valid"].bool()
        feasibility_valid = data.batch["feasibility_label_valid"].bool()
        actions = data.batch["executed_action_indices"].long()
        movement = torch.zeros_like(feasibility_valid)
        for action in OBSERVED_MOVEMENT_ACTION_INDICES:
            movement |= actions == action
        local_sample_count = row_valid.sum().to(device=self.device, dtype=torch.long)
        local_feasibility_count = (
            row_valid & feasibility_valid & movement
        ).sum().to(device=self.device, dtype=torch.long)
        global_sample_count = _global_sum(local_sample_count)
        global_feasibility_count = _global_sum(local_feasibility_count)
        if int(global_sample_count.item()) < 1:
            raise ValueError("SFT1-v2 update contains no globally valid sample")
        if int(global_feasibility_count.item()) < 1:
            raise ValueError("SFT1-v2 update contains no globally valid movement label")
        world_size = _distributed_world_size(self.device)
        normalization = SFT1V2Normalization(
            global_sample_valid_count=int(global_sample_count.item()),
            global_feasibility_valid_count=int(global_feasibility_count.item()),
            gradient_average_world_size=world_size,
        )

        micro_batches = list(
            sft1_v2_micro_batches(
                data,
                max_padded_tokens=self.max_padded_tokens,
                max_rows=self.max_rows,
            )
        )
        global_micro_count = _global_max_int(len(micro_batches), self.device)
        while len(micro_batches) < global_micro_count:
            micro_batches.append(_padding_micro_batch(data))
        if not micro_batches:
            raise RuntimeError("SFT1-v2 worker produced no micro-batch")

        self.optimizer.zero_grad(set_to_none=True)
        local_sums: dict[str, torch.Tensor] = {}
        local_counts: dict[str, torch.Tensor] = {}
        for index, micro_batch in enumerate(micro_batches):
            inputs = sft1_v2_update_inputs(
                micro_batch,
                input_builder=self.input_builder,
            )
            no_sync = getattr(self.root, "no_sync", None)
            context = (
                no_sync()
                if index + 1 < len(micro_batches) and callable(no_sync)
                else contextlib.nullcontext()
            )
            with context:
                output = self.root(
                    inputs.student_batch,
                    inputs.targets,
                    normalization,
                )
                if not torch.isfinite(output.total_loss):
                    raise RuntimeError("SFT1-v2 total loss is non-finite")
                output.total_loss.backward()
            for name, value in output.loss_sums.items():
                local_sums[name] = local_sums.get(name, torch.zeros_like(value)) + value.detach()
            for name, value in output.local_valid_counts.items():
                local_counts[name] = local_counts.get(
                    name,
                    torch.zeros_like(value),
                ) + value.detach()

        gradient_norm = clip_complete_fsdp_grad_norm_(self.root, self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics: dict[str, float] = {}
        for name in sorted(local_sums):
            total_sum = _global_sum(local_sums[name].to(self.device))
            total_count = _global_sum(local_counts[name].to(self.device))
            count = int(total_count.item())
            metrics[f"loss/{name}"] = (
                float(total_sum.item()) / count if count > 0 else 0.0
            )
            metrics[f"count/{name}"] = float(count)
        return SFT1V2UpdateResult(
            metrics=metrics,
            gradient_norm=float(gradient_norm.detach().item()),
            micro_batch_count=global_micro_count,
        )


__all__ = [
    "SFT1V2UpdateCore",
    "SFT1V2UpdateResult",
    "SFT1V2WorkerAssembly",
    "build_sft1_v2_fsdp_worker",
]
