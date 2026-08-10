"""Tiny importable model factory used only by the real Ray/FSDP mechanics gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from nimloth.training.rl.planner_verl_worker import (
    PlannerObjectiveModule,
    PlannerWorkerModelComponents,
)


class GateTransition:
    """Pickle-safe row identity used to verify rank-local non-tensor dispatch."""

    def __init__(self, identity: str) -> None:
        self.identity = identity


class GateAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)


class GateAlgorithm:
    sigreg = None

    def __init__(self, *, rank: int) -> None:
        self.rank = rank

    def planner_transition_batch_step(
        self,
        runtime: Any,
        transitions: tuple[GateTransition, ...],
        **kwargs: Any,
    ) -> SimpleNamespace:
        expected_identity = f"rank-{self.rank}"
        if any(
            not isinstance(transition, GateTransition)
            or transition.identity != expected_identity
            for transition in transitions
        ):
            raise RuntimeError(
                "Ray DP_COMPUTE misrouted non-tensor transition identity: "
                f"rank={self.rank}"
            )
        device = runtime.agent.projection.weight.device
        computation_dtype = torch.bfloat16
        targets = torch.stack(kwargs["return_targets"]).to(
            device=device,
            dtype=computation_dtype,
        )
        inputs = torch.ones(
            (len(transitions), 1),
            device=device,
            dtype=computation_dtype,
        )
        predictions = runtime.agent.projection(inputs).reshape(-1)
        weights = torch.tensor(
            kwargs["loss_weights"],
            dtype=predictions.dtype,
            device=device,
        )
        loss = (
            (predictions - targets).square() * weights
        ).sum() / int(kwargs["total_transitions"])
        return SimpleNamespace(
            loss=loss,
            metrics={
                "gate_loss": float(loss.detach().cpu().item()),
                "gate_transition_identity": 1.0,
            },
        )


def build_tiny_gate_components(
    *,
    config: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
) -> PlannerWorkerModelComponents:
    del config, world_size
    torch.manual_seed(20260809)
    agent = GateAgent().to(device)
    objective = PlannerObjectiveModule(
        agent=agent,  # type: ignore[arg-type]
        input_builder=object(),  # type: ignore[arg-type]
        algorithm=GateAlgorithm(rank=rank),  # type: ignore[arg-type]
        max_state_tokens=None,
    )

    def optimizer_factory(root: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            root.parameters(),
            lr=0.05,
            weight_decay=0.0,
        )

    return PlannerWorkerModelComponents(
        objective_module=objective,
        optimizer_factory=optimizer_factory,
        wrap_policy={"min_num_params": 1},
        max_grad_norm=1.0,
    )


__all__ = [
    "GateAgent",
    "GateAlgorithm",
    "GateTransition",
    "build_tiny_gate_components",
]
