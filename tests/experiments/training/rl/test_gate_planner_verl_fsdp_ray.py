from __future__ import annotations

import torch

from nimloth.training.rl.planner_verl_gate_factory import (
    GateAlgorithm,
    GateAgent,
    build_tiny_gate_components,
)
from nimloth.training.rl.planner_verl_worker import PlannerWorkerModelComponents


def test_tiny_ray_fsdp_gate_factory_is_deterministic_and_trainable() -> None:
    first = build_tiny_gate_components(
        config={},
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
    )
    second = build_tiny_gate_components(
        config={},
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
    )

    assert isinstance(first, PlannerWorkerModelComponents)
    assert isinstance(first.objective_module.agent, GateAgent)
    assert isinstance(first.objective_module._algorithm, GateAlgorithm)
    assert first.objective_module._algorithm.rank == 0
    assert second.objective_module._algorithm.rank == 1
    assert first.wrap_policy == {"min_num_params": 1}
    assert all(parameter.requires_grad for parameter in first.objective_module.parameters())
    for left, right in zip(
        first.objective_module.parameters(),
        second.objective_module.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
