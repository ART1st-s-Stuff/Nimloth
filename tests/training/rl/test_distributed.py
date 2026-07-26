from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nimloth.training.rl.trainer import (
    RLTrainingStepModule,
    _wrap_distributed_modules,
    _wrap_training_step_ddp,
    _wrap_world_model_ddp,
)
from nimloth.wm import WorldModel


class _FakeDDP(nn.Module):
    def __init__(self, module: nn.Module, **kwargs) -> None:
        super().__init__()
        self.module = module
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.sigreg = nn.Linear(4, 4)

    def training_step(self, runtime, batch):  # type: ignore[no-untyped-def]
        return runtime.agent(batch)


def test_training_step_module_registers_complete_loss_modules() -> None:
    agent = nn.Linear(4, 4)
    token_value_head = nn.Linear(4, 1)
    algorithm = _FakeAlgorithm()
    training_step = RLTrainingStepModule(
        algorithm=algorithm,  # type: ignore[arg-type]
        runtime=SimpleNamespace(agent=agent),  # type: ignore[arg-type]
        token_value_head=token_value_head,
    )

    assert training_step.agent is agent
    assert training_step.token_value_head is token_value_head
    assert training_step.sigreg is algorithm.sigreg
    assert set(training_step.parameters()) == {
        *agent.parameters(),
        *token_value_head.parameters(),
        *algorithm.sigreg.parameters(),
    }


def test_world_model_ddp_wraps_only_trainable_modules(monkeypatch) -> None:
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    frozen_projector = nn.Linear(4, 4)
    frozen_projector.requires_grad_(False)
    world_model = WorldModel(
        state_proj=frozen_projector,
        wm_predictor=nn.Linear(4, 4),
        value_head=nn.Linear(4, 8),
    )

    wrapped = _wrap_world_model_ddp(
        world_model,
        device=torch.device("cuda:3"),
        world_size=4,
    )

    assert wrapped.state_proj is frozen_projector
    assert isinstance(wrapped.wm_predictor, _FakeDDP)
    assert isinstance(wrapped.value_head, _FakeDDP)
    assert wrapped.wm_predictor.kwargs["device_ids"] == [3]
    assert wrapped.value_head.kwargs["static_graph"] is True


def test_pair_parallel_reserves_one_complete_step_ddp_boundary() -> None:
    model = nn.Linear(4, 4)
    world_model = WorldModel(
        state_proj=nn.Linear(4, 4),
        wm_predictor=nn.Linear(4, 4),
        value_head=nn.Linear(4, 8),
    )

    wrapped = _wrap_distributed_modules(
        model,
        world_model,
        None,
        world_size=4,
        model_parallel=True,
        training_device=torch.device("cuda:1"),
    )

    assert wrapped.model is model
    assert wrapped.world_model is world_model
    assert wrapped.strategy == "model_parallel_ddp"
    assert wrapped.optimizer_state_sharded is False


def test_pair_parallel_training_step_uses_multidevice_ddp(monkeypatch) -> None:
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    training_step = nn.Linear(4, 1)

    wrapped = _wrap_training_step_ddp(  # type: ignore[arg-type]
        training_step,
        world_size=4,
        model_parallel=True,
    )

    assert isinstance(wrapped, _FakeDDP)
    assert wrapped.module is training_step
    assert wrapped.kwargs["device_ids"] is None
    assert wrapped.kwargs["output_device"] is None
    assert wrapped.kwargs["find_unused_parameters"] is False
    assert wrapped.kwargs["static_graph"] is True


def test_planner_pair_parallel_uses_official_dynamic_ddp(monkeypatch) -> None:
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)

    wrapped = _wrap_training_step_ddp(  # type: ignore[arg-type]
        nn.Linear(4, 1),
        world_size=2,
        model_parallel=True,
        allow_unused_parameters=True,
    )

    assert isinstance(wrapped, _FakeDDP)
    assert wrapped.kwargs["device_ids"] is None
    assert wrapped.kwargs["find_unused_parameters"] is True
    assert wrapped.kwargs["static_graph"] is False
