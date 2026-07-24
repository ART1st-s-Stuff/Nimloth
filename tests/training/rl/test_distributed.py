from __future__ import annotations

import torch
from torch import nn

from nimloth.training.rl.trainer import (
    _wrap_distributed_modules,
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


def test_pair_parallel_uses_deterministic_manual_gradient_sync() -> None:
    model = nn.Linear(4, 4)
    world_model = WorldModel(
        state_proj=nn.Linear(4, 4),
        wm_predictor=nn.Linear(4, 4),
        value_head=nn.Linear(4, 8),
    )

    wrapped = _wrap_distributed_modules(
        model,
        world_model,
        world_size=4,
        model_parallel=True,
        training_device=torch.device("cuda:1"),
    )

    assert wrapped.model is model
    assert wrapped.world_model is world_model
    assert wrapped.strategy == "model_parallel_manual_sync"
    assert wrapped.manual_gradient_sync is True
    assert wrapped.optimizer_state_sharded is False
