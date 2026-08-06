from __future__ import annotations

import torch
from torch import nn

from nimloth.backbone import (
    Backbone,
    BackboneBatch,
    BackboneOutput,
    DistributedBackbone,
)
from nimloth.training.rl.trainer import (
    _wrap_distributed_modules,
    _wrap_world_model_ddp,
)
from nimloth.util import distributed as distributed_module
from nimloth.wm import WorldModel


class _FakeDDP(nn.Module):
    def __init__(self, module: nn.Module, **kwargs) -> None:
        super().__init__()
        self.module = module
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _Backbone(Backbone):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.language_model = model

    @property
    def model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        del include_lm_loss
        return BackboneOutput(hidden=self.model(batch.tensors["hidden"]))

    def with_model(self, model: nn.Module) -> "_Backbone":
        return _Backbone(model)

    def save_pretrained(self, *args, **kwargs) -> None:
        del args, kwargs


def test_failed_distributed_cleanup_does_not_enter_barrier(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        distributed_module.dist,
        "barrier",
        lambda: calls.append("barrier"),
    )
    monkeypatch.setattr(
        distributed_module.dist,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )

    distributed_module.cleanup_dist(synchronize=False)

    assert calls == ["destroy"]


def test_healthy_distributed_cleanup_synchronizes_before_destroy(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        distributed_module.dist,
        "barrier",
        lambda: calls.append("barrier"),
    )
    monkeypatch.setattr(
        distributed_module.dist,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )

    distributed_module.cleanup_dist()

    assert calls == ["barrier", "destroy"]


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
    assert wrapped.wm_predictor.kwargs["broadcast_buffers"] is False
    assert wrapped.value_head.kwargs["broadcast_buffers"] is False
    assert wrapped.value_head.kwargs["static_graph"] is True


def test_pair_parallel_wraps_actual_parameter_modules(monkeypatch) -> None:
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    model = nn.Linear(4, 4)
    backbone = _Backbone(model)
    token_value_head = nn.Linear(4, 1)
    world_model = WorldModel(
        state_proj=nn.Linear(4, 4),
        wm_predictor=nn.Linear(4, 4),
        value_head=nn.Linear(4, 8),
    )

    wrapped = _wrap_distributed_modules(
        backbone,
        world_model,
        token_value_head,
        world_size=4,
        model_parallel=True,
        synchronize_backbone_hidden=True,
        training_device=torch.device("cuda:1"),
    )

    assert isinstance(wrapped.backbone, DistributedBackbone)
    distributed_backbone = wrapped.backbone.wrapped
    assert isinstance(distributed_backbone, _FakeDDP)
    assert distributed_backbone.module is backbone
    assert distributed_backbone.kwargs["device_ids"] is None
    assert distributed_backbone.kwargs["output_device"] is None
    assert distributed_backbone.kwargs["broadcast_buffers"] is False
    assert distributed_backbone.kwargs["find_unused_parameters"] is False
    assert distributed_backbone.kwargs["static_graph"] is False
    assert wrapped.model is model
    assert wrapped.backbone.synchronized_modules == (distributed_backbone,)
    assert isinstance(wrapped.world_model.state_proj, _FakeDDP)
    assert isinstance(wrapped.world_model.wm_predictor, _FakeDDP)
    assert isinstance(wrapped.world_model.value_head, _FakeDDP)
    assert wrapped.world_model.state_proj.kwargs["device_ids"] == [1]
    assert isinstance(wrapped.token_value_head, _FakeDDP)
    assert wrapped.token_value_head.module is token_value_head
    assert wrapped.strategy == "model_parallel_ddp"
    assert wrapped.optimizer_state_sharded is False


def test_pair_parallel_actor_keeps_logits_model_inside_ddp(monkeypatch) -> None:
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    model = nn.Linear(4, 4)
    backbone = _Backbone(model)
    world_model = WorldModel(
        state_proj=nn.Linear(4, 4),
        wm_predictor=nn.Linear(4, 4),
        value_head=nn.Linear(4, 8),
    )

    wrapped = _wrap_distributed_modules(
        backbone,
        world_model,
        None,
        world_size=4,
        model_parallel=True,
        synchronize_backbone_hidden=False,
        training_device=torch.device("cuda:1"),
    )

    assert not isinstance(wrapped.backbone, DistributedBackbone)
    assert isinstance(wrapped.model, _FakeDDP)
    assert wrapped.model.module is model
    assert wrapped.model.kwargs["find_unused_parameters"] is False
    assert wrapped.model.kwargs["broadcast_buffers"] is False
    assert wrapped.model.kwargs["static_graph"] is True
    assert wrapped.backbone.synchronized_modules == (wrapped.model,)
