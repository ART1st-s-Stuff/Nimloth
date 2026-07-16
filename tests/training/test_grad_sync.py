from __future__ import annotations

import torch

from nimloth.training.common import grad_sync
from nimloth.training.common.grad_sync import (
    assert_consistent_relative_placement,
    average_module_gradients,
    average_partitioned_module_gradients,
    relative_trainable_placement,
)


def test_average_module_gradients_all_reduces_then_averages(monkeypatch) -> None:
    module = torch.nn.Linear(2, 1)
    for param in module.parameters():
        param.grad = torch.full_like(param, 3.0)
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)

    def fake_all_reduce(grad: torch.Tensor, *, group=None) -> None:
        assert group is None
        calls.append(grad)
        grad.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    average_module_gradients(module)

    assert len(calls) == len(list(module.parameters())) + 1
    assert all(torch.equal(param.grad, torch.full_like(param, 3.0)) for param in module.parameters())


def test_average_module_gradients_rejects_missing_trainable_grad(monkeypatch) -> None:
    module = torch.nn.Linear(2, 1)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _tensor, group=None: None)

    try:
        average_module_gradients(module)
    except RuntimeError as exc:
        assert "missing gradient" in str(exc)
    else:
        raise AssertionError("missing trainable gradients must fail before optimizer.step")


def test_average_module_gradients_can_sync_through_cpu_group(monkeypatch) -> None:
    module = torch.nn.Linear(2, 1).to(dtype=torch.bfloat16)
    for param in module.parameters():
        param.grad = torch.full_like(param, 2.0)
    group = object()
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda actual: 2)

    def fake_all_reduce(value: torch.Tensor, *, group=None) -> None:
        assert group is not None
        assert value.device.type == "cpu"
        calls.append(value)
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    average_module_gradients(module, group=group, cpu=True, cpu_bucket_numel=2)

    assert len(calls) == 3  # missing-gradient check plus two bounded buckets
    assert all(torch.equal(param.grad, torch.full_like(param, 2.0)) for param in module.parameters())


def test_relative_trainable_placement_uses_pair_local_slots() -> None:
    class FakeModule:
        def named_parameters(self):
            rows = [
                ("primary", type("P", (), {"requires_grad": True, "device": torch.device("cuda:2"), "shape": (3,)})()),
                ("secondary", type("P", (), {"requires_grad": True, "device": torch.device("cuda:3"), "shape": (4,)})()),
            ]
            return iter(rows)

    placement = relative_trainable_placement(FakeModule(), primary_device=2, stride=2)

    assert placement == (("primary", 0, (3,)), ("secondary", 1, (4,)))


def test_partitioned_gradient_sync_routes_slots_to_gpu_groups(monkeypatch) -> None:
    module = torch.nn.Linear(2, 1)
    for param in module.parameters():
        param.grad = torch.full_like(param, 3.0)
    groups = (object(), object())
    placement = (("weight", 0, (1, 2)), ("bias", 1, (1,)))
    monkeypatch.setattr(grad_sync, "relative_trainable_placement", lambda *args, **kwargs: placement)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    calls = []

    def fake_all_reduce(value: torch.Tensor, *, group=None) -> None:
        calls.append(group)
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    average_partitioned_module_gradients(module, groups=groups, primary_device=0)

    assert calls == [groups[0], groups[0], groups[1], groups[1]]
    assert all(torch.equal(param.grad, torch.full_like(param, 3.0)) for param in module.parameters())


def test_consistent_placement_rejects_rank_mismatch(monkeypatch) -> None:
    module = torch.nn.Linear(2, 1)
    monkeypatch.setattr(
        grad_sync,
        "relative_trainable_placement",
        lambda *args, **kwargs: (("weight", 0, (1, 2)),),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_gather(output, local, *, group=None) -> None:
        output[:] = [local, (("weight", 1, (1, 2)),)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_gather)

    try:
        assert_consistent_relative_placement(module, primary_device=0, stride=2, group=object())
    except RuntimeError as exc:
        assert "placement mismatch" in str(exc)
    else:
        raise AssertionError("rank-relative placement mismatch must fail")
