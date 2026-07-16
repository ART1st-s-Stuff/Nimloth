from __future__ import annotations

import torch

from nimloth.training.common.grad_sync import average_module_gradients


def test_average_module_gradients_all_reduces_then_averages(monkeypatch) -> None:
    module = torch.nn.Linear(2, 1)
    for param in module.parameters():
        param.grad = torch.full_like(param, 3.0)
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(grad: torch.Tensor) -> None:
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
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _tensor: None)

    try:
        average_module_gradients(module)
    except RuntimeError as exc:
        assert "missing gradient" in str(exc)
    else:
        raise AssertionError("missing trainable gradients must fail before optimizer.step")
