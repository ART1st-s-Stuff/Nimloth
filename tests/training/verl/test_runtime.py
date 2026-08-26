from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from nimloth.training.verl import runtime
from nimloth.training.verl.runtime import (
    assemble_training_root,
    clip_complete_fsdp_grad_norm_,
)


class _CompleteModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("offset", torch.tensor(0.0))


class _WrappedRoot(nn.Module):
    def __init__(self, module: nn.Module, events: list[str]) -> None:
        super().__init__()
        events.append("wrap")
        assert module.training
        assert all(value.device.type == "cpu" for value in module.parameters())
        assert all(value.device.type == "cpu" for value in module.buffers())
        self.module = module
        self.wrapper_scale = nn.Parameter(torch.tensor(1.0))

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        assert max_norm == 2.0
        return torch.tensor(1.25)


def test_complete_root_moves_then_wraps_then_builds_optimizer() -> None:
    events: list[str] = []
    module = _CompleteModule().eval()

    def wrap(value: nn.Module) -> nn.Module:
        return _WrappedRoot(value, events)

    def optimizer_factory(parameters):
        events.append("optimizer")
        values = list(parameters)
        assert len(values) == 2  # original weight plus wrapper-created parameter
        return torch.optim.SGD(values, lr=0.1)

    assembly = assemble_training_root(
        module,
        device=torch.device("cpu"),
        wrap=wrap,
        optimizer_factory=optimizer_factory,
    )

    assert events == ["wrap", "optimizer"]
    assert isinstance(assembly.root, _WrappedRoot)
    assert module.training
    assert clip_complete_fsdp_grad_norm_(assembly.root, 2.0).item() == 1.25


def test_production_fsdp_rejects_unavailable_device_before_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "verify_pinned_vagen_verl_source", lambda _root: None)
    monkeypatch.setattr(runtime, "require_pinned_verl_import", lambda _root: object)
    with pytest.raises(RuntimeError, match="requires an available CUDA"):
        runtime.wrap_complete_fsdp(
            _CompleteModule(),
            device=torch.device("cpu"),
            wrap_policy={"transformer_layer_cls_to_wrap": ["Layer"]},
            mixed_precision=runtime.MixedPrecisionConfig(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            repo_root=__file__,  # source import is replaced by the structural fixture
        )


def test_generic_runtime_contains_no_manual_gradient_all_reduce() -> None:
    source = inspect.getsource(runtime)
    assert "all_reduce" not in source
