"""Activation storage selection and its typed YAML/CLI path."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from nimloth.config.sft2 import SFT2LoopConfig, flatten_sft2_yaml_config
from nimloth.training.common import activation_offload
from nimloth.training.sft.stage3.cli import parse_sft2_args


_REQUIRED = ["--model", "/tmp/model", "--train-jsonl", "/tmp/train.jsonl",
             "--val-jsonl", "/tmp/val.jsonl", "--output-dir", "/tmp/output"]


def test_offload_uses_pytorch_saved_tensor_context_only_when_enabled(monkeypatch):
    calls = []
    @contextmanager
    def save_on_cpu(*, pin_memory):
        assert pin_memory is True
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", save_on_cpu)
    with activation_offload.saved_activation_context(False):
        assert calls == []
    with activation_offload.saved_activation_context(True):
        assert calls == ["enter"]
    assert calls == ["enter", "exit"]


def test_offload_config_defaults_yaml_and_explicit_cli_override():
    default = parse_sft2_args(_REQUIRED)
    assert not default.activation_offload
    assert not SFT2LoopConfig.from_namespace(default).activation_offload
    config = Path(__file__).resolve().parents[3] / "configs/training/sft2/action_outcome_k64_h1_t4.yaml"
    experiment = parse_sft2_args(["--config", str(config), *_REQUIRED])
    assert not experiment.activation_offload
    assert not SFT2LoopConfig.from_namespace(experiment).activation_offload
    disabled = parse_sft2_args(["--config", str(config), *_REQUIRED, "--no-activation-offload"])
    assert not disabled.activation_offload
    enabled = parse_sft2_args([*_REQUIRED, "--activation-offload"])
    assert enabled.activation_offload
    with pytest.raises(ValueError, match="must be a boolean"):
        flatten_sft2_yaml_config({"train": {"activation_offload": "false"}})


def test_cpu_saved_activation_values_and_gradients_are_unchanged():
    values = torch.linspace(-1, 1, 32, requires_grad=True)
    with activation_offload.saved_activation_context(True):
        loss = values.sin().square().sum()
    loss.backward()
    reference = values.detach().clone().requires_grad_()
    reference.sin().square().sum().backward()
    assert torch.equal(values.grad, reference.grad)
