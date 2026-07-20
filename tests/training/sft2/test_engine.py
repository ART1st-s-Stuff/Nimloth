"""Behavior contracts for the shared SFT2 train/validation engine."""

from __future__ import annotations

import contextlib

import pytest
import torch

from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.types import SFT2StepOutput
from nimloth.training.sft2.utils import preserve_module_modes


def test_preserve_module_modes_restores_caller_state() -> None:
    training_module = torch.nn.Linear(2, 2).train()
    evaluation_module = torch.nn.Linear(2, 2).eval()

    with preserve_module_modes((training_module, evaluation_module), training=False):
        assert training_module.training is False
        assert evaluation_module.training is False

    assert training_module.training is True
    assert evaluation_module.training is False


def test_evaluate_uses_shared_forward_in_validation_mode() -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.training_flags: list[bool] = []
            self.context_entered = False

        def unwrapped(self):
            return self

        @contextlib.contextmanager
        def validation_context(self):
            self.context_entered = True
            yield

        def forward(self, batch, *, training: bool) -> SFT2StepOutput:
            self.training_flags.append(training)
            return SFT2StepOutput(
                lm_loss=None,
                wm_loss=None,
                sigreg_loss=None,
                value_loss=torch.zeros(()),
                metrics={"wm_mse": float(batch)},
            )

    runner = FakeRunner()
    metrics = evaluate(runner, [1.0, 3.0])

    assert runner.context_entered is True
    assert runner.training_flags == [False, False]
    assert metrics["wm_mse"] == pytest.approx(2.0)
