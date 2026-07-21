"""SFT2 算法在训练/验证之间共享的行为契约。"""

from __future__ import annotations

import contextlib

import pytest
import torch

from nimloth.training.sft2.algorithm import SFT2Losses, SFT2Mode
from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.utils import preserve_module_modes


def test_preserve_module_modes_restores_caller_state() -> None:
    training_module = torch.nn.Linear(2, 2).train()
    evaluation_module = torch.nn.Linear(2, 2).eval()

    with preserve_module_modes((training_module, evaluation_module), training=False):
        assert training_module.training is False
        assert evaluation_module.training is False

    assert training_module.training is True
    assert evaluation_module.training is False


def test_evaluate_uses_algorithm_validation_mode() -> None:
    class FakeAlgorithm:
        def __init__(self) -> None:
            self.modes: list[SFT2Mode] = []
            self.context_entered = False

        def unwrapped(self):
            return self

        @contextlib.contextmanager
        def validation_context(self):
            self.context_entered = True
            yield

        def compute(self, batch, *, mode: SFT2Mode) -> SFT2Losses:
            self.modes.append(mode)
            return SFT2Losses(
                lm=None,
                dynamics=None,
                sigreg=None,
                value=torch.zeros(()),
                metrics={"wm_mse": float(batch)},
            )

    algorithm = FakeAlgorithm()
    metrics = evaluate(algorithm, [1.0, 3.0])

    assert algorithm.context_entered is True
    assert algorithm.modes == [SFT2Mode.VALIDATE, SFT2Mode.VALIDATE]
    assert metrics["wm_mse"] == pytest.approx(2.0)
