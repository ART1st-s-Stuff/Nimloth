"""SFT2 验证循环的 mode 与指标聚合测试。"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

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


def test_evaluate_uses_evaluation_step_and_batch_builder() -> None:
    module = torch.nn.Linear(1, 1).train()

    class FakeTarget:
        @contextlib.contextmanager
        def ema_context(self):
            yield

    class FakeAlgorithm:
        agent = SimpleNamespace(trainable_modules=(module,))
        target = FakeTarget()

        def __init__(self) -> None:
            self.values: list[float] = []

        def unwrapped(self):
            return self

        def evaluation_step(self, batch):
            self.values.append(float(batch))
            return SimpleNamespace(metrics={"wm_mse": float(batch)})

    class FakeBuilder:
        def prepare(self, batch):
            return batch

    algorithm = FakeAlgorithm()
    metrics = evaluate(
        algorithm,
        [1.0, 3.0],
        batch_builder=FakeBuilder(),
    )

    assert algorithm.values == [1.0, 3.0]
    assert metrics["wm_mse"] == pytest.approx(2.0)
    assert module.training is True
