"""SFT2 验证循环的 mode 与指标聚合测试。"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage3.evaluate import evaluate
from nimloth.training.sft.stage3.utils import preserve_module_modes


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

    class FakeAlgorithm:
        def __init__(self) -> None:
            self.values: list[float] = []

        def evaluation_step(self, _runtime, batch):
            self.values.append(float(batch))
            return SimpleNamespace(
                metrics={"wm_mse": float(batch)},
                sample_count=1,
            )

    class FakeRuntime:
        agent = SimpleNamespace(trainable_modules=(module,))

        def unwrapped(self):
            return self

        @contextlib.contextmanager
        def evaluation_context(self):
            yield

    class FakeBuilder:
        def prepare(self, batch):
            return batch

    algorithm = FakeAlgorithm()
    metrics = evaluate(
        algorithm,
        FakeRuntime(),
        [1.0, 3.0],
        batch_builder=FakeBuilder(),
    )

    assert algorithm.values == [1.0, 3.0]
    assert metrics["wm_mse"] == pytest.approx(2.0)
    assert module.training is True


@pytest.mark.parametrize("counts, expected", [([1, 2, 0], 10 / 3), ([0, 0, 0], None)])
def test_evaluate_lm_uses_only_successful_windows(counts, expected):
    class Algorithm:
        def evaluation_step(self, runtime, batch):
            return SimpleNamespace(metrics={"lm_ce": batch[0], "wm_mse": 1.0}, sample_count=3)

    class Runtime:
        agent = SimpleNamespace(trainable_modules=())

        def unwrapped(self):
            return self

        @contextlib.contextmanager
        def evaluation_context(self):
            yield

    class Builder:
        def prepare(self, batch):
            return batch

        def supervision_counts(self, batch):
            return 3, batch[1]

    result = evaluate(Algorithm(), Runtime(), list(zip([2., 4., 0.], counts)), batch_builder=Builder())
    if expected is None:
        assert "lm_ce" not in result
    else:
        assert result["lm_ce"] == pytest.approx(expected)
    assert result["wm_mse"] == 1.0


def test_zero_weight_padding_executes_but_does_not_bias_metrics():
    calls = []
    class Algorithm:
        def evaluation_step(self, runtime, batch):
            calls.append(batch)
            return SimpleNamespace(metrics={"wm_mse": batch[0]}, sample_count=batch[1])
    class Runtime:
        agent = SimpleNamespace(trainable_modules=())
        def unwrapped(self): return self
        def evaluation_context(self): return contextlib.nullcontext()
    builder = SimpleNamespace(prepare=lambda batch: batch)
    result = evaluate(Algorithm(), Runtime(), [(2.0, 1), (999.0, 0)], batch_builder=builder)
    assert len(calls) == 2
    assert result["wm_mse"] == 2.0


def test_preserve_modes_restores_mixed_descendants_even_on_exception():
    import pytest
    parent = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout()).train()
    parent[0].eval()
    before = {module: module.training for module in parent.modules()}
    with pytest.raises(RuntimeError, match="target failed"):
        with preserve_module_modes((parent, parent[0]), training=False):
            assert all(not module.training for module in parent.modules())
            raise RuntimeError("target failed")
    assert {module: module.training for module in parent.modules()} == before


def test_epoch_sets_online_training_before_any_batch_or_forward():
    from types import SimpleNamespace
    import pytest
    from nimloth.training.sft.stage3.loop import SFT2TrainingLoop
    from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
    online = torch.nn.Sequential(torch.nn.Linear(2, 2)).eval()
    teacher = torch.nn.Linear(2, 2).eval()
    runtime = SFT2ModelRuntime(agent=SimpleNamespace(trainable_modules=(online,)), history_cache=None)
    loop = object.__new__(SFT2TrainingLoop)
    loop.model_runtime = runtime
    def first_sampler_operation(epoch):
        assert epoch == 1
        assert all(module.training for module in online.modules())
        assert not teacher.training
        raise RuntimeError("reached sampler")
    loop._set_sampler_epoch = first_sampler_operation
    with pytest.raises(RuntimeError, match="reached sampler"):
        loop._run_epoch(1)
