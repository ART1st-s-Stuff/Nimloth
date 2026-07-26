from __future__ import annotations

import contextlib

import torch

from nimloth.util.optim import OptimizationRuntime, qwen_lr_schedule


def test_qwen_lr_starts_low_and_ramps_up() -> None:
    start = qwen_lr_schedule(0, warmup_steps=100, total_steps=1000, start_lr=1e-8, peak_lr=5e-7)
    mid = qwen_lr_schedule(50, warmup_steps=100, total_steps=1000, start_lr=1e-8, peak_lr=5e-7)
    peak = qwen_lr_schedule(99, warmup_steps=100, total_steps=1000, start_lr=1e-8, peak_lr=5e-7)
    assert start < mid < peak
    assert abs(peak - 5e-7) < 1e-12


def test_qwen_lr_decays_after_warmup() -> None:
    peak = qwen_lr_schedule(99, warmup_steps=100, total_steps=1000, start_lr=1e-8, peak_lr=5e-7)
    end = qwen_lr_schedule(999, warmup_steps=100, total_steps=1000, start_lr=1e-8, peak_lr=5e-7)
    assert end < peak


def test_optimization_runtime_owns_backward_step_and_callback() -> None:
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    callback_steps: list[int] = []
    runtime = OptimizationRuntime(
        optimizer=optimizer,
        synchronized_modules=(module,),
        after_step=lambda: callback_steps.append(1),
    )
    before = module.weight.detach().clone()

    runtime.zero_grad()
    runtime.backward(module(torch.ones(1, 2)).sum())
    runtime.step()

    assert not torch.equal(module.weight.detach(), before)
    assert callback_steps == [1]
    assert all(parameter.grad is None for parameter in module.parameters())


def test_optimization_runtime_enters_no_sync_only_during_accumulation() -> None:
    class DistributedModule(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 1)
            self.no_sync_calls = 0

        @contextlib.contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield

    module = DistributedModule()
    runtime = OptimizationRuntime(
        optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
        synchronized_modules=(module,),
        enable_no_sync=True,
    )

    with runtime.accumulation_context(sync_gradients=False):
        pass
    with runtime.accumulation_context(sync_gradients=True):
        pass

    assert module.no_sync_calls == 1
