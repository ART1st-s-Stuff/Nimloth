from __future__ import annotations

from nimloth.training.sft2.profiling import StepTimer


def test_step_timer_disabled_is_noop() -> None:
    timer = StepTimer(enabled=False)
    started = timer.start("dataloader")
    timer.stop("dataloader", started)
    timer.on_optimizer_step(global_step=1, epoch=1)
    assert timer.snapshot() == {}
