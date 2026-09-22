from __future__ import annotations

from nimloth.util.profiling import StepTimer


def test_step_timer_disabled_is_noop() -> None:
    timer = StepTimer(enabled=False)
    started = timer.start("dataloader")
    timer.stop("dataloader", started)
    timer.on_optimizer_step(global_step=1, epoch=1)
    assert timer.snapshot() == {}


def test_sampled_timer_profiles_whole_updates_without_unsampled_sync(monkeypatch, capsys) -> None:
    import json

    import nimloth.util.profiling as profiling

    syncs = []
    ticks = iter(float(i) for i in range(100))
    monkeypatch.setattr(StepTimer, "_sync_cuda", staticmethod(lambda: syncs.append(True)))
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(profiling, "is_main", lambda: True)
    timer = StepTimer(enabled=True, log_interval=1, sample_interval=10)
    # Simulate two accumulated microbatches per update, including a resumed global step.
    for local_step in range(1, 12):
        before = len(syncs)
        for _ in range(2):
            start = timer.start("forward")
            timer.stop("forward", start)
        timer.on_optimizer_step(global_step=100 + local_step, epoch=2)
        if local_step not in (1, 11):
            assert len(syncs) == before
            assert timer.snapshot() == {}
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(syncs) == 8
    assert [row["global_step"] for row in rows] == [101, 111]
    assert rows[-1]["optimizer_steps_logged"] == 11
    assert rows[-1]["optimizer_steps_profiled"] == 2
    assert rows[-1]["section_sample_counts"] == {"forward": 2}
    # Sum both microbatches; divide by sampled optimizer updates, not all 11 updates.
    assert rows[-1]["step_timing"] == {"forward": 2.0}


def test_log_interval_counts_profiled_updates(monkeypatch, capsys) -> None:
    import json

    import nimloth.util.profiling as profiling

    monkeypatch.setattr(StepTimer, "_sync_cuda", staticmethod(lambda: None))
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(profiling, "is_main", lambda: True)
    timer = StepTimer(enabled=True, log_interval=2, sample_interval=3)
    for step in range(1, 7):
        timer.stop("forward", timer.start("forward"))
        timer.on_optimizer_step(global_step=step, epoch=1)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["global_step"] for row in rows] == [4]
    assert rows[0]["optimizer_steps_profiled"] == 2


def test_sample_interval_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="sample_interval"):
        StepTimer(sample_interval=0)
