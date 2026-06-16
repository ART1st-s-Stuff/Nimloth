from __future__ import annotations

from nimloth.sft2.schedules import qwen_lr_schedule


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
