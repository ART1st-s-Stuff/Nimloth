"""用于训练和评估瓶颈分析的可选分段计时器。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from nimloth.util.distributed import is_main


@dataclass
class StepTimer:
    """Sample complete optimizer updates and log cumulative sampled phase means.

    Sampling starts with the first local update, then every ``sample_interval``
    updates. All microbatches in an update share the same sampling decision.
    Disabled/unsampled updates never synchronize CUDA through this timer.
    """

    enabled: bool = False
    log_interval: int = 50
    sample_interval: int = 1
    _profiled_optimizer_steps: int = 0
    _sections: dict[str, float] = field(default_factory=dict)
    _totals: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _optimizer_steps: int = 0

    def __post_init__(self) -> None:
        if self.sample_interval < 1:
            raise ValueError("sample_interval must be positive")

    @property
    def sampling(self) -> bool:
        return self.enabled and self._optimizer_steps % self.sample_interval == 0

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name: str) -> float:
        if not self.sampling:
            return 0.0
        self._sync_cuda()
        return time.perf_counter()

    def stop(self, name: str, started_at: float) -> None:
        if not self.sampling:
            return
        self._sync_cuda()
        elapsed = time.perf_counter() - started_at
        self._sections[name] = self._sections.get(name, 0.0) + elapsed

    def on_optimizer_step(self, *, global_step: int, epoch: int) -> None:
        if not self.enabled:
            self._sections.clear()
            return
        sampled = self.sampling
        if sampled:
            self._profiled_optimizer_steps += 1
        self._optimizer_steps += 1
        for name, value in self._sections.items():
            self._totals[name] = self._totals.get(name, 0.0) + value
            self._counts[name] = self._counts.get(name, 0) + 1
        self._sections.clear()
        if (
            not sampled
            or self.log_interval <= 0
            or self._profiled_optimizer_steps % self.log_interval != 0
        ):
            return
        cuda_memory = None
        if torch.cuda.is_available():
            gib = 1024**3
            cuda_memory = {
                "allocated_gib": torch.cuda.memory_allocated() / gib,
                "reserved_gib": torch.cuda.memory_reserved() / gib,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
            }
            torch.cuda.reset_peak_memory_stats()
        if not is_main():
            return
        averages = {
            name: self._totals[name] / max(self._counts[name], 1)
            for name in sorted(self._totals)
        }
        payload: dict[str, Any] = {
            "step_timing": averages,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps_logged": self._optimizer_steps,
            "optimizer_steps_profiled": self._profiled_optimizer_steps,
            "sample_interval": self.sample_interval,
            "timing_aggregation": "cumulative_sampled_optimizer_updates",
            "section_sample_counts": dict(self._counts),
        }
        if cuda_memory is not None:
            payload["cuda_memory"] = cuda_memory
        print(json.dumps(payload))

    def snapshot(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return dict(self._sections)
