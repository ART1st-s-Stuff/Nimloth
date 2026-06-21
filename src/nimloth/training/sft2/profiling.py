"""Optional step-level timing for SFT2 bottleneck analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from nimloth.training.common.dist import is_main


@dataclass
class StepTimer:
    """Collect per-micro-step timings; log rolling averages on optimizer steps."""

    enabled: bool = False
    log_interval: int = 50
    _sections: dict[str, float] = field(default_factory=dict)
    _totals: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _optimizer_steps: int = 0

    def start(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        return time.perf_counter()

    def stop(self, name: str, started_at: float) -> None:
        if not self.enabled:
            return
        self._sections[name] = self._sections.get(name, 0.0) + (time.perf_counter() - started_at)

    def on_optimizer_step(self, *, global_step: int, epoch: int) -> None:
        if not self.enabled:
            self._sections.clear()
            return
        self._optimizer_steps += 1
        for name, value in self._sections.items():
            self._totals[name] = self._totals.get(name, 0.0) + value
            self._counts[name] = self._counts.get(name, 0) + 1
        self._sections.clear()
        if self.log_interval <= 0 or self._optimizer_steps % self.log_interval != 0:
            return
        if not is_main():
            return
        averages = {
            name: self._totals[name] / max(self._counts[name], 1)
            for name in sorted(self._totals)
        }
        print(
            json.dumps(
                {
                    "step_timing": averages,
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_steps_logged": self._optimizer_steps,
                }
            )
        )

    def snapshot(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return dict(self._sections)
