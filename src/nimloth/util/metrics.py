"""训练和评估循环共享的指标累计。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, metrics: dict[str, float], count: int = 1) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * count
            self.counts[key] = self.counts.get(key, 0) + count

    def averages(self) -> dict[str, float]:
        return {key: self.sums[key] / max(1, self.counts[key]) for key in self.sums}

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()
