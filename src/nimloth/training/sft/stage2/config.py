"""Validated query-alignment objective configuration."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryAlignmentConfig:
    grid_size: int = 4
    projector_hidden_dim: int = 2048
    weight_lm: float = 1.0
    weight_dino: float = 1.0

    def __post_init__(self):
        if self.grid_size < 1 or self.projector_hidden_dim < 1:
            raise ValueError("grid and projector dimensions must be positive")
        if any(
            not math.isfinite(w) or w <= 0 for w in (self.weight_lm, self.weight_dino)
        ):
            raise ValueError(
                "query alignment requires finite positive LM and DINO weights"
            )

    @property
    def grid_tokens(self) -> int:
        return self.grid_size**2
