"""Explicit epoch-level loss plateau policy, independent of runtime budgets."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ConvergencePolicy:
    min_epochs: int
    patience_epochs: int
    min_relative_improvement: float

    def __post_init__(self) -> None:
        for name in ("min_epochs", "patience_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.min_relative_improvement) or not (
            0 <= self.min_relative_improvement < 1
        ):
            raise ValueError("min_relative_improvement must be finite and in [0, 1)")

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConvergenceState:
    schema: str = "adjacent_epoch_lm_v1"
    best_loss: float | None = None
    previous_loss: float | None = None
    bad_epochs: int = 0
    last_epoch: int = 0
    converged: bool = False

    def observe(self, *, epoch: int, loss: float, policy: ConvergencePolicy) -> bool:
        """Record consecutive completed epochs; return whether patience is exhausted.

        ``best_loss`` is the absolute minimum. ``previous_loss`` is the immediately
        preceding epoch, so small improvements never accumulate across rounds.
        Patience counts all evaluations after the baseline; minimum epochs gates stopping.
        """
        if self.converged:
            raise ValueError("cannot observe an already converged run")
        if type(epoch) is not int or epoch != self.last_epoch + 1:
            raise ValueError("epoch must be the next consecutive completed epoch")
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("monitored LM loss must be finite and nonnegative")
        improved = self.previous_loss is None or (
            loss < self.previous_loss
            and self.previous_loss - loss
            >= policy.min_relative_improvement * self.previous_loss
        )
        self.best_loss = loss if self.best_loss is None else min(self.best_loss, loss)
        self.previous_loss = loss
        if improved:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.last_epoch = epoch
        self.converged = (
            epoch >= policy.min_epochs and self.bad_epochs >= policy.patience_epochs
        )
        return self.converged

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> ConvergenceState:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid convergence state fields")
        state = cls(**value)
        if state.schema != "adjacent_epoch_lm_v1":
            raise ValueError("incompatible convergence state schema")
        for name in ("bad_epochs", "last_epoch"):
            number = getattr(state, name)
            if type(number) is not int or number < 0:
                raise ValueError(f"invalid convergence state {name}")
        if type(state.converged) is not bool or state.bad_epochs > state.last_epoch:
            raise ValueError("invalid convergence counters")
        for name in ("best_loss", "previous_loss"):
            number = getattr(state, name)
            if number is not None and (not math.isfinite(number) or number < 0):
                raise ValueError(f"invalid convergence state {name}")
        if state.last_epoch == 0:
            if state != cls():
                raise ValueError("nonempty initial convergence state")
        elif state.best_loss is None or state.previous_loss is None:
            raise ValueError("missing monitored loss in convergence state")
        elif state.best_loss > state.previous_loss:
            raise ValueError("best loss exceeds reference loss")
        return state
