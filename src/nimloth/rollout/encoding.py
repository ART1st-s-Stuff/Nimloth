"""rollout 到训练 transition 的通用编码契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class EncodedTransition:
    """backbone hidden、return 与 policy provenance。"""

    record_id: str
    step_index: int
    current_hidden: torch.Tensor
    next_hidden: torch.Tensor
    action_index: int
    value_target: float
    old_log_prob: float
    policy_messages: list[dict[str, Any]]
    policy_image_paths: list[str]
    sampling_temperature: float
    sampling_top_p: float
    latent_token_count: int


@dataclass(frozen=True)
class EncodedTrajectory:
    """保留连续顺序的 backbone latent trajectory。"""

    record_id: str
    transitions: tuple[EncodedTransition, ...]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("encoded trajectory must contain at least one transition")
        for expected_step, transition in enumerate(self.transitions):
            if transition.record_id != self.record_id:
                raise ValueError(
                    "encoded transition record_id does not match its trajectory: "
                    f"{transition.record_id!r} != {self.record_id!r}"
                )
            if transition.step_index != expected_step:
                raise ValueError(
                    "encoded trajectory steps must be consecutive from zero: "
                    f"expected {expected_step}, got {transition.step_index}"
                )

    @property
    def num_steps(self) -> int:
        return len(self.transitions)


class RolloutEncoder(Protocol):
    """把持久化 trajectory 编码为保留连续顺序的 latent trajectory。"""

    def __call__(
        self,
        trajectories: Sequence[Any],
        *,
        gamma: float,
    ) -> list[EncodedTrajectory]: ...


__all__ = ["EncodedTrajectory", "EncodedTransition", "RolloutEncoder"]
