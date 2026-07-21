"""rollout 到训练 transition 的通用编码契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class EncodedTransition:
    """backbone hidden、return 与 policy provenance。"""

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


class RolloutEncoder(Protocol):
    """把持久化 trajectory 编码为训练 transition。"""

    def __call__(
        self,
        trajectories: Sequence[Any],
        *,
        gamma: float,
    ) -> list[EncodedTransition]: ...


__all__ = ["EncodedTransition", "RolloutEncoder"]
