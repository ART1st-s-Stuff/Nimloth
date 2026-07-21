"""Agent policy 的公共请求、决策与行为概率契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from nimloth.agent.template import AgentPrompt


def validate_action_log_probs(
    action_index: int,
    action_log_probs: tuple[float, ...] | list[float],
    *,
    action_count: int | None = None,
) -> tuple[float, ...]:
    """校验 behavior distribution，并保留 top-p 产生的 ``-inf``。"""

    values = tuple(float(value) for value in action_log_probs)
    expected_count = len(values) if action_count is None else action_count
    if len(values) != expected_count:
        raise ValueError(
            f"policy must return {expected_count} action log probabilities, "
            f"got {len(values)}"
        )
    if not 0 <= action_index < expected_count:
        raise ValueError(
            f"action_index must be in [0, {expected_count}), got {action_index}"
        )
    if not math.isfinite(values[action_index]):
        raise ValueError("chosen action must have a finite behavior log probability")
    if any(math.isnan(value) or value == float("inf") for value in values):
        raise ValueError(
            "action log probabilities may contain -inf, but not NaN or +inf"
        )
    probability_sum = sum(
        math.exp(value) for value in values if math.isfinite(value)
    )
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            f"action log probabilities must normalize to 1, got {probability_sum}"
        )
    return values


@dataclass(frozen=True)
class PolicyDecision:
    """Policy 返回的动作 index 与完整 behavior distribution。"""

    action_index: int
    action_log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_action_log_probs(self.action_index, self.action_log_probs)


class AgentPolicy(Protocol):
    """接收结构化 AgentPrompt 的模型适配协议。"""

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        ...
