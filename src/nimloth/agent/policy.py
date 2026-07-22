"""Agent policy 的公共请求、决策与行为概率契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch

from nimloth.agent.template import AgentPrompt


def behavior_log_probs(
    action_scores: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """把任意 Agent action score 转成实际采样使用的对数概率。"""

    if action_scores.ndim != 1:
        raise ValueError(
            f"action_scores must have shape (A,), got {tuple(action_scores.shape)}"
        )
    if not torch.isfinite(action_scores).any():
        raise ValueError("action_scores must contain at least one finite value")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0.0:
        chosen = action_scores.argmax(dim=-1, keepdim=True)
        log_probs = torch.full_like(action_scores, float("-inf"))
        return log_probs.scatter(dim=-1, index=chosen, value=0.0)

    scaled_scores = action_scores / temperature
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(
            scaled_scores,
            dim=-1,
            descending=True,
        )
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_keep = cumulative_before < top_p
        keep = torch.zeros_like(sorted_keep).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_keep,
        )
        scaled_scores = scaled_scores.masked_fill(~keep, float("-inf"))
    return torch.log_softmax(scaled_scores, dim=-1)


def categorical_entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """计算允许包含 top-p ``-inf`` mask 的离散分布 entropy。"""

    probabilities = log_probs.exp()
    terms = torch.where(
        probabilities > 0,
        probabilities * log_probs,
        torch.zeros_like(log_probs),
    )
    return -terms.sum(dim=-1).mean()


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


def sample_policy_decision(
    action_scores: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> PolicyDecision:
    """按统一采样规则从 action score 构造可审计的行为决策。"""

    log_probs = behavior_log_probs(
        action_scores,
        temperature=temperature,
        top_p=top_p,
    )
    if temperature == 0.0:
        action_index = int(action_scores.argmax().item())
    else:
        action_index = int(torch.multinomial(log_probs.exp(), 1).item())
    return PolicyDecision(
        action_index=action_index,
        action_log_probs=tuple(float(value) for value in log_probs.cpu().tolist()),
    )


class AgentPolicy(Protocol):
    """接收结构化 AgentPrompt 的 episode policy 协议。"""

    def reset_episode(self) -> None:
        """清除上一个 episode 的 policy 运行期状态。"""
        ...

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        ...


@dataclass(frozen=True)
class PolicyReplayInput:
    """PPO 重放一次已执行动作所需的完整 Agent 输入。"""

    prompt: AgentPrompt
    action_index: int
    sampling_temperature: float
    sampling_top_p: float
    latent_token_count: int


class ActionLogProbReplay(Protocol):
    """用当前 policy 重放 trajectory 中保存的动作分布。"""

    def __call__(
        self,
        samples: tuple[PolicyReplayInput, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
