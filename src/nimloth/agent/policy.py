"""Agent policy 的公共请求、决策与行为概率契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

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
class PolicyTokenTrace:
    """一次 policy continuation 的逐 token behavior provenance。

    ``token_ids`` 从 ``AgentPrompt`` 最后一个 token 之后开始，包含模型采样 token
    与为保持 Nimloth 协议而注入的 token。只有 ``loss_mask=True`` 的位置参加 PPO；
    注入位置的 ``old_log_probs`` 必须为 ``None``。
    """

    token_ids: tuple[int, ...]
    old_log_probs: tuple[float | None, ...]
    loss_mask: tuple[bool, ...]
    token_roles: tuple[Literal["reasoning", "action", "injected"], ...]
    action_token_ids: tuple[int, ...]
    reasoning_text: str | None = None
    finish_reason: Literal["stop", "length"] | None = None
    reasoning_truncated: bool = False

    def __post_init__(self) -> None:
        lengths = {
            len(self.token_ids),
            len(self.old_log_probs),
            len(self.loss_mask),
            len(self.token_roles),
        }
        if len(lengths) != 1 or not self.token_ids:
            raise ValueError("policy token trace fields must have one non-empty length")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("policy token ids must be non-negative")
        if any(token_id < 0 for token_id in self.action_token_ids):
            raise ValueError("policy action token ids must be non-negative")
        if not self.action_token_ids or len(set(self.action_token_ids)) != len(
            self.action_token_ids
        ):
            raise ValueError("policy token trace requires unique action token ids")
        for index, (old_log_prob, selected, role) in enumerate(
            zip(
                self.old_log_probs,
                self.loss_mask,
                self.token_roles,
                strict=True,
            )
        ):
            if role not in {"reasoning", "action", "injected"}:
                raise ValueError(f"unknown policy token role at {index}: {role!r}")
            if selected:
                if old_log_prob is None or not math.isfinite(old_log_prob):
                    raise ValueError(
                        f"selected policy token {index} requires a finite old log-prob"
                    )
                if role == "injected":
                    raise ValueError("injected policy tokens cannot participate in PPO")
            elif old_log_prob is not None:
                raise ValueError(
                    f"unselected policy token {index} must not store an old log-prob"
                )
        if sum(role == "action" for role in self.token_roles) != 1:
            raise ValueError("policy token trace requires exactly one action token")
        action_position = self.token_roles.index("action")
        if not self.loss_mask[action_position]:
            raise ValueError("the sampled action token must participate in PPO")
        if self.token_ids[action_position] not in self.action_token_ids:
            raise ValueError("policy action token is outside the recorded action mapping")
        has_reasoning = "reasoning" in self.token_roles
        if has_reasoning:
            if self.reasoning_text is None:
                raise ValueError("reasoning token trace requires reasoning_text")
            if self.finish_reason not in {"stop", "length"}:
                raise ValueError("reasoning token trace requires a finish reason")
            if self.reasoning_truncated != (self.finish_reason == "length"):
                raise ValueError("reasoning truncation must match finish_reason")
        elif self.reasoning_text is not None or self.finish_reason is not None:
            raise ValueError("action-only token trace cannot store reasoning metadata")
        elif self.reasoning_truncated:
            raise ValueError("action-only token trace cannot be truncated")

    @property
    def selected_old_log_probs(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value, selected in zip(
                self.old_log_probs,
                self.loss_mask,
                strict=True,
            )
            if selected and value is not None
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Policy 返回的动作、behavior distribution 与可选 token provenance。"""

    action_index: int
    action_log_probs: tuple[float, ...]
    response: str | None = None
    token_trace: PolicyTokenTrace | None = None

    def __post_init__(self) -> None:
        action_log_probs = validate_action_log_probs(
            self.action_index,
            self.action_log_probs,
        )
        if self.response is not None and not self.response:
            raise ValueError("policy response must be non-empty when provided")
        if self.token_trace is not None:
            if len(self.token_trace.action_token_ids) != len(action_log_probs):
                raise ValueError(
                    "decision distribution and trace action mapping must align"
                )
            if not 0 <= self.action_index < len(self.token_trace.action_token_ids):
                raise ValueError("decision action_index is outside trace action mapping")
            action_position = self.token_trace.token_roles.index("action")
            expected_token_id = self.token_trace.action_token_ids[self.action_index]
            if self.token_trace.token_ids[action_position] != expected_token_id:
                raise ValueError("decision action does not match policy token trace")
            trace_log_prob = self.token_trace.old_log_probs[action_position]
            if trace_log_prob is None or not math.isclose(
                trace_log_prob,
                action_log_probs[self.action_index],
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    "decision action log-prob does not match policy token trace"
                )


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
    credit_assignment: Literal["action", "turn"] = "action"
    token_trace: PolicyTokenTrace | None = None
    old_action_log_prob: float | None = None
    assistant_response: str | None = None

    def __post_init__(self) -> None:
        if self.credit_assignment not in {"action", "turn"}:
            raise ValueError(
                f"unsupported PPO credit assignment: {self.credit_assignment!r}"
            )
        if self.credit_assignment == "turn":
            if self.token_trace is None:
                raise ValueError("turn credit requires a policy token trace")
            if not any(
                role == "reasoning" and selected
                for role, selected in zip(
                    self.token_trace.token_roles,
                    self.token_trace.loss_mask,
                    strict=True,
                )
            ):
                raise ValueError("turn credit requires sampled reasoning tokens")
            if not self.assistant_response:
                raise ValueError("turn credit requires the sampled assistant response")
        if self.token_trace is None:
            if self.old_action_log_prob is None or not math.isfinite(
                self.old_action_log_prob
            ):
                raise ValueError(
                    "action-only replay without a token trace requires a finite "
                    "old action log-prob"
                )
        elif self.old_action_log_prob is not None:
            raise ValueError(
                "token-trace replay must not duplicate the old action log-prob"
            )
        if self.token_trace is not None:
            if not 0 <= self.action_index < len(self.token_trace.action_token_ids):
                raise ValueError("replay action_index is outside action token mapping")
            action_position = self.token_trace.token_roles.index("action")
            expected_token_id = self.token_trace.action_token_ids[self.action_index]
            if self.token_trace.token_ids[action_position] != expected_token_id:
                raise ValueError(
                    "token trace action does not match replay action_index"
                )

    @property
    def selected_old_log_probs(self) -> tuple[float, ...]:
        if self.token_trace is not None:
            return self.token_trace.selected_old_log_probs
        assert self.old_action_log_prob is not None
        return (float(self.old_action_log_prob),)


@dataclass(frozen=True)
class PolicyReplayOutput:
    """当前 policy 对 loss-mask token 的重放结果。"""

    selected_log_probs: torch.Tensor
    entropies: torch.Tensor

    def __post_init__(self) -> None:
        if self.selected_log_probs.ndim != 1 or self.entropies.ndim != 1:
            raise ValueError("policy replay outputs must both have shape (N,)")
        if self.selected_log_probs.shape != self.entropies.shape:
            raise ValueError("policy replay log-probs and entropies must align")


class ActionLogProbReplay(Protocol):
    """用当前 policy 重放 trajectory 中保存的动作分布。"""

    def __call__(
        self,
        samples: tuple[PolicyReplayInput, ...],
    ) -> PolicyReplayOutput: ...
