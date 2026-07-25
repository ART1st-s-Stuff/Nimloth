"""PPO step return 到 sampled policy token 的 credit 分配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from nimloth.agent import PolicyReplayInput


def expand_step_advantages(
    step_advantages: torch.Tensor,
    samples: Sequence[PolicyReplayInput],
    *,
    credit_assignment: Literal["action", "turn"],
) -> torch.Tensor:
    """把每个 environment step 的 advantage 分配给该步 loss-mask token。

    当前 ValueHead 是 environment-step/action critic，因此 ``turn`` 采用 VAGEN 的
    turn-wise 语义：同一轮所有 sampled reasoning/action token 共享该步 advantage。
    token/bi-level GAE 需要 token critic，不在这里近似实现。
    """

    if step_advantages.ndim != 1 or len(samples) != step_advantages.numel():
        raise ValueError(
            "step advantages must have one scalar per policy replay sample"
        )
    if credit_assignment not in {"action", "turn"}:
        raise ValueError(
            f"unsupported PPO credit assignment: {credit_assignment!r}"
        )
    counts: list[int] = []
    for sample in samples:
        if sample.credit_assignment != credit_assignment:
            raise ValueError(
                "trajectory credit assignment does not match RL config: "
                f"{sample.credit_assignment!r} != {credit_assignment!r}"
            )
        count = (
            sum(sample.token_trace.loss_mask)
            if sample.token_trace is not None
            else 1
        )
        if credit_assignment == "action" and count != 1:
            raise ValueError("action credit requires exactly one selected token per step")
        if count < 1:
            raise ValueError("each policy replay sample needs at least one selected token")
        counts.append(count)
    return torch.repeat_interleave(
        step_advantages,
        torch.tensor(counts, dtype=torch.long, device=step_advantages.device),
    )


@dataclass(frozen=True)
class TokenCreditOutput:
    """Token GAE policy advantages and unnormalized critic return targets."""

    advantages: torch.Tensor
    returns: torch.Tensor


def token_level_gae(
    turn_returns: torch.Tensor,
    token_values: torch.Tensor,
    samples: Sequence[PolicyReplayInput],
    *,
    gamma: float,
    gae_lambda: float,
) -> TokenCreditOutput:
    """Run low-level GAE inside each sampled turn.

    Each environment-step Monte Carlo return is placed at the final selected
    token of its turn. Earlier selected tokens have zero immediate reward. The
    critic predicts a separate value before every selected token. Turns are
    independent at this level; environment-step discounting already belongs to
    ``turn_returns``.
    """

    if turn_returns.ndim != 1 or len(samples) != turn_returns.numel():
        raise ValueError("turn returns must have one scalar per replay sample")
    if token_values.ndim != 1:
        raise ValueError("token values must have shape (N,)")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("token gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("token GAE lambda must be in [0, 1]")

    counts: list[int] = []
    for sample in samples:
        if sample.credit_assignment != "token" or sample.token_trace is None:
            raise ValueError("token GAE requires token-credit traced samples")
        count = sum(sample.token_trace.loss_mask)
        if count < 2:
            raise ValueError(
                "token credit requires at least one reasoning token and one action token"
            )
        counts.append(count)
    if sum(counts) != token_values.numel():
        raise ValueError(
            "token values do not align with replay sample loss-mask counts"
        )

    raw_advantages = torch.zeros_like(token_values)
    return_targets = torch.zeros_like(token_values)
    offset = 0
    for turn_return, count in zip(turn_returns, counts, strict=True):
        values = token_values[offset : offset + count]
        last_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
        for local_index in range(count - 1, -1, -1):
            reward = (
                turn_return.to(device=values.device, dtype=values.dtype)
                if local_index == count - 1
                else torch.zeros((), dtype=values.dtype, device=values.device)
            )
            next_value = (
                values[local_index + 1]
                if local_index + 1 < count
                else torch.zeros((), dtype=values.dtype, device=values.device)
            )
            delta = reward + gamma * next_value - values[local_index]
            last_advantage = delta + gamma * gae_lambda * last_advantage
            raw_advantages[offset + local_index] = last_advantage
            return_targets[offset + local_index] = (
                last_advantage + values[local_index]
            )
        offset += count

    detached_advantages = raw_advantages.detach()
    normalized = (detached_advantages - detached_advantages.mean()) / (
        detached_advantages.std(unbiased=False) + 1e-8
    )
    return TokenCreditOutput(
        advantages=normalized,
        returns=return_targets.detach(),
    )


__all__ = ["TokenCreditOutput", "expand_step_advantages", "token_level_gae"]
