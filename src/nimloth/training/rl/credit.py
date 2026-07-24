"""PPO step return 到 sampled policy token 的 credit 分配。"""

from __future__ import annotations

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


__all__ = ["expand_step_advantages"]
