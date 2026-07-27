"""实际执行动作的 Monte Carlo value 监督。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionValueLoss:
    """ValueHead 对执行动作的 loss 分解。"""

    loss: torch.Tensor
    monte_carlo_mse: torch.Tensor
    ranking: torch.Tensor
    selected_action_values: torch.Tensor


def action_value_loss(
    action_values: torch.Tensor,
    executed_actions: torch.Tensor,
    monte_carlo_returns: torch.Tensor,
    *,
    ranking_margin: float = 0.0,
    ranking_weight: float = 0.0,
) -> ActionValueLoss:
    """回归执行动作的 MC return，并可选鼓励其高于未执行动作。

    调用方决定评分 state。默认只做执行 action 的 MC 回归；只有显式传入非零
    ``ranking_weight`` 时才读取未执行 action 并建立 ranking 分支。
    """

    selected_action_values = action_values.gather(
        -1,
        executed_actions.unsqueeze(-1),
    ).squeeze(-1)
    targets = monte_carlo_returns.to(
        device=action_values.device,
        dtype=action_values.dtype,
    )
    monte_carlo_mse = F.mse_loss(selected_action_values, targets)

    if ranking_weight == 0.0:
        # 纯 MC 回归不读取未执行 action，避免权重为零时仍建立反事实梯度分支。
        ranking = monte_carlo_mse.new_zeros(())
        loss = monte_carlo_mse
    else:
        selected_mask = F.one_hot(
            executed_actions,
            num_classes=action_values.shape[-1],
        ).bool()
        best_unselected_value = action_values.masked_fill(
            selected_mask,
            float("-inf"),
        ).max(dim=-1).values
        ranking = F.relu(
            ranking_margin + best_unselected_value - selected_action_values
        ).mean()
        loss = monte_carlo_mse + ranking_weight * ranking

    return ActionValueLoss(
        loss=loss,
        monte_carlo_mse=monte_carlo_mse,
        ranking=ranking,
        selected_action_values=selected_action_values,
    )


__all__ = ["ActionValueLoss", "action_value_loss"]
