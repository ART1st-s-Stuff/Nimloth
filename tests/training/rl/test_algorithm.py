"""RL 核心算法的梯度 ownership 保护测试。"""

from __future__ import annotations

import torch

from nimloth.config.rl import parse_rl_config
from nimloth.training.rl.algorithm import RLAlgorithm, RLBatch
from nimloth.training.rl.components import RLComponents, RLResumeState
from nimloth.wm.value_head import ValueHead


class _Predictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(state)


def _algorithm() -> tuple[RLAlgorithm, torch.nn.Linear, _Predictor, ValueHead]:
    state_proj = torch.nn.Linear(3, 2, bias=False)
    predictor = _Predictor()
    value_head = ValueHead(emb_dim=2, num_actions=3, hidden_dim=2)
    optimizer = torch.optim.AdamW(
        [
            *state_proj.parameters(),
            *predictor.parameters(),
            *value_head.parameters(),
        ]
    )
    components = RLComponents(
        model=torch.nn.Identity(),
        processor=None,
        token_id_map={},
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=value_head,
        vision_ema=None,
        optimizer=optimizer,
        base_model_path="unused",
        llm_tune="freeze",
        vision_tune="freeze",
        resume=RLResumeState(),
    )
    config = parse_rl_config(
        {
            "freeze": {"state_proj": False},
            "predictor": {"emb_dim": 2, "history_size": 1},
            "value_head": {"lambda_rank": 1.0},
            "rollout": {
                "train_datasets": ["base_train"],
                "eval_datasets": ["base"],
            },
        }
    )
    return (
        RLAlgorithm(
            components=components,
            config=config,
            actor_enabled=False,
            device=torch.device("cpu"),
        ),
        state_proj,
        predictor,
        value_head,
    )


def _batch() -> RLBatch:
    return RLBatch(
        transitions=(),
        current_hidden=torch.randn(2, 3, requires_grad=True),
        next_hidden=torch.randn(2, 3, requires_grad=True),
        action_indices=torch.tensor([0, 2]),
        return_targets=torch.tensor([1.0, -0.5]),
        old_log_probs=torch.zeros(2),
    )


def test_rl_dynamics_stops_gradient_on_next_projector_target() -> None:
    algorithm, state_proj, predictor, _ = _algorithm()
    batch = _batch()
    losses = algorithm.compute_losses(batch)

    losses.dynamics.loss.backward()

    assert batch.current_hidden.grad is not None
    assert batch.next_hidden.grad is None
    assert state_proj.weight.grad is not None
    assert predictor.linear.weight.grad is not None


def test_rl_value_updates_head_but_not_state_projector() -> None:
    algorithm, state_proj, _, value_head = _algorithm()
    batch = _batch()
    losses = algorithm.compute_losses(batch)

    losses.value.loss.backward()

    assert batch.current_hidden.grad is None
    assert state_proj.weight.grad is None
    assert value_head.net[0].weight.grad is not None
