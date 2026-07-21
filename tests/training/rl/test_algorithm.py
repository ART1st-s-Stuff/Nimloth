"""RL 核心算法的梯度 ownership 保护测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.training.rl.algorithm import RLAlgorithm, RLBatch
from nimloth.wm.model import WorldModel
from nimloth.wm.value_head import ValueHead


class _UnusedBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Identity()

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        return BackboneOutput(batch.tensors["hidden"])

    def with_model(self, model: torch.nn.Module) -> "_UnusedBackbone":
        return self

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


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
    agent = Agent(
        backbone=_UnusedBackbone(),
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=predictor,
            value_head=value_head,
        ),
    )
    optimizer = torch.optim.SGD(agent.wm.parameters(), lr=0.01)
    return (
        RLAlgorithm(
            agent=agent,
            optimizer=optimizer,
            device=torch.device("cpu"),
            vision_ema=None,
            policy_replay=None,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
            ppo_clip_ratio=0.2,
            entropy_weight=0.0,
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


def test_rl_wm_stops_gradient_on_next_projector_target() -> None:
    algorithm, state_proj, predictor, _ = _algorithm()
    batch = _batch()
    output = algorithm.training_step(batch)

    output.losses["wm"].backward()

    assert batch.current_hidden.grad is not None
    assert batch.next_hidden.grad is None
    assert state_proj.weight.grad is not None
    assert predictor.linear.weight.grad is not None


def test_rl_value_updates_head_but_not_state_projector() -> None:
    algorithm, state_proj, _, value_head = _algorithm()
    batch = _batch()
    output = algorithm.training_step(batch)

    output.losses["value"].backward()

    assert batch.current_hidden.grad is None
    assert state_proj.weight.grad is None
    assert value_head.net[0].weight.grad is not None
