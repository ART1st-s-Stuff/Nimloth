"""SFT2 Agent forward、objective 与梯度语义测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent, AgentBatch, AgentTarget
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.objective import (
    SFT2Objective,
    build_trajectory_sigreg_inputs,
)
from nimloth.training.sft2.schedule import wm_loss_weight_schedule
from nimloth.wm import StateProjector, ValueHead, WorldModel


class _TensorBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Identity()
        self.calls = 0

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        self.calls += 1
        hidden = self.language_model(batch.tensors["hidden"])
        return BackboneOutput(
            hidden=hidden,
            lm_loss=hidden.mean() if include_lm_loss else None,
        )

    def with_model(self, model: torch.nn.Module) -> "_TensorBackbone":
        return self

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _Predictor(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = torch.nn.Linear(dimension, dimension, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(state)


class _RecordingProjector(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4, bias=False)
        self.outputs: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = super().forward(hidden)
        output.retain_grad()
        self.outputs.append(output)
        return output


def _algorithm(sigreg: torch.nn.Module | None = None):
    backbone = _TensorBackbone()
    projector = _RecordingProjector()
    agent = Agent(
        backbone=backbone,
        wm=WorldModel(
            state_proj=projector,
            wm_predictor=_Predictor(4),
            value_head=ValueHead(emb_dim=4, num_actions=3),
        ),
    )
    return (
        SFT2Algorithm(
            agent=agent,
            target=AgentTarget(agent),
            objective=SFT2Objective(
                sigreg=sigreg,
                sigreg_weight=0.1,
                value_weight=1.0,
                ce_weight=1.0,
                value_rank_margin=0.1,
                value_rank_weight=1.0,
            ),
        ),
        backbone,
        projector,
    )


def _batch() -> AgentBatch:
    return AgentBatch(
        current=BackboneBatch({"hidden": torch.randn(2, 4, requires_grad=True)}),
        next=BackboneBatch({"hidden": torch.randn(2, 4, requires_grad=True)}),
        action_indices=torch.tensor([0, 2]),
        value_targets=torch.tensor([1.0, -0.5]),
        next_indices=torch.tensor([0, 1]),
        non_terminal_mask=torch.tensor([True, True]),
        trajectory_steps=(("rec_A", 0), ("rec_A", 1)),
    )


def test_state_projector_accepts_multi_latent_block() -> None:
    state_proj = StateProjector(
        qwen_hidden_dim=8,
        lewm_emb_dim=4,
        projector_hidden_dim=16,
        latent_token_count=3,
    )
    out = state_proj(torch.randn(2, 3, 8))
    assert out.shape == (2, 4)
    assert state_proj.input_dim == 24


def test_sft2_training_step_uses_agent_forward_and_two_sided_projector_gradient() -> None:
    algorithm, backbone, projector = _algorithm()
    batch = _batch()

    output = algorithm.training_step(batch, wm_weight=0.5)
    output.losses["wm"].backward()

    assert backbone.calls == 2
    assert output.model_output.lm_loss is not None
    assert projector.outputs[0].grad is not None
    assert projector.outputs[1].grad is not None
    assert batch.current.tensors["hidden"].grad is not None
    assert batch.next.tensors["hidden"].grad is None


def test_sft2_sigreg_is_computed_per_trajectory() -> None:
    class MockSIGReg(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert value.dim() == 3
            return value.pow(2).mean()

    algorithm, _, _ = _algorithm(MockSIGReg())
    output = algorithm.training_step(_batch(), wm_weight=1.0)
    assert output.losses["sigreg"] is not None
    assert output.metrics["sigreg_loss"] > 0.0


def test_build_trajectory_sigreg_inputs() -> None:
    current = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    )
    target = current + 0.1
    result = build_trajectory_sigreg_inputs(
        [("rec_A", 0), ("rec_A", 1), ("rec_B", 0), ("rec_C", 3)],
        current,
        target,
    )
    assert sorted(tuple(value.shape) for value in result) == [
        (2, 1, 2),
        (2, 1, 2),
        (3, 1, 2),
    ]


def test_wm_loss_weight_schedule_warms_up() -> None:
    assert wm_loss_weight_schedule(0, 100, start=0.1, end=1.0) == 0.1
    assert 0.1 < wm_loss_weight_schedule(15, 100, start=0.1, end=1.0) < 1.0
    assert wm_loss_weight_schedule(60, 100, start=0.1, end=1.0) == 1.0
