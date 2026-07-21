"""完整 Agent 与 WorldModel 模块边界测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.wm.model import WorldModel


class _Backbone(Backbone):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.language_model = model

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        hidden = self.language_model(batch.tensors["features"])
        loss = hidden.mean() if include_lm_loss else None
        return BackboneOutput(hidden=hidden, lm_loss=loss)

    def with_model(self, model: torch.nn.Module) -> "_Backbone":
        return _Backbone(model)

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _Predictor(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return state + 1.0


def test_agent_owns_backbone_wm_and_runs_complete_forward() -> None:
    language_model = torch.nn.Linear(3, 3)
    state_proj = torch.nn.Linear(3, 2)
    value_head = torch.nn.Linear(2, 4)
    agent = Agent(
        backbone=_Backbone(language_model),
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=_Predictor(),
            value_head=value_head,
        ),
    )

    output = agent(
        BackboneBatch({"features": torch.randn(2, 3)}),
        torch.tensor([0, 1]),
        include_lm_loss=True,
    )

    assert agent.backbone.model is language_model
    assert agent.wm.state_proj is state_proj
    assert output.hidden.shape == (2, 3)
    assert output.state.shape == (2, 2)
    assert output.predicted_next_state.shape == (2, 2)
    assert output.action_values.shape == (2, 4)
    assert output.lm_loss is not None
    assert set(agent.state_dict()) == {
        "backbone.language_model.weight",
        "backbone.language_model.bias",
        "wm.state_proj.weight",
        "wm.state_proj.bias",
        "wm.value_head.weight",
        "wm.value_head.bias",
    }


def test_world_model_forward_uses_all_owned_modules() -> None:
    model = WorldModel(
        state_proj=torch.nn.Linear(3, 2, bias=False),
        wm_predictor=_Predictor(),
        value_head=torch.nn.Linear(2, 3, bias=False),
    )
    output = model(torch.randn(2, 3), torch.tensor([0, 2]))

    assert output["state"].shape == (2, 2)
    assert output["predicted_next_state"].shape == (2, 2)
    assert output["action_values"].shape == (2, 3)
