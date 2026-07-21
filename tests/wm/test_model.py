"""完整 Nimloth 模型组合的结构测试。"""

from __future__ import annotations

import torch

from nimloth.model import NimlothModel
from nimloth.wm.model import WorldModel


class _Predictor(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return state + 1.0


def test_nimloth_model_owns_llm_wm_and_value_head() -> None:
    llm = torch.nn.Linear(3, 3)
    state_proj = torch.nn.Linear(3, 2)
    value_head = torch.nn.Linear(2, 4)
    model = NimlothModel(
        llm=llm,
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=_Predictor(),
            value_head=value_head,
        ),
    )

    assert model.llm is llm
    assert model.wm.state_proj is state_proj
    assert model.wm.value_head is value_head
    assert set(model.state_dict()) == {
        "llm.weight",
        "llm.bias",
        "wm.state_proj.weight",
        "wm.state_proj.bias",
        "wm.value_head.weight",
        "wm.value_head.bias",
    }


def test_world_model_loss_methods_use_owned_modules() -> None:
    model = WorldModel(
        state_proj=torch.nn.Linear(3, 2, bias=False),
        wm_predictor=_Predictor(),
        value_head=torch.nn.Linear(2, 3, bias=False),
    )
    hidden = torch.randn(2, 3)
    state = model.project_state(hidden)
    actions = torch.tensor([0, 2])
    output = model(hidden, actions)

    dynamics = model.compute_dynamics_loss(
        current_state=state,
        target_next_state=torch.zeros_like(state),
        action_indices=actions,
    )
    value = model.compute_action_value_loss(
        state=state,
        action_indices=actions,
        return_targets=torch.tensor([1.0, -1.0]),
    )

    assert dynamics.loss.ndim == 0
    assert value.loss.ndim == 0
    assert output["state"].shape == (2, 2)
    assert output["predicted_next_state"].shape == (2, 2)
    assert output["action_values"].shape == (2, 3)
