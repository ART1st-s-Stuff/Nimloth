import math

import pytest
import torch

from experiments.training.sft.diagnosis.spatial_plateau_probe import (
    _layout,
    projector_gradient_relationship,
    split_components,
)


class SharedScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, value):
        return value * self.scale


def layout():
    return _layout(
        {
            "grid_tokens": 5,
            "state_layout": {
                "schema": "nimloth_grid_state_layout_v2",
                "spatial_grid_size": 2,
                "spatial_tokens": 4,
                "global_tokens": 1,
                "global_role": "dino_cls",
                "state_tokens": 5,
                "ordering": "row_major_spatial_then_global",
            },
        }
    )


def batch(spatial_target, cls_target):
    states = torch.ones(2, 5, 1)
    target = torch.full_like(states, spatial_target)
    target[:, 4] = cls_target
    return [(states, target, ["a", "b"])]


def test_split_components_uses_layout_not_token_mean():
    value = torch.arange(10).reshape(2, 5, 1).float()
    spatial, target_spatial, cls, target_cls = split_components(value, value + 1, layout())
    assert spatial.shape == target_spatial.shape == (2, 4, 1)
    assert cls.shape == target_cls.shape == (2, 1, 1)
    torch.testing.assert_close(cls[:, 0], value[:, 4])


@pytest.mark.parametrize(
    ("spatial_target", "cls_target", "expected"),
    [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0)],
)
def test_projector_gradient_relationship_reports_direction(
    spatial_target, cls_target, expected
):
    result = projector_gradient_relationship(
        SharedScale(),
        batch(spatial_target, cls_target),
        layout(),
        device=torch.device("cpu"),
        max_batches=1,
    )
    assert result["gradient_cosine"] == pytest.approx(expected)
    assert result["batch_gradient_cosine_mean"] == pytest.approx(expected)
    assert result["scope"] == "shared_projector_only_frozen_hidden_states"
    assert result["answers"] == 2


def test_layout_rejects_missing_or_non_cls_global_token():
    with pytest.raises(ValueError, match="state_layout"):
        _layout({"grid_tokens": 5})
    with pytest.raises(ValueError, match="DINO CLS"):
        _layout(
            {
                "grid_tokens": 4,
                "state_layout": {
                    "schema": "nimloth_grid_state_layout_v2",
                    "spatial_grid_size": 2,
                    "spatial_tokens": 4,
                    "global_tokens": 0,
                    "global_role": "none",
                    "state_tokens": 4,
                    "ordering": "row_major_spatial_then_global",
                },
            }
        )
