from __future__ import annotations

import torch
from torch import nn

from nimloth.backbone.qwen_tuning import enforce_frozen_tune_boundaries


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.q_proj = nn.Linear(2, 2)
        self.visual = nn.Module()
        self.visual.q_proj = nn.Linear(2, 2)
        self.other = nn.Parameter(torch.ones(()))


def test_vision_freeze_is_reapplied_after_suffix_matched_lora() -> None:
    model = FakeBackbone()
    model.requires_grad_(True)

    enforce_frozen_tune_boundaries(model, llm_tune="lora", vision_tune="freeze")

    assert all(parameter.requires_grad for parameter in model.language_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.visual.parameters())
    assert model.other.requires_grad


def test_language_freeze_does_not_freeze_visual_lora() -> None:
    model = FakeBackbone()
    model.requires_grad_(True)

    enforce_frozen_tune_boundaries(model, llm_tune="freeze", vision_tune="lora")

    assert all(not parameter.requires_grad for parameter in model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.visual.parameters())
