"""Tests for Qwen checkpoint wrapper layout handling."""

from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.checkpoint import find_visual_module, save_full_vision_state


class WrappedQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Linear(2, 2)


def test_visual_module_lookup_and_save(tmp_path) -> None:
    model = WrappedQwen()
    assert find_visual_module(model) is model.model.visual

    path = tmp_path / "vision_full_state.pt"
    save_full_vision_state(model, path)

    assert path.is_file()
    assert torch.load(path, weights_only=True).keys() == model.model.visual.state_dict().keys()
