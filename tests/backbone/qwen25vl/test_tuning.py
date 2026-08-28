from __future__ import annotations

import argparse

from torch import nn

from nimloth.backbone.qwen25vl.tuning import (
    configure_qwen_tuning,
    is_llm_param,
    is_vision_param,
    resolve_tune_modes,
)
from nimloth.backbone.qwen25vl.vision_ema import resolve_vision_ema, vision_is_trainable


def test_resolve_tune_modes_legacy_lora() -> None:
    args = argparse.Namespace(lora=True, llm_tune="freeze", vision_tune="full")
    assert resolve_tune_modes(args) == ("lora", "freeze")


def test_vision_is_trainable() -> None:
    assert vision_is_trainable("full")
    assert vision_is_trainable("lora")
    assert not vision_is_trainable("freeze")


def test_resolve_vision_ema_defaults_on_for_full_vision() -> None:
    args = argparse.Namespace(vision_ema=None, no_vision_ema=False)
    assert resolve_vision_ema(args, "full") is True
    assert resolve_vision_ema(args, "freeze") is False


def test_resolve_vision_ema_explicit_disable() -> None:
    args = argparse.Namespace(vision_ema=None, no_vision_ema=True)
    assert resolve_vision_ema(args, "full") is False


def test_full_language_tuning_includes_top_level_lm_head_and_freezes_visual() -> None:
    class _FakeQwen(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Linear(4, 4)
            self.model.visual = nn.Module()
            self.model.visual.patch_embed = nn.Linear(4, 4)
            self.model.visual.merger = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)

    model = _FakeQwen()
    configured = configure_qwen_tuning(
        model,
        argparse.Namespace(
            lora=False,
            llm_tune="full",
            vision_tune="freeze",
        ),
    )
    trainable = {
        name for name, parameter in configured.named_parameters()
        if parameter.requires_grad
    }

    assert trainable == {
        "model.language_model.weight",
        "model.language_model.bias",
        "lm_head.weight",
    }
    assert is_llm_param("model.language_model.layers.0.mlp.up_proj.weight")
    assert is_llm_param("lm_head.weight")
    assert is_llm_param("wrapped.lm_head.weight")
    assert not is_llm_param("model.visual.merger.weight")
    assert is_vision_param("model.visual.merger.weight")
