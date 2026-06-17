from __future__ import annotations

import argparse

from nimloth.training.common.qwen_tuning import resolve_tune_modes
from nimloth.training.common.vision_ema import resolve_vision_ema, vision_is_trainable


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
