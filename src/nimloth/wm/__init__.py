"""World-model utilities for Nimloth (LeWM-style JEPA on navigation rollouts)."""

from nimloth.wm.dataset import TransitionSample, expand_record_transitions, load_jsonl_records
from nimloth.wm.lewm import LeWMConfig, LeWMWrapper, build_lewm, freeze_module
from nimloth.wm.preprocess import LeWMImageTransform, default_image_transform

__all__ = [
    "LeWMConfig",
    "LeWMWrapper",
    "LeWMImageTransform",
    "TransitionSample",
    "build_lewm",
    "default_image_transform",
    "expand_record_transitions",
    "freeze_module",
    "load_jsonl_records",
]
