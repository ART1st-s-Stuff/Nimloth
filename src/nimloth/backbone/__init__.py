"""Agent backbone 的公共构造入口；训练代码不直接选择具体实现。"""

from nimloth.backbone.base import (
    Backbone,
    BackboneBatch,
    BackboneEMA,
    BackboneInputBuilder,
    BackboneOutput,
    LoadedBackbone,
)
from nimloth.backbone.dino_grid import (
    CachedDINOGridTargets,
    DINOIdentity,
    DINOV2_LARGE_IDENTITY,
)


def load_backbone(*args, **kwargs) -> LoadedBackbone:
    from nimloth.backbone.qwen25vl.factory import load_backbone as load
    return load(*args, **kwargs)


def build_input_builder(*args, **kwargs) -> BackboneInputBuilder:
    from nimloth.backbone.qwen25vl.factory import build_input_builder as build
    return build(*args, **kwargs)


def build_agent_policy(*args, **kwargs):
    from nimloth.backbone.qwen25vl.factory import build_agent_policy as build
    return build(*args, **kwargs)


def build_action_log_prob_replay(*args, **kwargs):
    from nimloth.backbone.qwen25vl.factory import build_action_log_prob_replay as build
    return build(*args, **kwargs)


def build_vision_ema(*args, **kwargs):
    from nimloth.backbone.qwen25vl.factory import build_vision_ema as build
    return build(*args, **kwargs)


def resolve_tune_modes(args):
    from nimloth.backbone.qwen25vl.tuning import resolve_tune_modes as resolve
    return resolve(args)


def uses_lora(args) -> bool:
    from nimloth.backbone.qwen25vl.tuning import uses_lora as resolve
    return resolve(args)


def resolve_vision_ema(args, vision_tune: str) -> bool:
    from nimloth.backbone.qwen25vl.vision_ema import resolve_vision_ema as resolve
    return resolve(args, vision_tune)


def backbone_hidden_size(config) -> int:
    from nimloth.backbone.qwen25vl.loading import qwen_hidden_size
    return qwen_hidden_size(config)


__all__ = [
    "LoadedBackbone",
    "Backbone",
    "BackboneBatch",
    "BackboneEMA",
    "BackboneInputBuilder",
    "BackboneOutput",
    "CachedDINOGridTargets",
    "DINOIdentity",
    "DINOV2_LARGE_IDENTITY",
    "backbone_hidden_size",
    "build_action_log_prob_replay",
    "build_agent_policy",
    "build_input_builder",
    "build_vision_ema",
    "load_backbone",
    "resolve_tune_modes",
    "resolve_vision_ema",
    "uses_lora",
]
