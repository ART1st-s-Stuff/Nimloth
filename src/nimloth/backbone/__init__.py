"""Agent backbone 的公共构造入口；训练代码不直接选择具体实现。"""

from nimloth.backbone.base import (
    Backbone,
    BackboneBatch,
    BackboneEMA,
    BackboneOutput,
    LoadedBackbone,
    RLBackboneAdapters,
)


def load_sft2_backbone(*args, **kwargs) -> LoadedBackbone:
    from nimloth.backbone.qwen25vl.factory import load_sft2_backbone as load
    return load(*args, **kwargs)


def load_rl_backbone(*args, **kwargs) -> LoadedBackbone:
    from nimloth.backbone.qwen25vl.factory import load_rl_backbone as load
    return load(*args, **kwargs)


def build_rl_adapters(*args, **kwargs) -> RLBackboneAdapters:
    from nimloth.backbone.qwen25vl.factory import build_rl_adapters as build
    return build(*args, **kwargs)


def build_sft2_batch_builder(*args, **kwargs):
    from nimloth.backbone.qwen25vl.factory import build_sft2_batch_builder as build
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
    "RLBackboneAdapters",
    "Backbone",
    "BackboneBatch",
    "BackboneEMA",
    "BackboneOutput",
    "backbone_hidden_size",
    "build_vision_ema",
    "build_rl_adapters",
    "build_sft2_batch_builder",
    "load_rl_backbone",
    "load_sft2_backbone",
    "resolve_tune_modes",
    "resolve_vision_ema",
    "uses_lora",
]
