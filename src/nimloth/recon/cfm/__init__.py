"""Conditional flow matching for post-hoc latent-state visualization."""

from .flow import (
    condition_sensitivity,
    conditional_flow_matching_loss,
    sample_euler,
    sample_euler_cfg,
    spatial_cls_condition_sensitivity,
    spatial_cls_condition_variants,
)
from .model import (
    CFMConfig,
    SpatialCLSCFMConfig,
    SpatialCLSConditionedFlowUNet,
    SpatialConditionedFlowUNet,
    TokenConditionedFlowUNet,
)

__all__ = [
    "CFMConfig",
    "SpatialCLSCFMConfig",
    "SpatialCLSConditionedFlowUNet",
    "SpatialConditionedFlowUNet",
    "TokenConditionedFlowUNet",
    "condition_sensitivity",
    "conditional_flow_matching_loss",
    "sample_euler",
    "sample_euler_cfg",
    "spatial_cls_condition_sensitivity",
    "spatial_cls_condition_variants",
]
