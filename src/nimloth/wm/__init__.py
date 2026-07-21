"""World-model utilities for Nimloth."""

from nimloth.rollout.transitions import (
    TransitionSample,
    expand_record_transitions,
    load_jsonl_records,
)
from nimloth.wm.lewm import LeWMConfig, action_one_hot, freeze_module
from nimloth.wm._vendor_lewm import SIGReg
from nimloth.wm.objectives import (
    ActionValueLoss,
    DynamicsLoss,
    compute_action_value_loss,
    compute_dynamics_loss,
)
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead

__all__ = [
    "ActionValueLoss",
    "DynamicsLoss",
    "LatentWMPredictor",
    "LeWMConfig",
    "SIGReg",
    "StateProjector",
    "TransitionSample",
    "ValueHead",
    "WMImageDecoder",
    "WMImageDecoderConfig",
    "action_one_hot",
    "compute_action_value_loss",
    "compute_dynamics_loss",
    "expand_record_transitions",
    "freeze_module",
    "load_jsonl_records",
]
