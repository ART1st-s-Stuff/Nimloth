"""World-model 神经网络模块的公共导出。"""

from nimloth.wm.lewm import LeWMConfig, action_one_hot, freeze_module
from nimloth.wm._vendor_lewm import SIGReg
from nimloth.wm.model import WorldModel
from nimloth.wm.predictor import (
    ONE_STEP_WM_HISTORY_SIZE,
    ONE_STEP_WM_PREDICTION_OFFSET,
    ONE_STEP_WM_SEQUENCE_LENGTH,
    LatentWMPredictor,
    require_one_step_wm_predictor,
)
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig
from nimloth.wm.sigreg import OneStepSIGReg
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead

__all__ = [
    "LatentWMPredictor",
    "LeWMConfig",
    "ONE_STEP_WM_HISTORY_SIZE",
    "ONE_STEP_WM_PREDICTION_OFFSET",
    "ONE_STEP_WM_SEQUENCE_LENGTH",
    "OneStepSIGReg",
    "SIGReg",
    "StateProjector",
    "ValueHead",
    "WMImageDecoder",
    "WMImageDecoderConfig",
    "WorldModel",
    "action_one_hot",
    "freeze_module",
    "require_one_step_wm_predictor",
]
