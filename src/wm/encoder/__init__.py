"""WM Encoder 模块。

提供图像编码器构建与特征提取功能。
"""

from src.wm.encoder.base import EncoderOutput, WMImageEncoder
from src.wm.encoder.factory import build_trainable_image_encoder, build_wm_image_encoder

__all__ = [
    "EncoderOutput",
    "WMImageEncoder",
    "build_wm_image_encoder",
    "build_trainable_image_encoder",
]