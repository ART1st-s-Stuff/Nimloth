"""世界模型（WM）相关模块。

模块结构：
- encoder/: 图像编码器（DINOv2, Qwen）
- predictor/: 世界模型预测器（CFM, LeWM）
- sigreg_modules.py: SIGReg 正则化模块
- inverse_dynamics.py: 逆动力学模型
- action_mapper.py: 动作映射器
- losses.py: 损失函数
"""

__all__ = [
    # Encoder
    "WMImageEncoder",
    "EncoderOutput",
    "build_wm_image_encoder",
    "build_trainable_image_encoder",
    # Predictor
    "CFMWorldModel",
    "WMModel",
    "LeWMWorldModel",
    "LeWMModel",
    # Factory
    "build_world_model",
    "resolve_patch_layout",
    "resolve_wm_type",
]


def __getattr__(name: str):
    if name in {
        "WMImageEncoder",
        "EncoderOutput",
        "build_wm_image_encoder",
        "build_trainable_image_encoder",
    }:
        from src.wm import encoder

        return getattr(encoder, name)
    if name in {"CFMWorldModel", "WMModel", "LeWMWorldModel", "LeWMModel"}:
        from src.wm import predictor

        return getattr(predictor, name)
    if name in {"build_world_model", "resolve_patch_layout", "resolve_wm_type"}:
        from src.wm.predictor import factory

        return getattr(factory, name)
    raise AttributeError(name)
