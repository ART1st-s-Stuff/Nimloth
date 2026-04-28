"""Encoder 工厂函数。"""

from __future__ import annotations

from typing import Any

from src.wm.encoder.dino import DinoV2MiniEncoder, TrainableDinoV2Encoder
from src.wm.encoder.qwen import QwenImageEncoder, TrainableQwenLatentAdapter, QwenLLMLatentEncoder


def build_wm_image_encoder(wm_cfg: Any) -> Any | None:
    """构建冻结的图像 encoder（用于推理/预计算）。

    冻结的 encoder 不区分来源（CFM/LeWM 训练后的权重通用）。

    Args:
        wm_cfg: WM 配置对象

    Returns:
        WMImageEncoder 实例，或 None
    """
    encoder_cfg = getattr(wm_cfg, "encoder", None)
    if encoder_cfg is None:
        return None

    encoder_name = str(getattr(encoder_cfg, "name", "none")).lower()
    latent_dim = int(getattr(wm_cfg, "latent_dim", 128))

    if encoder_name in {"none", ""}:
        return None

    # 冻结的 DINOv2 - 通用实现
    if encoder_name == "frozen_dinov2m":
        return DinoV2MiniEncoder(
            latent_dim=latent_dim,
            freeze_backbone=bool(getattr(encoder_cfg, "freeze_backbone", True)),
            image_size=int(getattr(encoder_cfg, "image_size", 224)),
            patch_size=int(getattr(encoder_cfg, "patch_size", 14)),
            num_patches=int(getattr(encoder_cfg, "num_patches", 0)) or None,
            model_name=str(getattr(encoder_cfg, "backbone_name", "dinov2_vits14")),
            token_strategy=str(getattr(encoder_cfg, "token_strategy", "patch_mean")),
        )

    # Qwen encoder - 基于 QwenVLMAdapter
    if "qwen" in encoder_name:
        model_name = str(getattr(encoder_cfg, "model_name", "Qwen/Qwen2.5-VL-8B-Instruct"))
        enabled = bool(getattr(encoder_cfg, "enabled", True))
        fallback_enabled = bool(getattr(encoder_cfg, "fallback_enabled", True))
        num_patches = int(getattr(encoder_cfg, "num_patches", 0)) or None
        token_strategy = str(getattr(encoder_cfg, "token_strategy", "patch_mean"))

        # Qwen LLM hidden state encoder
        if encoder_name == "qwen_llm":
            prompt_template = str(getattr(encoder_cfg, "prompt_template", "") or "")
            return QwenLLMLatentEncoder(
                latent_dim=latent_dim,
                name=encoder_name,
                model_name=model_name,
                enabled=enabled,
                fallback_enabled=fallback_enabled,
                prompt_template=prompt_template if prompt_template else None,
            )

        return QwenImageEncoder(
            latent_dim=latent_dim,
            name=encoder_name,
            model_name=model_name,
            enabled=enabled,
            fallback_enabled=fallback_enabled,
            num_patches=num_patches,
            token_strategy=token_strategy,
            encoder_embed_dim=int(getattr(encoder_cfg, "encoder_embed_dim", 0)) or None,
        )

    raise ValueError(f"未知 WM encoder 配置: {encoder_name}")


def build_trainable_image_encoder(wm_cfg: Any, train_cfg: Any | None = None) -> Any | None:
    """构建可微调的图像 encoder（方案2，用于 Phase2 训练）。

    可训练 encoder 需要区分来源（cfm_trainable vs lewm_trainable）以加载正确权重。

    Args:
        wm_cfg: WM 配置
        train_cfg: 训练配置（包含 SIGReg 参数）

    Returns:
        TrainableDinoV2Encoder 实例，或 None
    """
    encoder_cfg = getattr(wm_cfg, "encoder", None)
    if encoder_cfg is None:
        return None

    encoder_name = str(getattr(encoder_cfg, "name", "none")).lower()
    latent_dim = int(getattr(wm_cfg, "latent_dim", 128))

    # 可训练 encoder - Qwen latent adapter
    encoder_finetune_cfg = getattr(train_cfg, "encoder_finetune", {}) if train_cfg else {}
    encoder_finetune_enabled = bool(getattr(encoder_finetune_cfg, "enabled", False))
    if encoder_finetune_enabled and "qwen" in encoder_name:
        return TrainableQwenLatentAdapter(
            latent_dim=latent_dim,
            hidden_dim=int(getattr(encoder_finetune_cfg, "adapter_hidden_dim", latent_dim)),
            mode=str(getattr(encoder_finetune_cfg, "mode", "adapter_only")),
            trainable_blocks=int(getattr(encoder_finetune_cfg, "trainable_blocks", 0)),
            distill_teacher=str(getattr(encoder_finetune_cfg, "distill_teacher", "frozen_qwen")),
        )

    # 可训练 encoder - 需要区分来源
    if encoder_name == "cfm_trainable_dinov2m":
        source_prefix = "cfm_"
    elif encoder_name == "lewm_trainable_dinov2m":
        source_prefix = "lewm_"
    else:
        return None

    sigreg_cfg = getattr(train_cfg, "sigreg", {}) if train_cfg else {}
    sigreg_enabled = bool(getattr(sigreg_cfg, "enabled", False))

    return TrainableDinoV2Encoder(
        latent_dim=latent_dim,
        freeze_backbone=bool(getattr(encoder_cfg, "freeze_backbone", False)),
        sigreg_enabled=sigreg_enabled,
        sigreg_latent_dim=int(getattr(sigreg_cfg, "latent_dim", 0)) or None,
        sigreg_num_quadrature_points=int(getattr(sigreg_cfg, "num_quadrature_points", 16)),
        sigreg_num_proj=int(getattr(sigreg_cfg, "num_projections", 256)),
        sigreg_t_min=float(getattr(sigreg_cfg, "t_min", 0.2)),
        sigreg_t_max=float(getattr(sigreg_cfg, "t_max", 4.0)),
        sigreg_kernel_sigma=float(getattr(sigreg_cfg, "kernel_sigma", 1.0)),
        image_size=int(getattr(encoder_cfg, "image_size", 224)),
        patch_size=int(getattr(encoder_cfg, "patch_size", 14)),
        num_patches=int(getattr(encoder_cfg, "num_patches", 0)) or None,
        model_name=str(getattr(encoder_cfg, "backbone_name", "dinov2_vits14")),
        token_strategy=str(getattr(encoder_cfg, "token_strategy", "patch_tokens")),
        source_prefix=source_prefix,
    )