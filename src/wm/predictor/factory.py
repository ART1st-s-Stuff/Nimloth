"""Predictor 工厂函数。"""

from __future__ import annotations

from typing import Any


def build_world_model(wm_cfg: Any) -> Any:
    """根据配置构建世界模型。

    Args:
        wm_cfg: WM 配置对象

    Returns:
        CFMWorldModel 或 LeWMWorldModel 实例
    """
    from src.wm.predictor.cfm import CFMWorldModel
    from src.wm.predictor.lewm import LeWMWorldModel

    wm_type = resolve_wm_type(wm_cfg)
    latent_dim = int(getattr(wm_cfg, "latent_dim", 128))
    # 从配置读取 num_patches 和 token_dim（Qwen LLM 模式可能设置为 1）
    num_patches = int(getattr(wm_cfg, "num_patches", 0)) or int(getattr(wm_cfg.encoder, "num_patches", 16))
    token_dim = int(getattr(wm_cfg, "token_dim", 0))
    if token_dim <= 0:
        token_dim = latent_dim // num_patches
    if wm_type == "cfm":
        sigreg_cfg = getattr(wm_cfg, "sigreg", {})
        return CFMWorldModel(
            latent_dim=latent_dim,
            action_dim=int(getattr(wm_cfg, "action_dim", 3)),
            hidden_dim=int(getattr(wm_cfg, "hidden_dim", 256)),
            history_len=int(getattr(wm_cfg, "history_len", 4)),
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=int(getattr(wm_cfg, "transformer", {}).get("num_layers", 4)),
            num_heads=int(getattr(wm_cfg, "transformer", {}).get("num_heads", 4)),
            dropout=float(getattr(wm_cfg, "transformer", {}).get("dropout", 0.1)),
            conditioning_mode=str(getattr(wm_cfg, "conditioning", {}).get("mode", "adaln")),
            action_input_mode=str(getattr(wm_cfg, "action_injection", {}).get("mode", "adaln")),
            flow_matching_variant=str(getattr(wm_cfg, "flow_matching", {}).get("variant", "rectified_flow")),
            solver=str(getattr(wm_cfg, "flow_matching", {}).get("solver", "heun")),
            num_integration_steps=int(getattr(wm_cfg, "flow_matching", {}).get("num_steps", 16)),
            t_eps=float(getattr(wm_cfg, "flow_matching", {}).get("t_eps", 0.001)),
            sigreg_enabled=bool(getattr(sigreg_cfg, "enabled", False)),
            sigreg_latent_dim=int(getattr(sigreg_cfg, "latent_dim", 0)) or None,
            sigreg_encoder_hidden_dim=int(getattr(sigreg_cfg, "encoder_hidden_dim", 0)) or None,
            sigreg_encoder_num_layers=int(getattr(sigreg_cfg, "encoder_num_layers", 2)),
        )
    elif wm_type == "lewm":
        return LeWMWorldModel(
            latent_dim=latent_dim,
            action_dim=int(getattr(wm_cfg, "action_dim", 3)),
            hidden_dim=int(getattr(wm_cfg, "hidden_dim", 256)),
            history_len=int(getattr(wm_cfg, "history_len", 4)),
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=int(getattr(wm_cfg, "transformer", {}).get("num_layers", 4)),
            num_heads=int(getattr(wm_cfg, "transformer", {}).get("num_heads", 4)),
            dim_head=int(getattr(wm_cfg, "transformer", {}).get("dim_head", 64)),
            mlp_ratio=float(getattr(wm_cfg, "lewm", {}).get("mlp_ratio", 4.0)),
            dropout=float(getattr(wm_cfg, "transformer", {}).get("dropout", 0.1)),
            emb_dropout=float(getattr(wm_cfg, "lewm", {}).get("emb_dropout", 0.0)),
            sigreg_enabled=bool(getattr(wm_cfg, "lewm", {}).get("sigreg_enabled", False)),
            sigreg_latent_dim=int(getattr(wm_cfg, "lewm", {}).get("sigreg_latent_dim", 0)) or None,
            sigreg_encoder_hidden_dim=int(getattr(wm_cfg, "lewm", {}).get("sigreg_encoder_hidden_dim", 0)) or None,
            sigreg_encoder_num_layers=int(getattr(wm_cfg, "lewm", {}).get("sigreg_encoder_num_layers", 2)),
        )
    else:
        raise ValueError(f"未知的 WM 类型: {wm_type}")


def resolve_wm_type(wm_cfg: Any) -> str:
    """从配置推断 WM 类型。

    Args:
        wm_cfg: WM 配置对象

    Returns:
        "cfm" 或 "lewm"
    """
    wm_name = str(getattr(wm_cfg, "name", "")).lower()
    if "lewm" in wm_name:
        return "lewm"
    elif "cfm" in wm_name or "frozen" in wm_name:
        return "cfm"
    return "cfm"


def resolve_patch_layout(
    num_patches: int | None = None,
    latent_dim: int | None = None,
    wm_cfg: Any = None,
    allow_zero: bool = False,
) -> tuple[int, int]:
    """验证并返回 patch 布局。

    支持两种调用方式：
    1. resolve_patch_layout(num_patches=4, latent_dim=4096)
    2. resolve_patch_layout(wm_cfg=cfg) - 从配置对象读取

    Args:
        num_patches: patch 数量
        latent_dim: latent 总维度
        wm_cfg: 配置对象（从中读取 num_patches 和 latent_dim）
        allow_zero: 是否允许 num_patches=0（返回默认值）

    Returns:
        (num_patches, token_dim)

    Raises:
        ValueError: 如果配置不合法
    """
    if wm_cfg is not None:
        num_patches = int(getattr(wm_cfg, "num_patches", 0)) or int(getattr(wm_cfg.encoder, "num_patches", 0)) if hasattr(wm_cfg, "encoder") else int(getattr(wm_cfg, "num_patches", 0))
        latent_dim = int(getattr(wm_cfg, "latent_dim", 0))

    if num_patches is None:
        num_patches = 0
    if latent_dim is None:
        latent_dim = 0

    num_patches = int(num_patches)
    latent_dim = int(latent_dim)

    if num_patches <= 0:
        if allow_zero:
            num_patches = 1
        else:
            raise ValueError(f"num_patches 必须为正数: {num_patches}")
    if latent_dim <= 0:
        if allow_zero:
            return num_patches, 0
        raise ValueError(f"latent_dim 必须为正数: {latent_dim}")
    if latent_dim % num_patches != 0:
        raise ValueError(f"latent_dim 必须能被 num_patches 整除: {latent_dim} % {num_patches} != 0")
    token_dim = latent_dim // num_patches
    return num_patches, token_dim