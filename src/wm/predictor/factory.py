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
    if wm_type == "cfm":
        return CFMWorldModel(
            latent_dim=int(getattr(wm_cfg, "latent_dim", 128)),
            action_dim=int(getattr(wm_cfg, "action_dim", 3)),
            hidden_dim=int(getattr(wm_cfg, "hidden_dim", 256)),
            history_len=int(getattr(wm_cfg, "history_len", 4)),
            num_patches=int(getattr(wm_cfg, "encoder", {}).get("num_patches", 16)),
            token_dim=int(getattr(wm_cfg, "latent_dim", 128)) // int(getattr(wm_cfg, "encoder", {}).get("num_patches", 16)),
            num_layers=int(getattr(wm_cfg, "transformer", {}).get("num_layers", 4)),
            num_heads=int(getattr(wm_cfg, "transformer", {}).get("num_heads", 4)),
            dropout=float(getattr(wm_cfg, "transformer", {}).get("dropout", 0.1)),
            conditioning_mode=str(getattr(wm_cfg, "conditioning", {}).get("mode", "adaln")),
            action_input_mode=str(getattr(wm_cfg, "action_injection", {}).get("mode", "adaln")),
            flow_matching_variant=str(getattr(wm_cfg, "flow_matching", {}).get("variant", "rectified_flow")),
            solver=str(getattr(wm_cfg, "flow_matching", {}).get("solver", "heun")),
            num_integration_steps=int(getattr(wm_cfg, "flow_matching", {}).get("num_steps", 16)),
            t_eps=float(getattr(wm_cfg, "flow_matching", {}).get("t_eps", 0.001)),
            sigreg_enabled=bool(getattr(wm_cfg, "sigreg", {}).get("enabled", False)),
            sigreg_latent_dim=int(getattr(wm_cfg, "sigreg", {}).get("latent_dim", 0)) or None,
            sigreg_encoder_hidden_dim=int(getattr(wm_cfg, "sigreg", {}).get("encoder_hidden_dim", 0)) or None,
            sigreg_encoder_num_layers=int(getattr(wm_cfg, "sigreg", {}).get("encoder_num_layers", 2)),
        )
    elif wm_type == "lewm":
        return LeWMWorldModel(
            latent_dim=int(getattr(wm_cfg, "latent_dim", 128)),
            action_dim=int(getattr(wm_cfg, "action_dim", 3)),
            hidden_dim=int(getattr(wm_cfg, "hidden_dim", 256)),
            history_len=int(getattr(wm_cfg, "history_len", 4)),
            num_patches=int(getattr(wm_cfg, "encoder", {}).get("num_patches", 16)),
            token_dim=int(getattr(wm_cfg, "latent_dim", 128)) // int(getattr(wm_cfg, "encoder", {}).get("num_patches", 16)),
            num_layers=int(getattr(wm_cfg, "transformer", {}).get("num_layers", 4)),
            num_heads=int(getattr(wm_cfg, "transformer", {}).get("num_heads", 4)),
            dim_head=int(getattr(wm_cfg, "transformer", {}).get("dim_head", 64)),
            mlp_ratio=float(getattr(wm_cfg, "lewm", {}).get("mlp_ratio", 4.0)),
            dropout=float(getattr(wm_cfg, "transformer", {}).get("dropout", 0.1)),
            emb_dropout=float(getattr(wm_cfg, "lewm", {}).get("emb_dropout", 0.0)),
            sigreg_enabled=bool(getattr(wm_cfg, "sigreg", {}).get("enabled", False)),
            sigreg_latent_dim=int(getattr(wm_cfg, "sigreg", {}).get("latent_dim", 0)) or None,
            sigreg_encoder_hidden_dim=int(getattr(wm_cfg, "sigreg", {}).get("encoder_hidden_dim", 0)) or None,
            sigreg_encoder_num_layers=int(getattr(wm_cfg, "sigreg", {}).get("encoder_num_layers", 2)),
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


def resolve_patch_layout(num_patches: int, latent_dim: int) -> tuple[int, int]:
    """验证并返回 patch 布局。

    Args:
        num_patches: patch 数量
        latent_dim: latent 总维度

    Returns:
        (num_patches, token_dim)

    Raises:
        ValueError: 如果配置不合法
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches 必须为正数: {num_patches}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim 必须为正数: {latent_dim}")
    if latent_dim % num_patches != 0:
        raise ValueError(f"latent_dim 必须能被 num_patches 整除: {latent_dim} % {num_patches} != 0")
    token_dim = latent_dim // num_patches
    return num_patches, token_dim