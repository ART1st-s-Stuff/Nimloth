"""WM 构建工厂：统一 CFM/LeWM 的实例化逻辑。"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf

from src.wm.lewm import LeWMWorldModel
from src.wm.model import CFMWorldModel


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """兼容 dict/DictConfig 的安全读取。"""
    if cfg is None:
        return default
    if OmegaConf.is_config(cfg):
        value = OmegaConf.select(cfg, key, default=default)
        return default if value is None else value
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_patch_layout(wm_cfg: Any, *, allow_zero: bool = False) -> tuple[int, int]:
    """解析 patch 布局并校验 latent 维度。"""
    encoder_cfg = _cfg_get(wm_cfg, "encoder", {})
    num_patches = int(_cfg_get(encoder_cfg, "num_patches", 0))
    latent_dim = int(_cfg_get(wm_cfg, "latent_dim", 0))
    if num_patches <= 0:
        if allow_zero:
            return 0, 0
        raise ValueError("wm.encoder.num_patches 必须为正整数。")
    if latent_dim % num_patches != 0:
        raise ValueError(f"wm.latent_dim 必须能被 num_patches 整除: {latent_dim} / {num_patches}")
    return num_patches, latent_dim // num_patches


def resolve_wm_type(wm_cfg: Any) -> str:
    """优先使用显式 wm.type，其次退化到 wm.name。"""
    wm_type_raw = str(_cfg_get(wm_cfg, "type", "")).strip().lower()
    if wm_type_raw:
        if wm_type_raw in {"cfm", "lewm"}:
            return wm_type_raw
        raise ValueError(f"不支持的 wm.type={wm_type_raw}，仅支持 cfm/lewm。")
    wm_name = str(_cfg_get(wm_cfg, "name", "")).strip().lower()
    if "lewm" in wm_name:
        return "lewm"
    if wm_name.startswith("cfm"):
        return "cfm"
    raise ValueError(f"无法从 wm 配置推断模型类型: wm.type={wm_type_raw}, wm.name={wm_name}")


def build_world_model(
    *,
    wm_cfg: Any,
    train_cfg: Any | None,
    action_dim: int,
    device: torch.device,
    allow_zero_patches: bool = False,
) -> torch.nn.Module:
    """基于配置创建 world model。"""
    wm_type = resolve_wm_type(wm_cfg)
    num_patches, token_dim = resolve_patch_layout(wm_cfg=wm_cfg, allow_zero=allow_zero_patches)
    latent_dim = int(_cfg_get(wm_cfg, "latent_dim", 0))

    if wm_type == "lewm":
        lewm_cfg = _cfg_get(wm_cfg, "lewm", {})
        transformer_cfg = _cfg_get(wm_cfg, "transformer", {})
        sigreg_cfg = _cfg_get(train_cfg, "sigreg", {}) if train_cfg else {}
        return LeWMWorldModel(
            latent_dim=latent_dim,
            action_dim=int(action_dim),
            hidden_dim=int(_cfg_get(wm_cfg, "hidden_dim", 0)),
            history_len=int(_cfg_get(wm_cfg, "history_len", 0)),
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=int(_cfg_get(transformer_cfg, "num_layers", 0)),
            num_heads=int(_cfg_get(transformer_cfg, "num_heads", 0)),
            dim_head=int(_cfg_get(lewm_cfg, "dim_head", 64)),
            mlp_ratio=float(_cfg_get(lewm_cfg, "mlp_ratio", 4.0)),
            dropout=float(_cfg_get(transformer_cfg, "dropout", 0.0)),
            emb_dropout=float(_cfg_get(lewm_cfg, "emb_dropout", 0.0)),
            sigreg_knots=int(_cfg_get(lewm_cfg, "sigreg_knots", 17)),
            sigreg_num_proj=int(_cfg_get(lewm_cfg, "sigreg_num_proj", 256)),
            sigreg_num_quadrature_points=int(_cfg_get(sigreg_cfg, "num_quadrature_points", _cfg_get(lewm_cfg, "sigreg_knots", 16))),
            sigreg_t_min=float(_cfg_get(sigreg_cfg, "t_min", 0.2)),
            sigreg_t_max=float(_cfg_get(sigreg_cfg, "t_max", 4.0)),
            sigreg_kernel_sigma=float(_cfg_get(sigreg_cfg, "kernel_sigma", 1.0)),
        ).to(device)

    train_flow_cfg = {}
    if train_cfg is not None:
        train_flow_cfg = _cfg_get(train_cfg, "flow_matching", {})
    flow_cfg = _cfg_get(wm_cfg, "flow_matching", train_flow_cfg)
    conditioning_cfg = _cfg_get(wm_cfg, "conditioning", {})
    action_cfg = _cfg_get(wm_cfg, "action_injection", {})
    history_cfg = _cfg_get(wm_cfg, "history_injection", {})
    transformer_cfg = _cfg_get(wm_cfg, "transformer", {})
    history_mode = str(_cfg_get(history_cfg, "mode", "cross_attention")).strip().lower()
    if history_mode != "cross_attention":
        raise ValueError(f"当前仅支持 history_injection.mode=cross_attention，实际为 {history_mode}")
    history_direction = str(
        _cfg_get(history_cfg, "cross_attention_direction", "query_xt_key_history")
    ).strip().lower()
    if history_direction != "query_xt_key_history":
        raise ValueError(
            "当前仅支持 history_injection.cross_attention_direction=query_xt_key_history"
        )
    return CFMWorldModel(
        latent_dim=latent_dim,
        action_dim=int(action_dim),
        hidden_dim=int(_cfg_get(wm_cfg, "hidden_dim", 0)),
        history_len=int(_cfg_get(wm_cfg, "history_len", 0)),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(_cfg_get(transformer_cfg, "num_layers", 0)),
        num_heads=int(_cfg_get(transformer_cfg, "num_heads", 0)),
        dropout=float(_cfg_get(transformer_cfg, "dropout", 0.0)),
        conditioning_mode=str(_cfg_get(conditioning_cfg, "mode", "adaln")),
        action_input_mode=str(_cfg_get(action_cfg, "mode", _cfg_get(conditioning_cfg, "action_input_mode", "adaln"))),
        flow_matching_variant=str(_cfg_get(flow_cfg, "variant", "rectified_flow")),
        x0_source=str(_cfg_get(flow_cfg, "x0_source", "current_latent")),
        solver=str(_cfg_get(flow_cfg, "solver", "heun")),
        num_integration_steps=int(_cfg_get(flow_cfg, "num_steps", 16)),
        t_eps=float(_cfg_get(flow_cfg, "t_eps", 1e-3)),
        noise_std=float(_cfg_get(flow_cfg, "noise_std", 1.0)),
    ).to(device)
