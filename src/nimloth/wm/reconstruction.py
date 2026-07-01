"""Post-hoc image decoder for WM latent reconstruction diagnostics.

The decoder is intentionally separate from the SFT2/RL training objectives.  It
is trained after a WM checkpoint is produced, then used to visualize and score
whether true and predicted WM states retain image-level information.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from einops import rearrange
from torch import nn


@dataclass
class WMImageDecoderConfig:
    """Configuration for a self-attention patch decoder.

    The 1024-dim state vector is linearly expanded to all patch positions,
    added to learnable positional embeddings, then processed through
    self-attention layers before being projected to RGB patches.
    """

    emb_dim: int = 1024
    image_size: int = 255  # matches VAGEN navigation render resolution
    patch_size: int = 15
    hidden_dim: int = 1024
    depth: int = 4
    heads: int = 16
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")


class _SelfAttentionBlock(nn.Module):
    """Standard ViT block: self-attention + MLP with pre-norm residuals."""

    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
        )
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class WMImageDecoder(nn.Module):
    """Decode a WM state embedding into an RGB image.

    Architecture: state vector -> linear expand to all patches ->
    positional encoding -> self-attention layers -> RGB patch projection.
    """

    def __init__(self, config: WMImageDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or WMImageDecoderConfig()
        cfg = self.config
        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid
        self.state_expand = nn.Linear(cfg.emb_dim, self.num_patches * cfg.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, cfg.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_SelfAttentionBlock(cfg.hidden_dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.patch_head = nn.Linear(cfg.hidden_dim, cfg.patch_size * cfg.patch_size * 3)

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Return reconstructed images in ``[0, 1]`` with shape ``(B, 3, H, W)``."""

        if state_emb.ndim != 2:
            raise ValueError(f"state_emb must have shape (B, D), got {tuple(state_emb.shape)}")
        cfg = self.config
        b = state_emb.shape[0]
        x = self.state_expand(state_emb)  # (B, num_patches * hidden_dim)
        x = x.view(b, self.num_patches, cfg.hidden_dim)  # (B, N, D)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        patches = self.patch_head(self.norm(x))
        grid = cfg.image_size // cfg.patch_size
        images = rearrange(
            patches,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            gh=grid,
            gw=grid,
            ph=cfg.patch_size,
            pw=cfg.patch_size,
            c=3,
        )
        return torch.sigmoid(images)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "decoder.pt")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "WMImageDecoder":
        path = Path(path)
        cfg = WMImageDecoderConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        module = cls(cfg)
        state = torch.load(path / "decoder.pt", map_location=map_location, weights_only=True)
        module.load_state_dict(state)
        return module
