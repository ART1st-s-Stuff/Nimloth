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
    """Configuration for a patch decoder (cross-attention from a single latent vector).

    Scaled to match Qwen-latent dimension: 1024-dim input, 255x255 output,
    multiple cross-attention layers (following LeWM paper decoder description).
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


class _DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn, _ = self.cross_attn(
            self.norm_q(query),
            self.norm_kv(memory),
            self.norm_kv(memory),
            need_weights=False,
        )
        query = query + attn
        query = query + self.mlp(self.norm_mlp(query))
        return query


class WMImageDecoder(nn.Module):
    """Decode a WM state embedding into an RGB image.

    Mirrors the LeWM paper's diagnostic decoder: learned patch queries
    cross-attend to a single global latent vector and are projected to
    RGB patches.  Trained post-hoc with frozen Qwen/WM modules.
    """

    def __init__(self, config: WMImageDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or WMImageDecoderConfig()
        cfg = self.config
        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid
        self.state_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim)
        self.query = nn.Parameter(torch.randn(self.num_patches, cfg.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_DecoderBlock(cfg.hidden_dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.patch_head = nn.Linear(cfg.hidden_dim, cfg.patch_size * cfg.patch_size * 3)

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Return reconstructed images in ``[0, 1]`` with shape ``(B, 3, H, W)``."""

        if state_emb.ndim != 2:
            raise ValueError(f"state_emb must have shape (B, D), got {tuple(state_emb.shape)}")
        cfg = self.config
        b = state_emb.shape[0]
        memory = self.state_proj(state_emb).unsqueeze(1)
        query = self.query.unsqueeze(0).expand(b, -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        patches = self.patch_head(self.norm(query))
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
