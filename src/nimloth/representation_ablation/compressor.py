"""Attention-based token compressors for representation ablations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class AttentionTokenCompressorConfig:
    """Perceiver-style compressor from many input tokens to K latent tokens."""

    input_dim: int = 2048
    output_dim: int = 512
    input_tokens: int = 81
    num_output_tokens: int = 16
    depth: int = 2
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if self.input_tokens <= 0 or self.num_output_tokens <= 0:
            raise ValueError("input_tokens and num_output_tokens must be positive")
        if self.output_dim % self.heads != 0:
            raise ValueError("output_dim must be divisible by heads")


class _CompressorBlock(nn.Module):
    def __init__(self, cfg: AttentionTokenCompressorConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(cfg.output_dim)
        self.input_norm = nn.LayerNorm(cfg.output_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.output_dim,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(cfg.output_dim)
        self.self_attn = nn.MultiheadAttention(
            cfg.output_dim,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        hidden = cfg.output_dim * cfg.mlp_ratio
        self.mlp_norm = nn.LayerNorm(cfg.output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.output_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.output_dim),
        )

    def forward(self, queries: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(queries)
        kv = self.input_norm(inputs)
        queries = queries + self.cross_attn(q, kv, kv, need_weights=False)[0]
        q = self.self_norm(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class AttentionTokenCompressor(nn.Module):
    """Compress ``(B, N, Din)`` tokens into ``(B, K, Dout)`` tokens.

    This is intended for the Qwen-vision predictor diagnostic: Qwen visual
    tokens are frozen, while this compressor and the predictor are trained with
    LeWM-style prediction + SIGReg losses.
    """

    def __init__(self, config: AttentionTokenCompressorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.output_dim)
        self.input_pos = nn.Parameter(torch.randn(1, config.input_tokens, config.output_dim) * 0.02)
        self.query_tokens = nn.Parameter(torch.randn(1, config.num_output_tokens, config.output_dim) * 0.02)
        self.blocks = nn.ModuleList([_CompressorBlock(config) for _ in range(config.depth)])
        self.out_norm = nn.LayerNorm(config.output_dim)

    @property
    def emb_dim(self) -> int:
        return self.config.output_dim

    @property
    def num_tokens(self) -> int:
        return self.config.num_output_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (B, N, D), got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.config.input_tokens or tokens.shape[2] != self.config.input_dim:
            raise ValueError(
                f"expected tokens shape (B, {self.config.input_tokens}, {self.config.input_dim}), "
                f"got {tuple(tokens.shape)}"
            )
        inputs = self.input_proj(tokens) + self.input_pos.to(device=tokens.device, dtype=tokens.dtype)
        queries = self.query_tokens.expand(tokens.shape[0], -1, -1).to(device=tokens.device, dtype=inputs.dtype)
        for block in self.blocks:
            queries = block(queries, inputs)
        return self.out_norm(queries.float()).to(dtype=inputs.dtype)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "compressor.pt")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "AttentionTokenCompressor":
        path = Path(path)
        cfg = AttentionTokenCompressorConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        module = cls(cfg)
        state_path = path / "compressor.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing AttentionTokenCompressor checkpoint: {state_path}")
        module.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
        return module
