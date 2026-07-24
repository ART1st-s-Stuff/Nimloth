"""Token-conditioned velocity network for conditional flow matching."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous flow time in ``[0, 1]``."""

    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def _choose_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_choose_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_choose_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        hidden = hidden + self.time_proj(torch.nn.functional.silu(time_emb)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class _TokenCrossAttention2d(nn.Module):
    def __init__(self, channels: int, token_dim: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels {channels} must be divisible by heads {heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(_choose_groups(channels), channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(token_dim, channels)
        self.v_proj = nn.Linear(token_dim, channels)
        self.out_proj = nn.Linear(channels, channels)
        # Preserve an unconditional initial path while allowing condition use to grow.
        self.out_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query_input = self.norm(x).flatten(2).transpose(1, 2)
        query = self.q_proj(query_input).view(
            batch, height * width, self.heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(tokens).view(
            batch, tokens.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(tokens).view(
            batch, tokens.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)
        attention = torch.softmax(
            (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1
        )
        output = (attention @ value).transpose(1, 2).contiguous().view(
            batch, height * width, channels
        )
        output = self.out_proj(output).transpose(1, 2).view(
            batch, channels, height, width
        )
        return x + self.out_scale * output


class _Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"))


@dataclass(frozen=True)
class CFMConfig:
    image_size: int = 128
    token_count: int = 1
    token_dim: int = 1024
    base_channels: int = 64
    condition_dim: int = 256
    time_dim: int = 512
    input_channels: int = 3
    output_channels: int = 3

    def __post_init__(self) -> None:
        if self.image_size < 8 or self.image_size % 8:
            raise ValueError("image_size must be >= 8 and divisible by 8")
        if self.token_count < 1 or self.token_dim < 1:
            raise ValueError("token_count and token_dim must be positive")
        if self.base_channels < 4 or self.base_channels % 4:
            raise ValueError("base_channels must be >= 4 and divisible by 4")
        if self.input_channels < 1 or self.output_channels < 1:
            raise ValueError("input_channels and output_channels must be positive")

    @property
    def flat_condition_dim(self) -> int:
        return self.token_count * self.token_dim

    def to_metadata(self) -> dict[str, int]:
        return asdict(self)


class TokenConditionedFlowUNet(nn.Module):
    """UNet velocity field with global and spatial token conditioning.

    For SFT2 reconstruction the projected WM state is represented as one
    1024-dimensional condition token. The architecture intentionally matches
    the previously validated direct-latent CFM diagnostic.
    """

    def __init__(self, config: CFMConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        self.token_norm = nn.LayerNorm(config.token_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(config.token_dim, config.condition_dim),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.in_conv = nn.Conv2d(config.input_channels, base, 3, padding=1)
        self.block1 = _ResBlock(base, base, config.time_dim)
        self.down1 = _Downsample(base)
        self.block2 = _ResBlock(base, base * 2, config.time_dim)
        self.down2 = _Downsample(base * 2)
        self.block3 = _ResBlock(base * 2, base * 4, config.time_dim)
        self.attention3 = _TokenCrossAttention2d(base * 4, config.condition_dim)
        self.down3 = _Downsample(base * 4)
        self.block4 = _ResBlock(base * 4, base * 6, config.time_dim)
        self.attention4 = _TokenCrossAttention2d(base * 6, config.condition_dim)
        self.middle1 = _ResBlock(base * 6, base * 6, config.time_dim)
        self.middle_attention = _TokenCrossAttention2d(base * 6, config.condition_dim)
        self.middle2 = _ResBlock(base * 6, base * 6, config.time_dim)
        self.up3 = _Upsample(base * 6)
        self.up_block3 = _ResBlock(base * 10, base * 4, config.time_dim)
        self.up_attention3 = _TokenCrossAttention2d(base * 4, config.condition_dim)
        self.up2 = _Upsample(base * 4)
        self.up_block2 = _ResBlock(base * 6, base * 2, config.time_dim)
        self.up1 = _Upsample(base * 2)
        self.up_block1 = _ResBlock(base * 3, base, config.time_dim)
        self.out_norm = nn.GroupNorm(_choose_groups(base), base)
        self.out_conv = nn.Conv2d(base, config.output_channels, 3, padding=1)

    def encode_condition(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.config.flat_condition_dim
        if condition.ndim != 2 or condition.shape[1] != expected:
            raise ValueError(f"expected condition shape (B, {expected}), got {tuple(condition.shape)}")
        tokens = condition.view(
            condition.shape[0], self.config.token_count, self.config.token_dim
        )
        tokens = self.token_proj(self.token_norm(tokens.float()))
        return tokens, self.condition_mlp(tokens.mean(dim=1))

    def forward(
        self,
        image: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        tokens, condition_emb = self.encode_condition(condition)
        time_emb = self.time_mlp(
            timestep_embedding(time, self.config.time_dim).to(dtype=image.dtype)
        ) + condition_emb.to(dtype=image.dtype)
        tokens = tokens.to(dtype=image.dtype)

        hidden0 = self.in_conv(image)
        hidden1 = self.block1(hidden0, time_emb)
        hidden2 = self.block2(self.down1(hidden1), time_emb)
        hidden3 = self.attention3(
            self.block3(self.down2(hidden2), time_emb), tokens
        )
        hidden4 = self.attention4(
            self.block4(self.down3(hidden3), time_emb), tokens
        )
        hidden = self.middle2(
            self.middle_attention(self.middle1(hidden4, time_emb), tokens), time_emb
        )
        hidden = self.up3(hidden)
        hidden = self.up_attention3(
            self.up_block3(torch.cat([hidden, hidden3], dim=1), time_emb), tokens
        )
        hidden = self.up2(hidden)
        hidden = self.up_block2(torch.cat([hidden, hidden2], dim=1), time_emb)
        hidden = self.up1(hidden)
        hidden = self.up_block1(torch.cat([hidden, hidden1], dim=1), time_emb)
        return self.out_conv(torch.nn.functional.silu(self.out_norm(hidden)))
