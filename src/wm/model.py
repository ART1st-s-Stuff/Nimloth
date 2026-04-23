"""Transformer 世界模型实现。"""

from __future__ import annotations

import torch
from torch import nn


class ActionConditioning(nn.Module):
    """将动作条件注入到 token 特征。"""

    def __init__(self, hidden_dim: int, action_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode.strip().lower()
        if self.mode not in {"adaln", "film"}:
            raise ValueError(f"不支持的 conditioning.mode={mode}")
        self.norm = nn.LayerNorm(hidden_dim)
        self.modulator = nn.Sequential(
            nn.Linear(action_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, tokens: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        # tokens: [B, H, P, D], action_history: [B, H, A]
        gamma_beta = self.modulator(action_history).unsqueeze(2)  # [B, H, 1, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        if self.mode == "adaln":
            base = self.norm(tokens)
            return (1.0 + gamma) * base + beta
        return (1.0 + gamma) * tokens + beta


class CFMWorldModel(nn.Module):
    """输入历史 latent/action 序列，输出下一步残差 latent（delta）。"""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        history_len: int,
        num_patches: int,
        token_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        conditioning_mode: str = "adaln",
        action_input_mode: str = "explicit_token_concat",
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        if self.num_patches <= 0 or self.token_dim <= 0:
            raise ValueError(f"非法 patch 配置: num_patches={self.num_patches}, token_dim={self.token_dim}")
        expected_latent_dim = self.num_patches * self.token_dim
        if int(self.latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")
        self.token_proj = nn.Linear(self.token_dim, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, history_len, 1, hidden_dim))
        self.patch_embedding = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_dim))
        self.action_input_mode = action_input_mode.strip().lower()
        if self.action_input_mode not in {"explicit_token_concat", "cross_attention", "modulation"}:
            raise ValueError(f"不支持的 conditioning.action_input_mode={action_input_mode}")
        self.conditioning = ActionConditioning(hidden_dim=hidden_dim, action_dim=action_dim, mode=conditioning_mode)
        self.action_token_proj = nn.Linear(action_dim, hidden_dim)
        self.action_slot_embedding = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )

    def _validate_inputs(self, z_history: torch.Tensor, action_history: torch.Tensor) -> None:
        if z_history.dim() != 4:
            raise ValueError(f"z_history 形状不合法，期望 [B,H,P,D]，实际 {tuple(z_history.shape)}")
        if action_history.dim() != 3:
            raise ValueError(f"action_history 形状不合法，期望 [B,H,A]，实际 {tuple(action_history.shape)}")
        if z_history.size(1) != self.history_len:
            raise ValueError(f"history_len 不一致: {z_history.size(1)} != {self.history_len}")
        if z_history.size(2) != self.num_patches:
            raise ValueError(f"num_patches 不一致: {z_history.size(2)} != {self.num_patches}")
        if z_history.size(3) != self.token_dim:
            raise ValueError(f"token_dim 不一致: {z_history.size(3)} != {self.token_dim}")
        if action_history.size(1) != self.history_len:
            raise ValueError(f"action_history history_len 不一致: {action_history.size(1)} != {self.history_len}")
        if action_history.size(0) != z_history.size(0):
            raise ValueError("z_history 与 action_history batch 不一致")

    def forward(self, z_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """预测下一步相对当前步的残差 delta_z。"""
        self._validate_inputs(z_history=z_history, action_history=action_history)
        x = self.token_proj(z_history)
        x = x + self.time_embedding[:, : self.history_len, :, :] + self.patch_embedding[:, :, : self.num_patches, :]
        if self.action_input_mode == "explicit_token_concat":
            action_tokens = self.action_token_proj(action_history).unsqueeze(2)
            action_tokens = action_tokens + self.action_slot_embedding
            x = torch.cat([x, action_tokens], dim=2)
            x = x.reshape(x.size(0), self.history_len * (self.num_patches + 1), x.size(-1))
            hidden = self.encoder(x)
            hidden = hidden.reshape(hidden.size(0), self.history_len, self.num_patches + 1, hidden.size(-1))
            current_step_tokens = hidden[:, -1, : self.num_patches, :]
        elif self.action_input_mode == "cross_attention":
            x_flat = x.reshape(x.size(0), self.history_len * self.num_patches, x.size(-1))
            action_tokens = self.action_token_proj(action_history)
            cross_out, _ = self.cross_attention(query=x_flat, key=action_tokens, value=action_tokens, need_weights=False)
            x = self.cross_attention_norm(x_flat + cross_out)
            hidden = self.encoder(x)
            hidden = hidden.reshape(hidden.size(0), self.history_len, self.num_patches, hidden.size(-1))
            current_step_tokens = hidden[:, -1, :, :]
        else:
            x = self.conditioning(tokens=x, action_history=action_history)
            x = x.reshape(x.size(0), self.history_len * self.num_patches, x.size(-1))
            hidden = self.encoder(x)
            current_step_tokens = hidden[:, -self.num_patches :, :]
        return self.head(current_step_tokens)

    def predict_next(self, z_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """将残差输出还原为下一步 latent 绝对值。"""
        delta_z = self.forward(z_history=z_history, action_history=action_history)
        return z_history[:, -1, :, :] + delta_z

