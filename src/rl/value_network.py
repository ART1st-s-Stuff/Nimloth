"""Value 函数网络 (Critic)。

估计状态价值 V(s)，用于 GAE 优势函数计算。
架构与 PolicyModel 类似，但输出标量。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ValueNetwork(nn.Module):
    """Value 函数网络。

    将 [z_history; s_t] 映射到状态价值 V(s)。
    与 PolicyModel 共享特征编码器，仅最后一层不同。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        history_len: int,
        num_patches: int,
        token_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        semantic_dim: int = 0,
        use_vlm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.semantic_dim = semantic_dim
        self.use_vlm = use_vlm

        expected_latent_dim = num_patches * token_dim
        if int(latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")

        # === Latent 历史编码 ===
        self.patch_token_proj = nn.Linear(token_dim, hidden_dim)
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, history_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === VLM 语义特征融合 ===
        if use_vlm and semantic_dim > 0:
            self.semantic_proj = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            fusion_dim = hidden_dim
        else:
            self.semantic_proj = None
            self.gate_proj = None
            fusion_dim = hidden_dim

        # === Value 头 ===
        self.value_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _validate_inputs(self, z_history: Tensor) -> None:
        if z_history.dim() != 4:
            raise ValueError(f"z_history 形状不合法，期望 [B,H,P,D]，实际 {tuple(z_history.shape)}")

    def forward(self, z_history: Tensor, semantic: Tensor | None = None) -> Tensor:
        """估计状态价值。

        Args:
            z_history: [B, H, P, D] latent 历史
            semantic: [B, D_s] VLM 语义特征

        Returns:
            value: [B, 1] 状态价值估计
        """
        self._validate_inputs(z_history)
        B = z_history.size(0)

        patch_hidden = self.patch_token_proj(z_history)
        pooled = torch.einsum("BHPD->BHD", patch_hidden) / float(self.num_patches)
        pooled = pooled + self.pos_embedding[:, : pooled.size(1), :]
        hidden = self.encoder(pooled)[:, -1, :]

        if self.use_vlm and self.semantic_dim > 0 and semantic is not None:
            s_proj = self.semantic_proj(semantic)
            gate = self.gate_proj(torch.cat([hidden, s_proj], dim=-1))
            hidden = hidden * gate + s_proj * (1 - gate)

        value = self.value_head(hidden)  # [B, 1]
        return value.squeeze(-1)  # [B]


class SharedEncoderValueNetwork(ValueNetwork):
    """与 PolicyModel 共享编码器的 Value 网络。

    用于 actor-critic 共享主干网络，减少参数量。
    """

    def __init__(
        self,
        policy_model: "PolicyModel",  # noqa: F821
        value_hidden_dim: int | None = None,
    ) -> None:
        # 从 policy_model 获取参数
        latent_dim = policy_model.latent_dim
        hidden_dim = value_hidden_dim or policy_model.patch_token_proj.out_features
        history_len = policy_model.history_len
        num_patches = policy_model.num_patches
        token_dim = policy_model.token_dim
        semantic_dim = policy_model.semantic_dim
        use_vlm = policy_model.use_vlm

        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            history_len=history_len,
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=2,  # Value 网络通常比 Policy 更浅
            num_heads=4,
            dropout=0.1,
            semantic_dim=semantic_dim,
            use_vlm=use_vlm,
        )

        # 如果 hidden_dim 不同，需要新的投影层
        if hidden_dim != policy_model.patch_token_proj.out_features:
            self.patch_token_proj = nn.Linear(token_dim, hidden_dim)
            self.patch_pool = nn.Linear(hidden_dim, hidden_dim)
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, history_len, hidden_dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

            if use_vlm and semantic_dim > 0:
                self.semantic_proj = nn.Sequential(
                    nn.Linear(semantic_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.gate_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )

            self.value_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
