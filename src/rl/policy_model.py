"""策略模型 (Policy Model / PM)。

基于 InverseDynamicsModel，扩展支持 [z_t; s_t] 联合输入。
输出动作分布（连续动作用 Gaussian 表示）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.wm.inverse_dynamics import InverseDynamicsModel


class PolicyModel(nn.Module):
    """策略网络：输入 [z_history; s_t]（可选），输出动作分布。

    基于 InverseDynamicsModel 的 Transformer 架构，
    添加 VLM 语义特征融合和动作分布输出头。
    """

    def __init__(
        self,
        # latent 历史参数（与 InverseDynamicsModel 兼容）
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        history_len: int,
        num_patches: int,
        token_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        # VLM 语义特征维度（可选，0 表示不使用）
        semantic_dim: int = 0,
        # 动作分布参数
        action_std_init: float = 0.5,
        action_std_min: float = 0.01,
        use_vlm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        self.semantic_dim = semantic_dim
        self.action_std_min = action_std_min
        self.use_vlm = use_vlm

        expected_latent_dim = self.num_patches * self.token_dim
        if int(self.latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")

        # === 基础 Transformer（处理 z_history） ===
        self.patch_token_proj = nn.Linear(self.token_dim, hidden_dim)
        # 使用 Einstein sum 实现 patch 维度的平均，保持 [B, H, 1, D] 形状
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
        if self.use_vlm and self.semantic_dim > 0:
            self.semantic_proj = nn.Sequential(
                nn.Linear(self.semantic_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # 门控融合
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            fusion_input_dim = hidden_dim
        else:
            self.semantic_proj = None
            self.gate_proj = None
            fusion_input_dim = hidden_dim

        # === 动作分布头 ===
        self.mean_head = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 对数标准差（可学习参数）
        self.log_std = nn.Parameter(torch.ones(action_dim) * action_std_init)

    def _validate_inputs(self, z_history: Tensor) -> None:
        if z_history.dim() != 4:
            raise ValueError(f"z_history 形状不合法，期望 [B,H,P,D]，实际 {tuple(z_history.shape)}")
        if z_history.size(1) != self.history_len:
            raise ValueError(f"history_len 不一致: {z_history.size(1)} != {self.history_len}")
        if z_history.size(2) != self.num_patches:
            raise ValueError(f"num_patches 不一致: {z_history.size(2)} != {self.num_patches}")
        if z_history.size(3) != self.token_dim:
            raise ValueError(f"token_dim 不一致: {z_history.size(3)} != {self.token_dim}")

    def forward(self, z_history: Tensor, semantic: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """前向传播，返回 (mean, std)。

        Args:
            z_history: [B, H, P, D] latent 历史
            semantic: [B, D_s] VLM 语义特征（可选）

        Returns:
            mean: [B, A] 动作均值
            std: [B, A] 动作标准差
        """
        self._validate_inputs(z_history)
        B = z_history.size(0)

        # 1. 处理 latent 历史
        patch_hidden = self.patch_token_proj(z_history)  # [B, H, P, hidden_dim]
        # Patch 维度平均，使用 einsum 保持形状 [B, H, hidden_dim]
        pooled = torch.einsum("BHPD->BHD", patch_hidden) / float(self.num_patches)  # [B, H, hidden_dim]
        pooled = pooled + self.pos_embedding[:, : pooled.size(1), :]
        hidden = self.encoder(pooled)[:, -1, :]  # [B, hidden_dim]

        # 2. 融合 VLM 语义特征
        if self.use_vlm and self.semantic_dim > 0 and semantic is not None:
            if semantic.size(0) != B:
                raise ValueError(f"semantic batch 不一致: {semantic.size(0)} != {B}")
            s_proj = self.semantic_proj(semantic)  # [B, hidden_dim]
            gate = self.gate_proj(torch.cat([hidden, s_proj], dim=-1))  # [B, hidden_dim]
            hidden = hidden * gate + s_proj * (1 - gate)  # 门控融合

        # 3. 动作分布
        mean = self.mean_head(hidden)
        # 使用 clone 避免 in-place 操作问题
        std = torch.exp(self.log_std).clamp(min=self.action_std_min).clone()
        std = std.unsqueeze(0).expand(B, -1)  # [B, A]

        return mean, std

    @torch.no_grad()
    def act(self, z_history: Tensor, semantic: Tensor | None = None, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """选择动作，用于收集经验。

        Args:
            z_history: [B, H, P, D] latent 历史
            semantic: [B, D_s] VLM 语义特征
            deterministic: True 时返回均值（用于评估）

        Returns:
            action: [B, A] 选定动作
            log_prob: [B] 动作 log 概率
            entropy: [B] 动作分布熵
        """
        mean, std = self.forward(z_history, semantic)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # 重参数化采样

        log_prob = dist.log_prob(action).sum(dim=-1)  # [B]
        entropy = dist.entropy().sum(dim=-1)  # [B]

        return action, log_prob, entropy

    def evaluate_actions(self, z_history: Tensor, semantic: Tensor | None, action: Tensor) -> tuple[Tensor, Tensor]:
        """评估给定动作的 log_prob 和 entropy，用于 PPO 更新。

        Args:
            z_history: [B, H, P, D]
            semantic: [B, D_s]
            action: [B, A]

        Returns:
            log_prob: [B]
            entropy: [B]
        """
        mean, std = self.forward(z_history, semantic)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PolicyModelFromIDM(InverseDynamicsModel):
    """从已有 InverseDynamicsModel 快速转换的策略网络。

    保留原有 IDM 结构，仅替换输出头为动作分布头。
    适合从 Phase 2 IDM 预训练权重热启动。
    """

    def __init__(
        self,
        idm: InverseDynamicsModel,
        semantic_dim: int = 0,
        action_std_init: float = 0.5,
    ) -> None:
        latent_dim = idm.latent_dim
        action_dim = idm.head[-1].out_features
        hidden_dim = idm.patch_token_proj.out_features
        history_len = idm.history_len
        num_patches = idm.num_patches
        token_dim = idm.token_dim
        num_layers = idm.encoder.num_layers
        num_heads = idm.encoder.self_attn.num_heads
        dropout = idm.encoder.dropout.p

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            history_len=history_len,
            num_patches=num_patches,
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 替换输出头为动作分布
        self.semantic_dim = semantic_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.semantic_dim, hidden_dim),
            nn.GELU(),
        ) if semantic_dim > 0 else None
        self.log_std = nn.Parameter(torch.ones(action_dim) * action_std_init)

        # 将原有 head 权重复制给 mean_head
        self.mean_head = idm.head
        # 重新初始化 head（使用浅拷贝避免共享）
        self.head = nn.Identity()

    def forward(self, z_history: Tensor, semantic: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self._validate_inputs(z_history)
        B = z_history.size(0)

        patch_hidden = self.patch_token_proj(z_history)
        pooled = self.patch_pool(patch_hidden).mean(dim=2)
        pooled = pooled + self.pos_embedding[:, : pooled.size(1), :]
        hidden = self.encoder(pooled)[:, -1, :]

        if self.semantic_proj is not None and semantic is not None:
            s_proj = self.semantic_proj(semantic)
            hidden = hidden + s_proj

        mean = self.mean_head(hidden)
        std = torch.exp(self.log_std).clamp(min=0.01).clone()
        std = std.unsqueeze(0).expand(B, -1)
        return mean, std

    @torch.no_grad()
    def act(self, z_history: Tensor, semantic: Tensor | None = None, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        mean, std = self.forward(z_history, semantic)
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, z_history: Tensor, semantic: Tensor | None, action: Tensor) -> tuple[Tensor, Tensor]:
        mean, std = self.forward(z_history, semantic)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
