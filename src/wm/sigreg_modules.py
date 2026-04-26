"""SIGReg 相关模块实现。

方案1：MLP Encoder/Decoder 架构
    - 将输入向量编码到新的 latent space，在那里应用 SIGReg
    - decoder 将 predicted vector 映射回原始空间

方案2：图像 encoder 微调
    - 在 Phase2 训练中同时微调图像 encoder
    - 使用 SIGReg 作为 encoder 输出的约束
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer（单 GPU）。

    基于随机投影 + Epps-Pulley 统计量，度量分布与标准高斯的偏差。

    参数:
        num_quadrature_points: 积分节点数量
        num_proj: 随机投影数量
        t_min: Epps-Pulley 积分下界
        t_max: Epps-Pulley 积分上界
        kernel_sigma: Gaussian 窗的带宽参数
    """

    def __init__(
        self,
        num_quadrature_points: int = 16,
        num_proj: int = 256,
        t_min: float = 0.2,
        t_max: float = 4.0,
        kernel_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_proj = num_proj
        knots = num_quadrature_points
        t = torch.linspace(t_min, t_max, knots, dtype=torch.float32)
        dt = (t_max - t_min) / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        kernel = torch.exp(-t.square() / (2.0 * kernel_sigma * kernel_sigma))
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * kernel)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (T, B, D) — 时间步 x batch x 特征维度
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def sigreg_loss(
    latents: torch.Tensor,
    *,
    num_projections: int,
    num_quadrature_points: int,
    t_min: float,
    t_max: float,
    kernel_sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """基于随机投影 + Epps-Pulley 统计量的 SIGReg 近似实现。

    Args:
        latents: 输入张量，任意形状，最后一维是特征维度
        num_projections: 随机投影数量
        num_quadrature_points: 积分节点数量
        t_min: 积分下界
        t_max: 积分上界
        kernel_sigma: Gaussian 窗的带宽参数
        eps: 数值稳定性参数
    """
    if latents.dim() < 2:
        raise ValueError("latents 至少需要二维张量。")
    feature_dim = int(latents.size(-1))
    flat = latents.reshape(-1, feature_dim)
    if flat.size(0) <= 1:
        return torch.zeros((), device=latents.device, dtype=latents.dtype)
    projections = max(1, int(num_projections))
    quadrature_points = max(2, int(num_quadrature_points))
    t_low = float(t_min)
    t_high = float(t_max)
    sigma = max(float(kernel_sigma), eps)

    directions = torch.randn(feature_dim, projections, device=latents.device, dtype=latents.dtype)
    directions = directions / torch.clamp(directions.norm(dim=0, keepdim=True), min=eps)
    projected = flat @ directions  # [N, M]

    t_grid = torch.linspace(t_low, t_high, quadrature_points, device=latents.device, dtype=latents.dtype)
    phase = projected.unsqueeze(0) * t_grid.view(-1, 1, 1)  # [T, N, M]
    phi_real = torch.cos(phase).mean(dim=1)
    phi_imag = torch.sin(phase).mean(dim=1)
    phi_standard = torch.exp(-0.5 * t_grid.square()).view(-1, 1)
    weights = torch.exp(-t_grid.square() / (2.0 * sigma * sigma)).view(-1, 1)
    diff_sq = (phi_real - phi_standard).square() + phi_imag.square()
    integrand = weights * diff_sq
    score_per_projection = torch.trapz(integrand, t_grid, dim=0)
    return score_per_projection.mean()


# ---------------------------------------------------------------------------
# 方案1：SIGReg MLP Encoder/Decoder
# ---------------------------------------------------------------------------


class SIGRegEncoderDecoder(nn.Module):
    """SIGReg Encoder-Decoder 模块。

    模拟原版 LeWM-Encoder 的作用：
    1. Encoder 将输入向量编码到新的 latent space
    2. 在该空间应用 SIGReg 正则化
    3. Decoder 将 predicted vector 映射回原始空间

    结构：
        Input [B, P, token_dim] -> Encoder -> [B, P, sigreg_latent_dim]
        SIGReg 作用于 encoder 输出
        Decoder -> Output [B, P, token_dim]

    Args:
        token_dim: 输入/输出空间的维度
        sigreg_latent_dim: SIGReg 正则化空间的维度
        hidden_dim: Encoder/Decoder 内部隐藏层维度
        num_layers: Encoder/Decoder 的层数
    """

    def __init__(
        self,
        token_dim: int,
        sigreg_latent_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.sigreg_latent_dim = sigreg_latent_dim

        if hidden_dim is None:
            hidden_dim = max(token_dim, sigreg_latent_dim)

        # Encoder: token_dim -> sigreg_latent_dim
        encoder_layers = []
        in_dim = token_dim
        for i in range(num_layers - 1):
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, sigreg_latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: sigreg_latent_dim -> token_dim
        decoder_layers = []
        in_dim = sigreg_latent_dim
        for i in range(num_layers - 1):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, token_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到 SIGReg latent space。"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码回原始空间。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播：编码 -> 解码。

        Args:
            x: [B, P, token_dim] 输入张量

        Returns:
            (encoded, decoded): 编码后的 latent 和解码后的输出
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


class LeWMSIGRegWrapper(nn.Module):
    """LeWM + SIGReg Encoder/Decoder 包装器。

    将 SIGReg Encoder-Decoder 集成到 LeWM 预测流程中：
    1. 输入 latent 经过 encoder 编码到 SIGReg space
    2. 在该空间进行 Transformer 处理
    3. Decoder 将输出映射回原始空间

    适用于：方案1 - MLP Encoder/Decoder 架构
    """

    def __init__(
        self,
        base_wm: nn.Module,
        token_dim: int,
        sigreg_latent_dim: int,
        sigreg_encoder_hidden_dim: int | None = None,
        sigreg_encoder_num_layers: int = 2,
        dropout: float = 0.0,
        sigreg_num_quadrature_points: int = 16,
        sigreg_num_proj: int = 256,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_kernel_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_wm = base_wm
        self.sigreg_ed = SIGRegEncoderDecoder(
            token_dim=token_dim,
            sigreg_latent_dim=sigreg_latent_dim,
            hidden_dim=sigreg_encoder_hidden_dim,
            num_layers=sigreg_encoder_num_layers,
            dropout=dropout,
        )
        self.sigreg = SIGReg(
            num_quadrature_points=sigreg_num_quadrature_points,
            num_proj=sigreg_num_proj,
            t_min=sigreg_t_min,
            t_max=sigreg_t_max,
            kernel_sigma=sigreg_kernel_sigma,
        )

    def compute_sigreg(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """计算 SIGReg 正则损失。

        Args:
            z_sequence: [B, T, P, D] 或 [B, T, D] 原始 latent 序列

        Returns:
            SIGReg 损失值
        """
        # 编码到 SIGReg space
        if z_sequence.dim() == 4:
            B, T, P, D = z_sequence.shape
            # [B, T, P, D] -> [T, B, P, sigreg_latent_dim]
            z_flat = z_sequence.reshape(B, T * P, D)
            z_encoded = self.sigreg_ed.encode(z_flat)  # [B, T*P, sigreg_latent_dim]
            z_encoded = z_encoded.reshape(T, B, P, -1)
        else:
            T, B, D = z_sequence.shape
            z_encoded = self.sigreg_ed.encode(z_sequence.reshape(B, T, D))  # [B, T, sigreg_latent_dim]
            z_encoded = z_encoded.permute(1, 0, 2)  # [T, B, sigreg_latent_dim]
        return self.sigreg(z_encoded)


class CFMSIGRegWrapper(nn.Module):
    """CFM + SIGReg Encoder/Decoder 包装器。

    将 SIGReg Encoder-Decoder 集成到 CFM 预测流程中：
    1. 输入 latent 经过 encoder 编码到 SIGReg space
    2. 在该空间进行 CFM 处理
    3. Decoder 将输出映射回原始空间

    适用于：方案1 - MLP Encoder/Decoder 架构
    """

    def __init__(
        self,
        base_wm: nn.Module,
        token_dim: int,
        sigreg_latent_dim: int,
        sigreg_encoder_hidden_dim: int | None = None,
        sigreg_encoder_num_layers: int = 2,
        dropout: float = 0.0,
        sigreg_num_quadrature_points: int = 16,
        sigreg_num_proj: int = 256,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_kernel_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_wm = base_wm
        self.sigreg_ed = SIGRegEncoderDecoder(
            token_dim=token_dim,
            sigreg_latent_dim=sigreg_latent_dim,
            hidden_dim=sigreg_encoder_hidden_dim,
            num_layers=sigreg_encoder_num_layers,
            dropout=dropout,
        )
        self.sigreg = SIGReg(
            num_quadrature_points=sigreg_num_quadrature_points,
            num_proj=sigreg_num_proj,
            t_min=sigreg_t_min,
            t_max=sigreg_t_max,
            kernel_sigma=sigreg_kernel_sigma,
        )

    def compute_sigreg_on_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """对 latent 序列计算 SIGReg 损失。

        Args:
            latents: [B, T, P, D] 或 [B, T, D] 原始 latent 序列

        Returns:
            SIGReg 损失值
        """
        # 编码到 SIGReg space
        if latents.dim() == 4:
            B, T, P, D = latents.shape
            z_flat = latents.reshape(B, T * P, D)
            z_encoded = self.sigreg_ed.encode(z_flat)  # [B, T*P, sigreg_latent_dim]
            z_encoded = z_encoded.reshape(T, B, P, -1)  # [T, B, P, sigreg_latent_dim]
        else:
            T, B, D = latents.shape
            z_encoded = self.sigreg_ed.encode(latents.reshape(B, T, D))
            z_encoded = z_encoded.permute(1, 0, 2)  # [T, B, sigreg_latent_dim]
        return self.sigreg(z_encoded)


# ---------------------------------------------------------------------------
# 方案2：图像 Encoder 微调 + SIGReg
# ---------------------------------------------------------------------------


class ImageEncoderSIGRegRegularizer(nn.Module):
    """图像 Encoder 输出空间的 SIGReg 正则化器。

    用于在 Phase2 训练中同时微调图像 encoder（DINOv2, Qwen等），
    并使用 SIGReg 作为约束的一部分。

    使用方式：
        1. 将图像 encoder 的输出作为输入
        2. 对该输出应用 SIGReg 正则化
        3. 在训练损失中加入 SIGReg 约束

    Args:
        sigreg_latent_dim: SIGReg 正则化空间的维度
            若与 encoder 输出维度不同，则添加投影层
        num_quadrature_points: 积分节点数量
        num_proj: 随机投影数量
        t_min: 积分下界
        t_max: 积分上界
        kernel_sigma: Gaussian 窗的带宽参数
    """

    def __init__(
        self,
        encoder_output_dim: int,
        sigreg_latent_dim: int | None = None,
        num_quadrature_points: int = 16,
        num_proj: int = 256,
        t_min: float = 0.2,
        t_max: float = 4.0,
        kernel_sigma: float = 1.0,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.sigreg_latent_dim = sigreg_latent_dim or encoder_output_dim

        # 如果维度不匹配，添加投影层
        if self.encoder_output_dim != self.sigreg_latent_dim:
            h_dim = hidden_dim or max(encoder_output_dim, self.sigreg_latent_dim)
            self.proj = nn.Sequential(
                nn.Linear(encoder_output_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Linear(h_dim, self.sigreg_latent_dim),
            )
        else:
            self.proj = nn.Identity()

        self.sigreg = SIGReg(
            num_quadrature_points=num_quadrature_points,
            num_proj=num_proj,
            t_min=t_min,
            t_max=t_max,
            kernel_sigma=kernel_sigma,
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """计算 SIGReg 正则损失。

        Args:
            encoder_output: [B, T, D] 或 [B, D] encoder 输出
                T 是时间步，D 是 encoder 特征维度

        Returns:
            SIGReg 损失值
        """
        # 投影到 SIGReg space
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)  # [B, 1, D]

        z = self.proj(encoder_output)  # [B, T, sigreg_latent_dim]

        # 转换为 [T, B, D] 格式
        T, B, D = z.shape[1], z.shape[0], z.shape[2]
        z_perm = z.permute(1, 0, 2)  # [T, B, D]

        return self.sigreg(z_perm)


class EncodedLatentSIGRegRegularizer(nn.Module):
    """对已编码的 latent 序列应用 SIGReg 正则化。

    直接对 z_history 和 z_future 的 latent 序列应用 SIGReg，
    不再需要额外的 encoder。

    这是方案1的简化版本：直接使用原始 latent 作为 SIGReg space。

    Args:
        sigreg_latent_dim: SIGReg 正则化空间的维度
        num_quadrature_points: 积分节点数量
        num_proj: 随机投影数量
        t_min: 积分下界
        t_max: 积分上界
        kernel_sigma: Gaussian 窗的带宽参数
    """

    def __init__(
        self,
        latent_dim: int,
        sigreg_latent_dim: int | None = None,
        num_quadrature_points: int = 16,
        num_proj: int = 256,
        t_min: float = 0.2,
        t_max: float = 4.0,
        kernel_sigma: float = 1.0,
        use_mlp_projection: bool = True,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.sigreg_latent_dim = sigreg_latent_dim or latent_dim

        if use_mlp_projection and self.latent_dim != self.sigreg_latent_dim:
            h_dim = hidden_dim or max(latent_dim, self.sigreg_latent_dim)
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Linear(h_dim, self.sigreg_latent_dim),
            )
        else:
            self.proj = nn.Identity()

        self.sigreg = SIGReg(
            num_quadrature_points=num_quadrature_points,
            num_proj=num_proj,
            t_min=t_min,
            t_max=t_max,
            kernel_sigma=kernel_sigma,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """计算 SIGReg 正则损失。

        Args:
            latents: [B, T, P, D] 或 [B, T, D] 或 [T, B, D] latent 序列

        Returns:
            SIGReg 损失值
        """
        # 确保是 [B, T, D] 或 [B, T, P, D] 格式
        if latents.dim() == 4:
            B, T, P, D = latents.shape
            latents = latents.reshape(B, T * P, D)  # [B, T*P, D]

        # 投影到 SIGReg space
        z = self.proj(latents)  # [B, T, sigreg_latent_dim]

        # 转换为 [T, B, D] 格式
        T, B, D = z.shape[1], z.shape[0], z.shape[2]
        z_perm = z.permute(1, 0, 2)  # [T, B, D]

        return self.sigreg(z_perm)
