"""LeWM（Latent Energy World Model）实现。

参考 lucas-maes/le-wm 的 ARPredictor，使用 AdaLN-Zero 条件化和因果注意力的
Transformer 结构预测下一步 latent 序列。
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.core.interfaces import Model
from src.wm.sigreg_modules import SIGReg, SIGRegEncoderDecoder

try:
    from torchvision.models import vgg16
except Exception:  # pragma: no cover
    vgg16 = None


def _update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for ema_p, src_p in zip(target.parameters(), source.parameters()):
        ema_p.mul_(decay).add_(src_p.detach(), alpha=(1.0 - decay))
    for ema_b, src_b in zip(target.buffers(), source.buffers()):
        ema_b.copy_(src_b)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Zero 调制：x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class _LeWMFeedForward(nn.Module):
    """标准 FFN：LayerNorm → Linear → GELU → Dropout → Linear → Dropout"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LeWMAttention(nn.Module):
    """Scaled dot-product attention，支持因果 mask"""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout_p = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if not (heads == 1 and dim_head == dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(x.size(0), -1, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        )
        drop = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return self.to_out(out)


class _LeWMConditionalBlock(nn.Module):
    """带 AdaLN-Zero 条件化的 Transformer Block"""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = _LeWMAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = _LeWMFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _LeWMTransformer(nn.Module):
    """LeWM Transformer：支持 ConditionalBlock（AdaLN-Zero）或标准 Block"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type = _LeWMConditionalBlock,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.cond_dim = input_dim

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)  # type: ignore[misc]
        return self.output_proj(self.norm(x))


class LatentImageDecoder(nn.Module):
    """Small decoder from latent tokens to an RGB image in [0, 1]."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_patches: int,
        image_size: int = 128,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        self.image_size = int(image_size)
        self.hidden_channels = max(64, ((int(hidden_channels) + 63) // 64) * 64)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_channels * 8 * 8),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, self.hidden_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, self.hidden_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.hidden_channels // 2, self.hidden_channels // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, self.hidden_channels // 4),
            nn.GELU(),
            nn.ConvTranspose2d(self.hidden_channels // 4, self.hidden_channels // 8, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels // 8, 3, kernel_size=3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 3:
            raise ValueError(f"latent 形状不合法，期望 [B,P,D]，实际 {tuple(latent.shape)}")
        pooled = latent.mean(dim=1)
        x = self.fc(pooled).reshape(latent.size(0), self.hidden_channels, 8, 8)
        image = torch.sigmoid(self.decoder(x))
        if int(image.size(-1)) != self.image_size or int(image.size(-2)) != self.image_size:
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return image


class VGGPerceptualLoss(nn.Module):
    """Frozen VGG16 feature L1 loss with an L1 fallback when torchvision is unavailable."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.features: nn.Module | None = None
        if vgg16 is not None:
            model = vgg16(weights=None).features[:16].eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.features = model.to(device)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        if self.features is None:
            return F.l1_loss(pred, target)
        pred_norm = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
        target_norm = (target - self.mean.to(target.device)) / self.std.to(target.device)
        return F.l1_loss(self.features(pred_norm), self.features(target_norm).detach())


class LeWMWorldModel(nn.Module):
    """LeWM 自回归世界模型。

    输入：
        z_history: [B, H, P, D] 历史 latent 序列
        action_history: [B, H, A] 动作历史

    输出：
        pred_z_next: [B, P, D] 预测的下一时刻 latent
    """

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
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        sigreg_enabled: bool = False,
        sigreg_latent_dim: int | None = None,
        sigreg_encoder_hidden_dim: int | None = None,
        sigreg_encoder_num_layers: int = 2,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 256,
        sigreg_num_quadrature_points: int = 16,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_kernel_sigma: float = 1.0,
        reward_enabled: bool = False,
        reward_hidden_dim: int | None = None,
        image_decoder_enabled: bool = False,
        image_decoder_hidden_channels: int = 128,
        image_size: int = 128,
        ensemble_size: int = 1,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        self.sigreg_enabled = sigreg_enabled
        self.reward_enabled = bool(reward_enabled)
        self.image_decoder_enabled = bool(image_decoder_enabled)
        self.ensemble_size = max(1, int(ensemble_size))
        expected_latent_dim = self.num_patches * self.token_dim
        if int(latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")

        mlp_dim = int(hidden_dim * mlp_ratio)
        T = history_len

        self.sigreg_ed: SIGRegEncoderDecoder | None = None
        if sigreg_enabled:
            ed_latent_dim = sigreg_latent_dim or token_dim
            self.sigreg_ed = SIGRegEncoderDecoder(
                token_dim=token_dim,
                sigreg_latent_dim=ed_latent_dim,
                hidden_dim=sigreg_encoder_hidden_dim,
                num_layers=sigreg_encoder_num_layers,
                dropout=dropout,
            )
            transformer_input_dim = ed_latent_dim
            transformer_output_dim = ed_latent_dim
        else:
            transformer_input_dim = token_dim
            transformer_output_dim = token_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, T * self.num_patches, transformer_input_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.action_embed_conv = nn.Conv1d(action_dim, transformer_input_dim, kernel_size=1, stride=1)
        self.action_embed_mlp = nn.Sequential(
            nn.Linear(transformer_input_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, transformer_input_dim),
        )

        self.patch_proj = nn.Linear(transformer_input_dim, hidden_dim)

        self.transformer = _LeWMTransformer(
            input_dim=transformer_input_dim,
            hidden_dim=hidden_dim,
            output_dim=transformer_output_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=_LeWMConditionalBlock,
        )
        self.ensemble_transformers = nn.ModuleList(
            [
                _LeWMTransformer(
                    input_dim=transformer_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=transformer_output_dim,
                    depth=num_layers,
                    heads=num_heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    block_class=_LeWMConditionalBlock,
                )
                for _ in range(self.ensemble_size - 1)
            ]
        )

        self.sigreg: SIGReg | None = None
        if sigreg_enabled:
            self.sigreg = SIGReg(
                num_quadrature_points=sigreg_num_quadrature_points,
                num_proj=sigreg_num_proj,
                t_min=sigreg_t_min,
                t_max=sigreg_t_max,
                kernel_sigma=sigreg_kernel_sigma,
            )

        reward_hidden = int(reward_hidden_dim or max(128, hidden_dim // 2))
        self.reward_head: nn.Module | None = None
        if self.reward_enabled:
            self.reward_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, reward_hidden),
                nn.GELU(),
                nn.Linear(reward_hidden, 1),
            )

        self.image_decoder: LatentImageDecoder | None = None
        if self.image_decoder_enabled:
            self.image_decoder = LatentImageDecoder(
                token_dim=token_dim,
                num_patches=num_patches,
                image_size=image_size,
                hidden_channels=image_decoder_hidden_channels,
            )

    def _validate_inputs(
        self, z_history: torch.Tensor, action_history: torch.Tensor
    ) -> None:
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

    def _encode_inputs(
        self, z_history: torch.Tensor, action_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        self._validate_inputs(z_history=z_history, action_history=action_history)
        B, H, P, D = z_history.shape

        if self.sigreg_ed is not None:
            z_encoded, _ = self.sigreg_ed(z_history.reshape(B, H * P, D))
            x = z_encoded
        else:
            x = z_history.reshape(B, H * P, D)

        x = x + self.pos_embedding[:, : H * P, :]
        x = self.dropout(x)

        x_conv = self.action_embed_conv(action_history.transpose(1, 2))
        B2, C, H2 = x_conv.shape
        x_conv = x_conv.permute(0, 2, 1).reshape(B2 * H2, C)
        act_emb = self.action_embed_mlp(x_conv).reshape(B2, H2, C)
        c = act_emb.unsqueeze(2).expand(-1, -1, P, -1).reshape(B, H * P, -1)
        return self.patch_proj(x), c, (B, H, P, D)

    def _decode_output(
        self, out: torch.Tensor, shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        B, H, P, D = shape
        if self.sigreg_ed is not None:
            pred_z_flat = self.sigreg_ed.decode(out)
        else:
            pred_z_flat = out
        return pred_z_flat.reshape(B, H, P, D)[:, -1, :, :]

    def predict_next_ensemble(
        self, z_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        """Return per-member next-latent predictions: [K, B, P, D]."""
        x, c, shape = self._encode_inputs(z_history=z_history, action_history=action_history)
        preds = [self._decode_output(self.transformer(x, c), shape)]
        for transformer in self.ensemble_transformers:
            preds.append(self._decode_output(transformer(x, c), shape))
        return torch.stack(preds, dim=0)

    def forward(self, z_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """返回预测的下一时刻 latent 序列：[B, P, D]"""
        return self.predict_next_ensemble(z_history, action_history).mean(dim=0)

    def predict_next(
        self, z_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        """返回预测的下一时刻绝对 latent"""
        return self.forward(z_history=z_history, action_history=action_history)

    def predict_reward(self, pred_z: torch.Tensor) -> torch.Tensor:
        if self.reward_head is None:
            raise RuntimeError("reward_head 未启用")
        return self.reward_head(pred_z.mean(dim=1)).squeeze(-1)

    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        if self.image_decoder is None:
            raise RuntimeError("image_decoder 未启用")
        return self.image_decoder(latent)

    def predict_next_with_aux(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        *,
        reconstruct_image: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ensemble_preds = self.predict_next_ensemble(z_history=z_history, action_history=action_history)
        pred_z = ensemble_preds.mean(dim=0)
        aux: dict[str, torch.Tensor] = {"ensemble_preds": ensemble_preds}
        if int(ensemble_preds.size(0)) > 1:
            member_var = ensemble_preds.float().var(dim=0, unbiased=False)
            aux["ensemble_uncertainty"] = member_var.flatten(1).mean(dim=1)
        if self.reward_head is not None:
            aux["reward_pred"] = self.predict_reward(pred_z)
        if reconstruct_image and self.image_decoder is not None:
            aux["image_recon"] = self.decode_image(pred_z)
        return pred_z, aux

    def compute_sigreg(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """计算 SIGReg 正则损失。

        SIGReg 期望输入为 [time, batch, dim]。WM latent 通常是
        [B,T,P,D]，这里将时间和 patch 合并为 time 维，并保留 batch
        维用于估计 batch 分布。
        """
        if self.sigreg is None or self.sigreg_ed is None:
            raise RuntimeError("SIGReg 未启用，无法计算 SIGReg 损失")

        if z_sequence.dim() == 4:
            B, T, P, D = z_sequence.shape
            z_flat = z_sequence.reshape(B, T * P, D)
            z_encoded = self.sigreg_ed.encode(z_flat)
            z_encoded = z_encoded.permute(1, 0, 2).contiguous()  # [T*P, B, E]
        elif z_sequence.dim() == 3:
            B, T, D = z_sequence.shape
            z_flat = z_sequence.reshape(B, T, D)
            z_encoded = self.sigreg_ed.encode(z_flat)
            z_encoded = z_encoded.permute(1, 0, 2).contiguous()  # [T, B, E]
        else:
            raise ValueError(f"SIGReg z_sequence 形状不合法: {tuple(z_sequence.shape)}")

        return self.sigreg(z_encoded)


class LeWMModel(Model):
    """LeWM 统一训练接口，接口与 WMModel（CFM）完全一致。"""

    def __init__(
        self,
        *,
        wm: LeWMWorldModel,
        inverse_dynamics: Any,
        action_mapper: Any,
        wm_optimizer: torch.optim.Optimizer,
        idm_optimizer: torch.optim.Optimizer,
        wm_scheduler: Any,
        idm_scheduler: Any,
        device: torch.device,
        training_mode: str = "unsupervised",
        reconstruction_weight: float = 1.0,
        semi_supervised_weight: float = 1.0,
        grad_clip_norm: float = 1.0,
        ema_decay: float = 0.999,
        detach_idm_in_wm: bool = True,
        sigreg_enabled: bool = False,
        sigreg_target_weight: float = 0.0,
        sigreg_warmup_steps: int = 0,
        reward_enabled: bool = False,
        reward_weight: float = 1.0,
        reward_loss_type: str = "mse",
        negative_action_contrastive_enabled: bool = False,
        negative_action_contrastive_weight: float = 0.0,
        negative_action_contrastive_margin: float = 0.05,
        negative_action_contrastive_num_negatives: int = 1,
        perceptual_enabled: bool = False,
        perceptual_weight: float = 0.1,
        image_recon_weight: float = 0.1,
        detach_target_latents: bool = True,
        fail_on_nonfinite: bool = True,
        wm_extra_clip_params: list[torch.nn.Parameter] | None = None,
    ) -> None:
        super().__init__()
        self.wm = wm.to(device)
        self.idm = inverse_dynamics.to(device)
        self.action_mapper = action_mapper.to(device)
        self.wm_optimizer = wm_optimizer
        self.idm_optimizer = idm_optimizer
        self.wm_scheduler = wm_scheduler
        self.idm_scheduler = idm_scheduler
        self.device = device
        self.training_mode = training_mode
        self.reconstruction_weight = reconstruction_weight
        self.semi_supervised_weight = semi_supervised_weight
        self.grad_clip_norm = grad_clip_norm
        self.detach_idm_in_wm = detach_idm_in_wm
        self.sigreg_enabled = sigreg_enabled
        self.sigreg_target_weight = sigreg_target_weight
        self.sigreg_warmup_steps = sigreg_warmup_steps
        self.reward_enabled = bool(reward_enabled)
        self.reward_weight = float(reward_weight)
        self.reward_loss_type = str(reward_loss_type).strip().lower()
        self.negative_action_contrastive_enabled = bool(negative_action_contrastive_enabled)
        self.negative_action_contrastive_weight = float(negative_action_contrastive_weight)
        self.negative_action_contrastive_margin = float(negative_action_contrastive_margin)
        self.negative_action_contrastive_num_negatives = max(1, int(negative_action_contrastive_num_negatives))
        self.perceptual_enabled = bool(perceptual_enabled)
        self.perceptual_weight = float(perceptual_weight)
        self.image_recon_weight = float(image_recon_weight)
        self.detach_target_latents = bool(detach_target_latents)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        self.perceptual_loss_fn: VGGPerceptualLoss | None = None
        if self.perceptual_enabled:
            self.perceptual_loss_fn = VGGPerceptualLoss(device=device)
        self._ema_model: nn.Module | None = None
        self._ema_decay = ema_decay
        self._epoch = 0
        self._global_step = 0
        self._last_debug_tensors: dict[str, Any] = {}
        self._wm_clip_params = list(self.wm.parameters()) + list(wm_extra_clip_params or [])
        self._last_wm_grad_norm = 0.0
        if self._ema_decay > 0:
            self._ema_model = copy.deepcopy(self.wm).to(device)
            self._ema_model.eval()
            for p in self._ema_model.parameters():
                p.requires_grad_(False)

    def _current_sigreg_weight(self) -> float:
        if not self.sigreg_enabled or self.sigreg_target_weight <= 0.0:
            return 0.0
        if self.sigreg_warmup_steps <= 0:
            return float(self.sigreg_target_weight)
        return float(self.sigreg_target_weight) * min(
            1.0,
            float(self._global_step + 1) / float(self.sigreg_warmup_steps),
        )

    def _compute_wm_loss(
        self,
        pred_z: torch.Tensor,
        target_z: torch.Tensor,
        z_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """计算 WM 损失（预测 vs 目标 MSELoss）+ SIGReg"""
        loss_recon = F.mse_loss(pred_z, target_z)
        loss_sigreg = torch.tensor(0.0, device=self.device)
        current_sigreg_weight = 0.0
        if self.sigreg_enabled:
            loss_sigreg = self.wm.compute_sigreg(z_sequence)
            current_sigreg_weight = self._current_sigreg_weight()
        return loss_recon, loss_sigreg, current_sigreg_weight

    def _wm_core(self) -> LeWMWorldModel:
        """兼容 DataParallel，返回底层 LeWMWorldModel。"""
        if hasattr(self.wm, "module"):
            return self.wm.module  # type: ignore[return-value]
        return self.wm

    def _check_finite(self, name: str, tensor: torch.Tensor) -> None:
        if torch.isfinite(tensor).all():
            return
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() > 0:
            stats = (
                f"finite_min={float(finite.min().item()):.6g} "
                f"finite_max={float(finite.max().item()):.6g} "
                f"finite_mean={float(finite.mean().item()):.6g}"
            )
        else:
            stats = "no finite values"
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        message = (
            f"non-finite tensor detected: {name} shape={tuple(tensor.shape)} "
            f"nan={nan_count} inf={inf_count} {stats}"
        )
        if self.fail_on_nonfinite:
            raise FloatingPointError(message)

    def _compute_reward_loss(
        self,
        reward_pred: torch.Tensor,
        reward_target: torch.Tensor,
    ) -> torch.Tensor:
        reward_target = reward_target.to(device=reward_pred.device, dtype=reward_pred.dtype)
        if self.reward_loss_type == "l1":
            return F.l1_loss(reward_pred, reward_target)
        return F.mse_loss(reward_pred, reward_target)

    def _negative_action_candidates(self, true_actions: torch.Tensor) -> list[torch.Tensor]:
        action_dim = int(true_actions.size(-1))
        if action_dim < 2:
            return []
        candidates: list[torch.Tensor] = []
        if torch.all((true_actions >= 0.0) & (true_actions <= 1.0)):
            action_ids = torch.argmax(true_actions, dim=-1)
            for offset in range(1, min(self.negative_action_contrastive_num_negatives, action_dim - 1) + 1):
                neg_ids = (action_ids + offset) % action_dim
                candidates.append(
                    F.one_hot(neg_ids, num_classes=action_dim).to(
                        device=true_actions.device,
                        dtype=true_actions.dtype,
                    )
                )
        else:
            for offset in range(1, self.negative_action_contrastive_num_negatives + 1):
                candidates.append(torch.roll(true_actions, shifts=offset, dims=-1))
        return candidates

    def _compute_negative_action_contrastive_loss(
        self,
        *,
        wm_core: LeWMWorldModel,
        teacher_z: torch.Tensor,
        teacher_action: torch.Tensor,
        true_action: torch.Tensor,
        pred_z: torch.Tensor,
        target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self.negative_action_contrastive_enabled
            or self.negative_action_contrastive_weight <= 0.0
        ):
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero
        negative_actions = self._negative_action_candidates(true_action)
        if not negative_actions:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero
        pos_dist = F.mse_loss(pred_z, target_z, reduction="none").flatten(1).mean(dim=1)
        losses: list[torch.Tensor] = []
        neg_dists: list[torch.Tensor] = []
        for neg_action in negative_actions:
            neg_teacher_action = teacher_action.clone()
            neg_teacher_action[:, -1, :] = neg_action
            neg_pred_z = wm_core.predict_next(teacher_z, neg_teacher_action)
            neg_dist = F.mse_loss(neg_pred_z, target_z, reduction="none").flatten(1).mean(dim=1)
            losses.append(F.relu(self.negative_action_contrastive_margin + pos_dist - neg_dist).mean())
            neg_dists.append(neg_dist.mean())
        return torch.stack(losses).mean(), torch.stack(neg_dists).mean()

    @torch.no_grad()
    def eval_step(self, batch: Any) -> dict[str, Any]:
        z_history = batch["z_history"].to(self.device)
        action_history = batch["action_history"].to(self.device)
        z_future = batch["z_future"].to(self.device)
        gt_action_future = batch["gt_action_future"].to(self.device)
        if self.fail_on_nonfinite:
            self._check_finite("z_history", z_history)
            self._check_finite("z_future", z_future)
            self._check_finite("action_history", action_history)
            self._check_finite("gt_action_future", gt_action_future)
        reward_targets = batch.get("reward_target")
        if reward_targets is not None:
            reward_targets = reward_targets.to(self.device)
        target_images = batch.get("target_images")
        if target_images is not None:
            target_images = target_images.to(self.device)
        self._last_debug_tensors = {
            "z_history": z_history.detach(),
            "z_future": z_future.detach(),
            "action_history": action_history.detach(),
            "gt_action_future": gt_action_future.detach(),
        }
        if reward_targets is not None:
            self._last_debug_tensors["reward_target"] = reward_targets.detach()
        if target_images is not None:
            self._last_debug_tensors["target_images"] = target_images.detach()
        if self.fail_on_nonfinite:
            self._check_finite("z_history", z_history)
            self._check_finite("z_future", z_future)
            self._check_finite("action_history", action_history)
            self._check_finite("gt_action_future", gt_action_future)
            if reward_targets is not None:
                self._check_finite("reward_target", reward_targets)
            if target_images is not None:
                self._check_finite("target_images", target_images)

        pred_action = None
        if self.training_mode in {"unsupervised", "semi_supervised"}:
            pred_action = self.idm(z_history.detach() if self.training_mode == "semi_supervised" else z_history)
        rollout_horizon = int(z_future.size(1))

        # fully_supervised 模式直接使用 GT 动作，不调用 IDM
        use_gt_actions = self.training_mode == "fully_supervised"

        teacher_z = z_history
        teacher_action = action_history.clone()
        loss_recon_steps: list[torch.Tensor] = []
        loss_reward_steps: list[torch.Tensor] = []
        loss_image_recon_steps: list[torch.Tensor] = []
        loss_perceptual_steps: list[torch.Tensor] = []
        loss_negative_action_steps: list[torch.Tensor] = []
        negative_action_dist_values: list[torch.Tensor] = []
        ensemble_uncertainty_values: list[torch.Tensor] = []

        wm_core = self._wm_core()
        for step_idx in range(rollout_horizon):
            teacher_action[:, -1, :] = gt_action_future[:, step_idx, :]

            pred_z, aux = wm_core.predict_next_with_aux(
                teacher_z,
                teacher_action,
                reconstruct_image=bool(self.perceptual_enabled and target_images is not None),
            )
            target_z = z_future[:, step_idx, :, :]
            if self.detach_target_latents:
                target_z = target_z.detach()
            if self.fail_on_nonfinite:
                self._check_finite(f"pred_z[{step_idx}]", pred_z)
                self._check_finite(f"target_z[{step_idx}]", target_z)
            ensemble_preds = aux.get("ensemble_preds")
            if isinstance(ensemble_preds, torch.Tensor):
                step_loss = F.mse_loss(
                    ensemble_preds, target_z.unsqueeze(0).expand_as(ensemble_preds)
                )
            else:
                step_loss = F.mse_loss(pred_z, target_z)
            uncertainty_step = aux.get("ensemble_uncertainty")
            if isinstance(uncertainty_step, torch.Tensor):
                ensemble_uncertainty_values.append(uncertainty_step.mean())
            loss_recon_steps.append(step_loss)
            negative_loss_step, negative_dist_step = self._compute_negative_action_contrastive_loss(
                wm_core=wm_core,
                teacher_z=teacher_z,
                teacher_action=teacher_action,
                true_action=gt_action_future[:, step_idx, :],
                pred_z=pred_z,
                target_z=target_z,
            )
            self._last_debug_tensors[f"loss_negative_action_step[{step_idx}]"] = negative_loss_step.detach()
            if self.fail_on_nonfinite:
                self._check_finite(f"loss_negative_action_step[{step_idx}]", negative_loss_step)
            loss_negative_action_steps.append(negative_loss_step)
            negative_action_dist_values.append(negative_dist_step.detach())
            if self.reward_enabled and reward_targets is not None and "reward_pred" in aux:
                reward_target_step = reward_targets[:, step_idx] if reward_targets.dim() > 1 else reward_targets
                loss_reward_steps.append(
                    self._compute_reward_loss(aux["reward_pred"], reward_target_step)
                )
            if self.perceptual_enabled and target_images is not None and "image_recon" in aux:
                image_target = target_images[:, step_idx, :, :, :]
                loss_image_recon_steps.append(F.l1_loss(aux["image_recon"], image_target))
                if self.perceptual_loss_fn is not None:
                    loss_perceptual_steps.append(self.perceptual_loss_fn(aux["image_recon"], image_target))

            next_teacher_z = z_future[:, step_idx, :, :]
            if self.detach_target_latents:
                next_teacher_z = next_teacher_z.detach()
            teacher_z = torch.cat([teacher_z[:, 1:, ...], next_teacher_z.unsqueeze(1)], dim=1)
            if step_idx < rollout_horizon - 1:
                teacher_action = torch.cat(
                    [teacher_action[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                    dim=1,
                )

        loss_recon = torch.stack(loss_recon_steps).mean()
        loss_reward = (
            torch.stack(loss_reward_steps).mean()
            if loss_reward_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_image_recon = (
            torch.stack(loss_image_recon_steps).mean()
            if loss_image_recon_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_perceptual = (
            torch.stack(loss_perceptual_steps).mean()
            if loss_perceptual_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_negative_action = (
            torch.stack(loss_negative_action_steps).mean()
            if loss_negative_action_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_negative_action_weighted = self.negative_action_contrastive_weight * loss_negative_action
        negative_action_dist_mean = (
            float(torch.stack(negative_action_dist_values).mean().item())
            if negative_action_dist_values
            else 0.0
        )
        ensemble_uncertainty_mean = (
            float(torch.stack(ensemble_uncertainty_values).mean().item())
            if ensemble_uncertainty_values
            else 0.0
        )
        latent_for_reg = torch.cat([z_history, z_future], dim=1)
        loss_sigreg = torch.tensor(0.0, device=self.device)
        current_sigreg_weight = 0.0
        if self.sigreg_enabled:
            loss_sigreg = wm_core.compute_sigreg(latent_for_reg)
            current_sigreg_weight = self._current_sigreg_weight()
        loss_action = torch.tensor(0.0, device=self.device)
        if self.training_mode == "semi_supervised":
            mapped_action = self.action_mapper(pred_action)
            loss_action = F.mse_loss(mapped_action, gt_action_future[:, 0, :])

        return {
            "loss": (
                float(loss_recon.item())
                + self.reward_weight * float(loss_reward.item())
                + self.image_recon_weight * float(loss_image_recon.item())
                + self.perceptual_weight * float(loss_perceptual.item())
                + float(loss_negative_action_weighted.item())
                + current_sigreg_weight * float(loss_sigreg.item())
                + self.semi_supervised_weight * float(loss_action.item())
            ),
            "loss_recon": float(loss_recon.item()),
            "loss_action": float(loss_action.item()),
            "loss_sigreg": float(loss_sigreg.item()),
            "loss_sigreg_weighted": current_sigreg_weight * float(loss_sigreg.item()),
            "sigreg_weight": current_sigreg_weight,
            "loss_reward": float(loss_reward.item()),
            "loss_image_recon": float(loss_image_recon.item()),
            "loss_perceptual": float(loss_perceptual.item()),
            "loss_negative_action": float(loss_negative_action.item()),
            "loss_negative_action_weighted": float(loss_negative_action_weighted.item()),
            "negative_action_dist_mean": negative_action_dist_mean,
            "ensemble_uncertainty_mean": ensemble_uncertainty_mean,
            "ensemble_size": float(wm_core.ensemble_size),
        }

    def train_step(self, batch: Any) -> dict[str, Any]:
        z_history = batch["z_history"].to(self.device)
        action_history = batch["action_history"].to(self.device)
        z_future = batch["z_future"].to(self.device)
        gt_action_future = batch["gt_action_future"].to(self.device)
        reward_targets = batch.get("reward_target")
        if reward_targets is not None:
            reward_targets = reward_targets.to(self.device)
        target_images = batch.get("target_images")
        if target_images is not None:
            target_images = target_images.to(self.device)
        self._last_debug_tensors = {
            "z_history": z_history.detach(),
            "z_future": z_future.detach(),
            "action_history": action_history.detach(),
            "gt_action_future": gt_action_future.detach(),
        }
        if reward_targets is not None:
            self._last_debug_tensors["reward_target"] = reward_targets.detach()
        if target_images is not None:
            self._last_debug_tensors["target_images"] = target_images.detach()
        if self.fail_on_nonfinite:
            self._check_finite("z_history", z_history)
            self._check_finite("z_future", z_future)
            self._check_finite("action_history", action_history)
            self._check_finite("gt_action_future", gt_action_future)
            if reward_targets is not None:
                self._check_finite("reward_target", reward_targets)
            if target_images is not None:
                self._check_finite("target_images", target_images)

        pred_action = None
        if self.training_mode in {"unsupervised", "semi_supervised"}:
            pred_action = self.idm(z_history.detach() if self.training_mode == "semi_supervised" else z_history)
        rollout_horizon = int(z_future.size(1))

        # fully_supervised 模式直接使用 GT 动作，不调用 IDM
        use_gt_actions = self.training_mode == "fully_supervised"

        teacher_z = z_history
        teacher_action = action_history.clone()
        loss_recon_steps: list[torch.Tensor] = []
        loss_reward_steps: list[torch.Tensor] = []
        loss_image_recon_steps: list[torch.Tensor] = []
        loss_perceptual_steps: list[torch.Tensor] = []
        loss_negative_action_steps: list[torch.Tensor] = []
        negative_action_dist_values: list[torch.Tensor] = []
        ensemble_uncertainty_values: list[torch.Tensor] = []
        reward_pred_values: list[torch.Tensor] = []

        wm_core = self._wm_core()
        for step_idx in range(rollout_horizon):
            teacher_action[:, -1, :] = gt_action_future[:, step_idx, :]

            pred_z, aux = wm_core.predict_next_with_aux(
                teacher_z,
                teacher_action,
                reconstruct_image=bool(self.perceptual_enabled and target_images is not None),
            )
            target_z = z_future[:, step_idx, :, :]
            if self.detach_target_latents:
                target_z = target_z.detach()
            self._last_debug_tensors[f"pred_z[{step_idx}]"] = pred_z.detach()
            self._last_debug_tensors[f"target_z[{step_idx}]"] = target_z.detach()
            if self.fail_on_nonfinite:
                self._check_finite(f"pred_z[{step_idx}]", pred_z)
                self._check_finite(f"target_z[{step_idx}]", target_z)
            ensemble_preds = aux.get("ensemble_preds")
            if isinstance(ensemble_preds, torch.Tensor):
                step_loss = F.mse_loss(
                    ensemble_preds, target_z.unsqueeze(0).expand_as(ensemble_preds)
                )
            else:
                step_loss = F.mse_loss(pred_z, target_z)
            uncertainty_step = aux.get("ensemble_uncertainty")
            if isinstance(uncertainty_step, torch.Tensor):
                ensemble_uncertainty_values.append(uncertainty_step.mean())
            self._last_debug_tensors[f"loss_recon_step[{step_idx}]"] = step_loss.detach()
            if self.fail_on_nonfinite:
                self._check_finite(f"loss_recon_step[{step_idx}]", step_loss)
            loss_recon_steps.append(step_loss)
            negative_loss_step, negative_dist_step = self._compute_negative_action_contrastive_loss(
                wm_core=wm_core,
                teacher_z=teacher_z,
                teacher_action=teacher_action,
                true_action=gt_action_future[:, step_idx, :],
                pred_z=pred_z,
                target_z=target_z,
            )
            self._last_debug_tensors[f"loss_negative_action_step[{step_idx}]"] = negative_loss_step.detach()
            if self.fail_on_nonfinite:
                self._check_finite(f"loss_negative_action_step[{step_idx}]", negative_loss_step)
            loss_negative_action_steps.append(negative_loss_step)
            negative_action_dist_values.append(negative_dist_step.detach())
            if self.reward_enabled and reward_targets is not None and "reward_pred" in aux:
                reward_pred_values.append(aux["reward_pred"].detach())
                reward_target_step = reward_targets[:, step_idx] if reward_targets.dim() > 1 else reward_targets
                reward_loss_step = self._compute_reward_loss(aux["reward_pred"], reward_target_step)
                self._last_debug_tensors[f"reward_pred[{step_idx}]"] = aux["reward_pred"].detach()
                self._last_debug_tensors[f"loss_reward_step[{step_idx}]"] = reward_loss_step.detach()
                if self.fail_on_nonfinite:
                    self._check_finite(f"reward_pred[{step_idx}]", aux["reward_pred"])
                    self._check_finite(f"loss_reward_step[{step_idx}]", reward_loss_step)
                loss_reward_steps.append(reward_loss_step)
            if self.perceptual_enabled and target_images is not None and "image_recon" in aux:
                image_target = target_images[:, step_idx, :, :, :]
                image_loss_step = F.l1_loss(aux["image_recon"], image_target)
                self._last_debug_tensors[f"image_recon[{step_idx}]"] = aux["image_recon"].detach()
                self._last_debug_tensors[f"image_target[{step_idx}]"] = image_target.detach()
                self._last_debug_tensors[f"loss_image_recon_step[{step_idx}]"] = image_loss_step.detach()
                if self.fail_on_nonfinite:
                    self._check_finite(f"image_recon[{step_idx}]", aux["image_recon"])
                    self._check_finite(f"image_target[{step_idx}]", image_target)
                    self._check_finite(f"loss_image_recon_step[{step_idx}]", image_loss_step)
                loss_image_recon_steps.append(image_loss_step)
                if self.perceptual_loss_fn is not None:
                    perceptual_loss_step = self.perceptual_loss_fn(aux["image_recon"], image_target)
                    self._last_debug_tensors[f"loss_perceptual_step[{step_idx}]"] = perceptual_loss_step.detach()
                    if self.fail_on_nonfinite:
                        self._check_finite(f"loss_perceptual_step[{step_idx}]", perceptual_loss_step)
                    loss_perceptual_steps.append(perceptual_loss_step)

            next_teacher_z = z_future[:, step_idx, :, :]
            if self.detach_target_latents:
                next_teacher_z = next_teacher_z.detach()
            teacher_z = torch.cat([teacher_z[:, 1:, ...], next_teacher_z.unsqueeze(1)], dim=1)
            if step_idx < rollout_horizon - 1:
                teacher_action = torch.cat(
                    [teacher_action[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                    dim=1,
                )

        loss_recon = torch.stack(loss_recon_steps).mean()
        loss_recon_weighted = self.reconstruction_weight * loss_recon
        loss_reward = (
            torch.stack(loss_reward_steps).mean()
            if loss_reward_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_reward_weighted = self.reward_weight * loss_reward
        loss_image_recon = (
            torch.stack(loss_image_recon_steps).mean()
            if loss_image_recon_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_image_recon_weighted = self.image_recon_weight * loss_image_recon
        loss_perceptual = (
            torch.stack(loss_perceptual_steps).mean()
            if loss_perceptual_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_perceptual_weighted = self.perceptual_weight * loss_perceptual
        loss_negative_action = (
            torch.stack(loss_negative_action_steps).mean()
            if loss_negative_action_steps
            else torch.tensor(0.0, device=self.device)
        )
        loss_negative_action_weighted = self.negative_action_contrastive_weight * loss_negative_action

        latent_for_reg = torch.cat([z_history, z_future], dim=1)
        loss_sigreg = torch.tensor(0.0, device=self.device)
        current_sigreg_weight = 0.0
        if self.sigreg_enabled:
            loss_sigreg = wm_core.compute_sigreg(latent_for_reg)
            current_sigreg_weight = self._current_sigreg_weight()
        loss_sigreg_weighted = current_sigreg_weight * loss_sigreg
        self._last_debug_tensors.update(
            {
                "loss_recon": loss_recon.detach(),
                "loss_reward": loss_reward.detach(),
                "loss_image_recon": loss_image_recon.detach(),
                "loss_perceptual": loss_perceptual.detach(),
                "loss_negative_action": loss_negative_action.detach(),
                "loss_negative_action_weighted": loss_negative_action_weighted.detach(),
                "loss_sigreg": loss_sigreg.detach(),
                "loss_sigreg_weighted": loss_sigreg_weighted.detach(),
            }
        )
        if self.fail_on_nonfinite:
            self._check_finite("loss_recon", loss_recon)
            self._check_finite("loss_reward", loss_reward)
            self._check_finite("loss_image_recon", loss_image_recon)
            self._check_finite("loss_perceptual", loss_perceptual)
            self._check_finite("loss_negative_action", loss_negative_action)
            self._check_finite("loss_negative_action_weighted", loss_negative_action_weighted)
            self._check_finite("loss_sigreg", loss_sigreg)
            self._check_finite("loss_sigreg_weighted", loss_sigreg_weighted)

        loss_action = torch.tensor(0.0, device=self.device)
        loss_action_weighted = torch.tensor(0.0, device=self.device)
        shared_backward = self.training_mode == "semi_supervised" and (not self.detach_idm_in_wm)

        if self.training_mode == "semi_supervised":
            self.idm_optimizer.zero_grad(set_to_none=True)
            mapped_action = self.action_mapper(pred_action)
            loss_action = F.mse_loss(mapped_action, gt_action_future[:, 0, :])
            loss_action_weighted = self.semi_supervised_weight * loss_action
            if not shared_backward:
                loss_action_weighted.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.idm.parameters()) + list(self.action_mapper.parameters()),
                    self.grad_clip_norm,
                )
                self.idm_optimizer.step()

        self.wm_optimizer.zero_grad(set_to_none=True)
        loss_wm_total = (
            loss_recon_weighted
            + loss_sigreg_weighted
            + loss_reward_weighted
            + loss_image_recon_weighted
            + loss_perceptual_weighted
            + loss_negative_action_weighted
        )
        self._last_debug_tensors["loss_wm_total"] = loss_wm_total.detach()
        if self.fail_on_nonfinite:
            self._check_finite("loss_wm_total", loss_wm_total)

        if shared_backward:
            total_loss = loss_wm_total + loss_action_weighted
            total_loss.backward()
            wm_grad_norm = torch.nn.utils.clip_grad_norm_(self._wm_clip_params, self.grad_clip_norm)
            self._last_wm_grad_norm = float(wm_grad_norm.item())
            torch.nn.utils.clip_grad_norm_(
                list(self.idm.parameters()) + list(self.action_mapper.parameters()),
                self.grad_clip_norm,
            )
            self.wm_optimizer.step()
            self.idm_optimizer.step()
        else:
            loss_wm_total.backward()
            wm_grad_norm = torch.nn.utils.clip_grad_norm_(self._wm_clip_params, self.grad_clip_norm)
            self._last_wm_grad_norm = float(wm_grad_norm.item())
            self.wm_optimizer.step()

        if self.training_mode == "unsupervised":
            torch.nn.utils.clip_grad_norm_(
                list(self.idm.parameters()) + list(self.action_mapper.parameters()),
                self.grad_clip_norm,
            )
            self.idm_optimizer.step()

        if self._ema_model is not None:
            _update_ema(target=self._ema_model, source=self.wm, decay=self._ema_decay)
        self.wm_scheduler.step()
        self.idm_scheduler.step()

        batch_loss = float(loss_wm_total.item()) + (
            self.semi_supervised_weight * float(loss_action.item())
            if self.training_mode == "semi_supervised"
            else 0.0
        )
        reward_pred_mean = (
            float(torch.cat(reward_pred_values).mean().item()) if reward_pred_values else 0.0
        )
        reward_target_mean = (
            float(reward_targets.mean().item()) if reward_targets is not None else 0.0
        )
        negative_action_dist_mean = (
            float(torch.stack(negative_action_dist_values).mean().item())
            if negative_action_dist_values
            else 0.0
        )
        ensemble_uncertainty_mean = (
            float(torch.stack(ensemble_uncertainty_values).mean().item())
            if ensemble_uncertainty_values
            else 0.0
        )
        return {
            "loss": batch_loss,
            "loss_recon": float(loss_recon.item()),
            "loss_action": float(loss_action.item()),
            "loss_sigreg": float(loss_sigreg.item()),
            "loss_sigreg_weighted": float(loss_sigreg_weighted.item()),
            "loss_reward": float(loss_reward.item()),
            "loss_image_recon": float(loss_image_recon.item()),
            "loss_perceptual": float(loss_perceptual.item()),
            "loss_negative_action": float(loss_negative_action.item()),
            "loss_negative_action_weighted": float(loss_negative_action_weighted.item()),
            "negative_action_dist_mean": negative_action_dist_mean,
            "ensemble_uncertainty_mean": ensemble_uncertainty_mean,
            "ensemble_size": float(wm_core.ensemble_size),
            "reward_pred_mean": reward_pred_mean,
            "reward_target_mean": reward_target_mean,
            "sigreg_weight": current_sigreg_weight,
            "grad_norm_wm": self._last_wm_grad_norm,
            "lr_wm": float(self.wm_scheduler.get_last_lr()[0]),
            "lr_qwen": float(self.wm_scheduler.get_last_lr()[1])
            if len(self.wm_scheduler.get_last_lr()) > 1
            else 0.0,
            "lr_idm": float(self.idm_scheduler.get_last_lr()[0]),
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.wm.state_dict(),
            "inverse_dynamics_state_dict": self.idm.state_dict(),
            "action_mapper_state_dict": self.action_mapper.state_dict(),
            "wm_optimizer_state_dict": self.wm_optimizer.state_dict(),
            "idm_optimizer_state_dict": self.idm_optimizer.state_dict(),
            "wm_scheduler_state_dict": self.wm_scheduler.state_dict(),
            "idm_scheduler_state_dict": self.idm_scheduler.state_dict(),
            "ema_model_state_dict": self._ema_model.state_dict() if self._ema_model is not None else None,
            "mode": self.training_mode,
        }

    def load_state(self, state: dict[str, Any], *, start_epoch: int = 0, global_step: int = 0) -> None:
        if "model_state_dict" in state:
            self.wm.load_state_dict(state["model_state_dict"])
        if "inverse_dynamics_state_dict" in state:
            self.idm.load_state_dict(state["inverse_dynamics_state_dict"])
        if "action_mapper_state_dict" in state:
            self.action_mapper.load_state_dict(state["action_mapper_state_dict"])
        if "wm_optimizer_state_dict" in state:
            self.wm_optimizer.load_state_dict(state["wm_optimizer_state_dict"])
        if "idm_optimizer_state_dict" in state:
            self.idm_optimizer.load_state_dict(state["idm_optimizer_state_dict"])
        if "wm_scheduler_state_dict" in state:
            self.wm_scheduler.load_state_dict(state["wm_scheduler_state_dict"])
        if "idm_scheduler_state_dict" in state:
            self.idm_scheduler.load_state_dict(state["idm_scheduler_state_dict"])
        if self._ema_model is not None and "ema_model_state_dict" in state:
            self._ema_model.load_state_dict(state["ema_model_state_dict"])
        self._epoch = start_epoch
        self._global_step = global_step

    def reset_step_counter(self) -> None:
        self._epoch = 0
        self._global_step = 0
