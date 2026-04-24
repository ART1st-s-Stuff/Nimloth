"""LeWM（Latent Energy World Model）实现。

参考 lucas-maes/le-wm 的 ARPredictor，使用 AdaLN-Zero 条件化和因果注意力的
Transformer 结构预测下一步 latent 序列。

核心区别于 CFMWorldModel：
- 输入展平为 [B, T, D]，使用因果注意力（CausalMask）而非双向 TransformerEncoder
- 动作条件通过 AdaLN-Zero 注入（6*dim 调制向量，零初始化确保从头训练稳定）
- 预测目标为整个序列而非仅当前步残差
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.core.interfaces import Model


# --------------------------------------------------------------------
# SIGReg（从 le-wm/module.py 移植，单 GPU）
# --------------------------------------------------------------------


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer（单 GPU）。"""

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# --------------------------------------------------------------------
# LeWM 基础模块（从 le-wm/module.py 移植并适配）
# --------------------------------------------------------------------


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
        # (B, T, H*D) -> (B, H, T, D)
        q, k, v = (
            t.reshape(x.size(0), -1, self.heads, x.size(-1) // self.heads).transpose(1, 2)
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
        self.cond_dim = input_dim  # Store original input_dim for conditioning projection

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        # 条件向量维度可能与 input_dim 不同，需要独立投影
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
        # x 已经具有 hidden_dim 维度（由 patch_proj 投影），input_proj 保持为 Identity
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)  # type: ignore[misc]
        return self.output_proj(self.norm(x))


# --------------------------------------------------------------------
# LeWMWorldModel
# --------------------------------------------------------------------


class LeWMWorldModel(nn.Module):
    """
    LeWM 自回归世界模型。

    输入：
        z_history: [B, H, P, D] 历史 latent 序列
        action_history: [B, H, A] 动作历史

    输出（predict_next）：
        pred_z_next: [B, P, D] 预测的下一时刻 latent

    与 CFMWorldModel 的主要区别：
        - 展平为 [B, H*P, D]，使用因果注意力的 AdaLN-Zero Transformer
        - 预测目标为绝对 latent 而非残差
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
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 256,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        expected_latent_dim = self.num_patches * self.token_dim
        if int(latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")

        mlp_dim = int(hidden_dim * mlp_ratio)
        T = history_len  # 序列长度

        self.pos_embedding = nn.Parameter(torch.randn(1, T * self.num_patches, self.token_dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 动作编码：[B, H, A] -> [B, H, token_dim]
        # Conv1d 处理 channel（action_dim -> token_dim），然后 reshape 让 MLP 只作用于 feature 维。
        self.action_embed_conv = nn.Conv1d(action_dim, self.token_dim, kernel_size=1, stride=1)
        self.action_embed_mlp = nn.Sequential(
            nn.Linear(self.token_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, self.token_dim),
        )

        # 图像 token 投影：每个 patch 独立投影
        self.patch_proj = nn.Linear(self.token_dim, hidden_dim)

        self.transformer = _LeWMTransformer(
            input_dim=token_dim,  # 条件向量维度为 token_dim
            hidden_dim=hidden_dim,
            output_dim=token_dim,  # 输出每个 token 的预测
            depth=num_layers,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=_LeWMConditionalBlock,
        )

        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

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

    def forward(self, z_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """
        返回预测的下一时刻 latent 序列：[B, P, D]
        """
        self._validate_inputs(z_history=z_history, action_history=action_history)

        B, H, P, D = z_history.shape

        # [B, H, P, D] -> [B, H*P, D]
        x = z_history.reshape(B, H * P, D)
        x = x + self.pos_embedding[:, :H * P, :]
        x = self.dropout(x)

        # 动作编码：[B, H, A] -> [B, H, token_dim]
        x_conv = self.action_embed_conv(action_history.transpose(1, 2))  # [B, token_dim, H]
        B2, C, H2 = x_conv.shape
        x_conv = x_conv.permute(0, 2, 1).reshape(B2 * H2, C)  # [B*H, token_dim]
        act_emb = self.action_embed_mlp(x_conv).reshape(B2, H2, C)  # [B, H, token_dim]
        # 动作条件复制到每个 patch token
        c = act_emb.unsqueeze(2).expand(-1, -1, P, -1).reshape(B, H * P, D)

        # 投影到 hidden_dim
        x = self.patch_proj(x)

        # Transformer 前向（因果注意力在内部处理）
        out = self.transformer(x, c)  # [B, H*P, D]

        # 取最后一帧的所有 patch token：[B, P, D]
        pred_z = out.reshape(B, H, P, D)[:, -1, :, :]
        return pred_z

    def predict_next(
        self, z_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        """返回预测的下一时刻绝对 latent"""
        return self.forward(z_history=z_history, action_history=action_history)

    def compute_sigreg(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """
        计算 SIGReg 正则损失。
        z_sequence: [B, T, P, D] 或 [B, T, D]
        """
        if z_sequence.dim() == 4:
            B, T, P, D = z_sequence.shape
            z_sequence = z_sequence.reshape(T, B, P * D)
        else:
            T, B, D = z_sequence.shape
        return self.sigreg(z_sequence)


# --------------------------------------------------------------------
# LeWMModel：封装 LeWMWorldModel + IDM + ActionMapper 的统一训练接口
# --------------------------------------------------------------------


def _update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for ema_p, src_p in zip(target.parameters(), source.parameters()):
        ema_p.mul_(decay).add_(src_p.detach(), alpha=(1.0 - decay))
    for ema_b, src_b in zip(target.buffers(), source.buffers()):
        ema_b.copy_(src_b)


class LeWMModel(Model):
    """
    LeWM 统一训练接口，接口与 WMModel（CFM）完全一致。

    区别：
        - 使用 LeWMWorldModel 而非 CFMWorldModel
        - 预测目标为绝对 latent（MSELoss），而非残差
        - 内部包含 SIGReg 正则
    """

    def __init__(
        self,
        *,
        wm: LeWMWorldModel,
        inverse_dynamics: nn.Module,
        action_mapper: nn.Module,
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
        self._ema_model: nn.Module | None = None
        self._ema_decay = ema_decay
        self._epoch = 0
        self._global_step = 0
        if self._ema_decay > 0:
            self._ema_model = copy.deepcopy(self.wm).to(device)
            self._ema_model.eval()
            for p in self._ema_model.parameters():
                p.requires_grad_(False)

    def _compute_wm_loss(
        self,
        pred_z: torch.Tensor,
        target_z: torch.Tensor,
        z_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 WM 损失（预测 vs 目标 MSELoss）+ SIGReg"""
        loss_recon = F.mse_loss(pred_z, target_z)
        loss_sigreg = torch.tensor(0.0, device=self.device)
        current_sigreg_weight = 0.0
        if self.sigreg_enabled:
            loss_sigreg = self.wm.compute_sigreg(z_sequence)
            if self.sigreg_target_weight > 0.0 and self.sigreg_warmup_steps > 0:
                current_sigreg_weight = self.sigreg_target_weight * min(
                    1.0, float(self._global_step + 1) / float(self.sigreg_warmup_steps)
                )
        return loss_recon, loss_sigreg, current_sigreg_weight

    @torch.no_grad()
    def eval_step(self, batch: Any) -> dict[str, Any]:
        z_history = batch["z_history"].to(self.device)
        action_history = batch["action_history"].to(self.device)
        z_future = batch["z_future"].to(self.device)
        gt_action_future = batch["gt_action_future"].to(self.device)

        pred_action = None
        if self.training_mode in {"unsupervised", "semi_supervised"}:
            pred_action = self.idm(z_history.detach() if self.training_mode == "semi_supervised" else z_history)
        rollout_horizon = int(z_future.size(1))

        teacher_z = z_history
        teacher_action = action_history.clone()
        loss_recon_steps: list[torch.Tensor] = []

        for step_idx in range(rollout_horizon):
            teacher_action[:, -1, :] = gt_action_future[:, step_idx, :]

            pred_z = self.wm.predict_next(teacher_z, teacher_action)
            target_z = z_future[:, step_idx, :, :]
            step_loss = F.mse_loss(pred_z, target_z)
            loss_recon_steps.append(step_loss)

            teacher_z = torch.cat([teacher_z[:, 1:, ...], z_future[:, step_idx, :, :].unsqueeze(1)], dim=1)
            if step_idx < rollout_horizon - 1:
                teacher_action = torch.cat(
                    [teacher_action[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                    dim=1,
                )

        loss_recon = torch.stack(loss_recon_steps).mean()
        loss_action = torch.tensor(0.0, device=self.device)
        if self.training_mode == "semi_supervised":
            mapped_action = self.action_mapper(pred_action)
            loss_action = F.mse_loss(mapped_action, gt_action_future[:, 0, :])

        return {
            "loss": float(loss_recon.item()) + self.semi_supervised_weight * float(loss_action.item()),
            "loss_recon": float(loss_recon.item()),
            "loss_action": float(loss_action.item()),
        }

    def train_step(self, batch: Any) -> dict[str, Any]:
        z_history = batch["z_history"].to(self.device)
        action_history = batch["action_history"].to(self.device)
        z_future = batch["z_future"].to(self.device)
        gt_action_future = batch["gt_action_future"].to(self.device)

        pred_action = None
        if self.training_mode in {"unsupervised", "semi_supervised"}:
            pred_action = self.idm(z_history.detach() if self.training_mode == "semi_supervised" else z_history)
        rollout_horizon = int(z_future.size(1))

        teacher_z = z_history
        teacher_action = action_history.clone()
        loss_recon_steps: list[torch.Tensor] = []

        for step_idx in range(rollout_horizon):
            teacher_action[:, -1, :] = gt_action_future[:, step_idx, :]

            pred_z = self.wm.predict_next(teacher_z, teacher_action)
            target_z = z_future[:, step_idx, :, :]
            step_loss = F.mse_loss(pred_z, target_z)
            loss_recon_steps.append(step_loss)

            teacher_z = torch.cat([teacher_z[:, 1:, ...], z_future[:, step_idx, :, :].unsqueeze(1)], dim=1)
            if step_idx < rollout_horizon - 1:
                teacher_action = torch.cat(
                    [teacher_action[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                    dim=1,
                )

        loss_recon = torch.stack(loss_recon_steps).mean()
        loss_recon_weighted = self.reconstruction_weight * loss_recon

        # SIGReg
        latent_for_reg = torch.cat([z_history, z_future], dim=1)
        loss_sigreg, current_sigreg_weight = torch.tensor(0.0, device=self.device), 0.0
        if self.sigreg_enabled:
            loss_sigreg = self.wm.compute_sigreg(latent_for_reg)

        # Action loss
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
        loss_wm_total = loss_recon_weighted + current_sigreg_weight * loss_sigreg

        if shared_backward:
            total_loss = loss_wm_total + loss_action_weighted
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(
                list(self.idm.parameters()) + list(self.action_mapper.parameters()),
                self.grad_clip_norm,
            )
            self.wm_optimizer.step()
            self.idm_optimizer.step()
        else:
            loss_wm_total.backward()
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip_norm)
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
        return {
            "loss": batch_loss,
            "loss_recon": float(loss_recon.item()),
            "loss_action": float(loss_action.item()),
            "loss_sigreg": float(loss_sigreg.item()),
            "sigreg_weight": current_sigreg_weight,
            "lr_wm": float(self.wm_scheduler.get_last_lr()[0]),
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
