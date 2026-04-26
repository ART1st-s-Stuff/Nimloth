"""CFM（条件流匹配）世界模型实现。"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from src.core.interfaces import Model
from src.wm.losses import action_supervision_loss, sigreg_loss, wm_cfm_loss


def _update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for ema_p, src_p in zip(target.parameters(), source.parameters()):
        ema_p.mul_(decay).add_(src_p.detach(), alpha=(1.0 - decay))
    for ema_b, src_b in zip(target.buffers(), source.buffers()):
        ema_b.copy_(src_b)


def _cfm_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN 调制：x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _CFMAdaLNZeroBlock(nn.Module):
    """带 AdaLN-Zero 条件化的双向 Transformer Block。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(mlp_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_dim), hidden_dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        gate_msa = gate_msa.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        attn_out, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
        x = x + gate_msa * _cfm_modulate(self.attn_norm(attn_out), shift_msa, scale_msa)
        x = x + gate_mlp * _cfm_modulate(self.mlp_norm(self.mlp(self.mlp_norm(x))), shift_mlp, scale_mlp)
        return x


class ActionConditioning(nn.Module):
    """Film 条件化。"""

    def __init__(self, hidden_dim: int, action_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode.strip().lower()
        if self.mode not in {"adaln", "film"}:
            raise ValueError(f"不支持的 conditioning.mode={mode}")
        self.norm = nn.LayerNorm(hidden_dim)
        self.modulator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, hidden_dim * 2),
        )
        nn.init.constant_(self.modulator[-1].weight, 0)
        nn.init.constant_(self.modulator[-1].bias, 0)

    def forward(self, tokens: torch.Tensor, action_cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.modulator(action_cond).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + gamma) * self.norm(tokens) + beta


class CFMWorldModel(nn.Module):
    """条件流匹配世界模型：学习速度场 v_theta(x_t, t | history, action)。

    SIGReg 支持（方案1）：
        - 支持 SIGRegEncoderDecoder 将输入编码到 SIGReg latent space
        - 在该空间应用 SIGReg 正则化训练 encoder
        - Decoder 将预测结果映射回原始空间
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
        dropout: float,
        conditioning_mode: str = "adaln",
        action_input_mode: str = "adaln",
        flow_matching_variant: str = "rectified_flow",
        solver: str = "heun",
        num_integration_steps: int = 16,
        t_eps: float = 1e-3,
        noise_std: float = 1.0,
        x0_source: str = "current_latent",
        sigreg_enabled: bool = False,
        sigreg_latent_dim: int | None = None,
        sigreg_encoder_hidden_dim: int | None = None,
        sigreg_encoder_num_layers: int = 2,
        sigreg_num_quadrature_points: int = 16,
        sigreg_num_proj: int = 256,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_kernel_sigma: float = 1.0,
    ) -> None:
        from src.wm.sigreg_modules import SIGRegEncoderDecoder, SIGReg

        super().__init__()
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        self.sigreg_enabled = sigreg_enabled
        if self.num_patches <= 0 or self.token_dim <= 0:
            raise ValueError(f"非法 patch 配置: num_patches={self.num_patches}, token_dim={self.token_dim}")
        expected_latent_dim = self.num_patches * self.token_dim
        if int(self.latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")

        self.sigreg_ed: SIGRegEncoderDecoder | None = None
        self.sigreg: SIGReg | None = None
        if sigreg_enabled:
            ed_latent_dim = sigreg_latent_dim or token_dim
            self.sigreg_ed = SIGRegEncoderDecoder(
                token_dim=token_dim,
                sigreg_latent_dim=ed_latent_dim,
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
            transformer_token_dim = ed_latent_dim
        else:
            transformer_token_dim = token_dim

        self.token_proj = nn.Linear(transformer_token_dim, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(1, history_len, 1, hidden_dim))
        self.patch_embedding = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_dim))
        action_mode = action_input_mode.strip().lower()
        action_alias = {
            "explicit_token_concat": "token",
            "token": "token",
            "modulation": conditioning_mode.strip().lower(),
            "film": "film",
            "adaln": "adaln",
        }
        self.action_input_mode = action_alias.get(action_mode, action_mode)
        if self.action_input_mode not in {"token", "film", "adaln"}:
            raise ValueError(f"不支持的 action_input_mode={action_input_mode}")
        self.flow_matching_variant = flow_matching_variant.strip().lower()
        if self.flow_matching_variant != "rectified_flow":
            raise ValueError(f"不支持的 flow_matching.variant={flow_matching_variant}")
        self.x0_source = x0_source.strip().lower()
        if self.x0_source != "current_latent":
            raise ValueError(f"不支持的 flow_matching.x0_source={x0_source}")
        self.solver = solver.strip().lower()
        if self.solver not in {"euler", "heun"}:
            raise ValueError(f"不支持的 flow_matching.solver={solver}")
        self.num_integration_steps = max(1, int(num_integration_steps))
        self.t_eps = float(t_eps)
        self.fm_noise_std = float(noise_std)
        self.conditioning_film = ActionConditioning(hidden_dim=hidden_dim, action_dim=action_dim, mode="film")
        self.xt_proj = nn.Linear(transformer_token_dim, hidden_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_token_proj = nn.Linear(action_dim, hidden_dim)
        self.action_token_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.history_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.history_cross_attention_norm = nn.LayerNorm(hidden_dim)
        mlp_dim = hidden_dim * 4
        self.encoder_layers = nn.ModuleList([
            _CFMAdaLNZeroBlock(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.action_cond_proj = nn.Linear(action_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, transformer_token_dim),
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

    def _encode_history_tokens(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        action_cond: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(z_history=z_history, action_history=action_history)
        x = self.token_proj(z_history)
        x = x + self.time_embedding[:, : self.history_len, :, :] + self.patch_embedding[:, :, : self.num_patches, :]
        if self.history_len <= 1:
            raise ValueError("history_len 必须大于 1，才能进行 history cross-attention。")
        history_tokens = x[:, :-1, :, :].reshape(x.size(0), (self.history_len - 1) * self.num_patches, x.size(-1))
        c = self.action_cond_proj(action_cond)
        for layer in self.encoder_layers:
            history_tokens = layer(history_tokens, c)
        return self.encoder_norm(history_tokens)

    @staticmethod
    def _expand_time(t: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t.dim() == 0:
            t = t.view(1, 1, 1).expand(batch_size, 1, 1)
        elif t.dim() == 1:
            t = t.view(batch_size, 1, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        return t.to(device=device, dtype=dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """预测速度场 v_theta(x_t, t | history, action)。"""
        if self.sigreg_ed is not None:
            B, P, D = x_t.shape
            x_t_encoded = self.sigreg_ed.encode(x_t)
            z_history_flat = z_history.reshape(B, -1, D)
            z_history_encoded = self.sigreg_ed.encode(z_history_flat)
            z_history = z_history_encoded.reshape_as(z_history_flat).reshape_as(z_history)
        else:
            x_t_encoded = x_t

        action_cond = action_history[:, -1, :]
        history_tokens = self._encode_history_tokens(
            z_history=z_history,
            action_history=action_history,
            action_cond=action_cond,
        )
        t = self._expand_time(t=t, batch_size=x_t.size(0), device=x_t.device, dtype=x_t.dtype)
        t_cond = self.t_embed(t.reshape(x_t.size(0), 1)).unsqueeze(1)
        query_tokens = self.xt_proj(x_t_encoded) + t_cond
        cross_out, _ = self.history_cross_attention(
            query=query_tokens,
            key=history_tokens,
            value=history_tokens,
            need_weights=False,
        )
        conditioned = self.history_cross_attention_norm(query_tokens + cross_out)
        if self.action_input_mode == "token":
            action_token = self.action_token_proj(action_cond).unsqueeze(1) + self.action_token_embedding
            conditioned = conditioned + action_token
        elif self.action_input_mode == "film":
            conditioned = self.conditioning_film(tokens=conditioned, action_cond=action_cond)
        elif self.action_input_mode == "adaln":
            pass
        output = self.head(conditioned)

        if self.sigreg_ed is not None:
            output = self.sigreg_ed.decode(output)

        return output

    def compute_sigreg(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """计算 SIGReg 正则损失。"""
        if self.sigreg is None or self.sigreg_ed is None:
            raise RuntimeError("SIGReg 未启用，无法计算 SIGReg 损失")

        if z_sequence.dim() == 4:
            B, T, P, D = z_sequence.shape
            z_flat = z_sequence.reshape(B, T * P, D)
        else:
            z_flat = z_sequence

        z_encoded = self.sigreg_ed.encode(z_flat)

        if z_sequence.dim() == 4:
            z_encoded = z_encoded.reshape(T, B, P, -1)
        else:
            z_encoded = z_encoded.permute(1, 0, 2)

        return self.sigreg(z_encoded)

    def _integrate_next(
        self,
        x0: torch.Tensor,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        *,
        solver: str,
        num_steps: int,
    ) -> torch.Tensor:
        batch_size = x0.size(0)
        dt = (1.0 - self.t_eps) / float(num_steps)
        x = x0
        for step_idx in range(num_steps):
            t_val = self.t_eps + float(step_idx) * dt
            t = torch.full((batch_size, 1, 1), t_val, device=x.device, dtype=x.dtype)
            v_t = self.forward(x_t=x, t=t, z_history=z_history, action_history=action_history)
            if solver == "euler":
                x = x + dt * v_t
                continue
            t_next_val = min(1.0, t_val + dt)
            x_euler = x + dt * v_t
            t_next = torch.full((batch_size, 1, 1), t_next_val, device=x.device, dtype=x.dtype)
            v_next = self.forward(x_t=x_euler, t=t_next, z_history=z_history, action_history=action_history)
            x = x + 0.5 * dt * (v_t + v_next)
        return x

    def predict_next(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        *,
        solver: str | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """通过 ODE 积分从当前帧 latent 采样下一步 latent。"""
        self._validate_inputs(z_history=z_history, action_history=action_history)
        solver_name = (solver or self.solver).strip().lower()
        if solver_name not in {"euler", "heun"}:
            raise ValueError(f"不支持的 solver={solver_name}")
        steps = max(1, int(num_steps if num_steps is not None else self.num_integration_steps))
        x0 = z_history[:, -1, :, :]
        return self._integrate_next(
            x0=x0,
            z_history=z_history,
            action_history=action_history,
            solver=solver_name,
            num_steps=steps,
        )


class WMModel(Model):
    """封装 WM/IDM/ActionMapper 及 EMA 的统一训练接口。"""

    def __init__(
        self,
        *,
        wm: CFMWorldModel,
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
        sigreg_num_projections: int = 256,
        sigreg_num_quadrature_points: int = 16,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_kernel_sigma: float = 1.0,
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
        self.sigreg_num_projections = sigreg_num_projections
        self.sigreg_num_quadrature_points = sigreg_num_quadrature_points
        self.sigreg_t_min = sigreg_t_min
        self.sigreg_t_max = sigreg_t_max
        self.sigreg_kernel_sigma = sigreg_kernel_sigma
        self._ema_model: nn.Module | None = None
        self._ema_decay = ema_decay
        self._epoch = 0
        self._global_step = 0
        if self._ema_decay > 0:
            self._ema_model = copy.deepcopy(self.wm).to(device)
            self._ema_model.eval()
            for p in self._ema_model.parameters():
                p.requires_grad_(False)
        self.wm.train()
        self.idm.train()
        self.action_mapper.train()

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
            target_z = z_future[:, step_idx, :, :]
            current_z = teacher_z[:, -1, :, :]
            t = torch.rand(target_z.size(0), 1, 1, device=self.device, dtype=target_z.dtype)
            x_t = (1.0 - t) * current_z + t * target_z
            target_velocity = target_z - current_z
            pred_velocity = self.wm(
                x_t=x_t,
                t=t,
                z_history=teacher_z,
                action_history=teacher_action,
            )
            step_loss = wm_cfm_loss(pred_velocity, target_velocity)
            loss_recon_steps.append(step_loss)
            teacher_z = torch.cat([teacher_z[:, 1:, ...], z_future[:, step_idx, :].unsqueeze(1)], dim=1)
            if step_idx < rollout_horizon - 1:
                teacher_action = torch.cat(
                    [teacher_action[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                    dim=1,
                )
        loss_recon = torch.stack(loss_recon_steps).mean()
        loss_action = torch.tensor(0.0, device=self.device)
        if self.training_mode == "semi_supervised":
            mapped_action = self.action_mapper(pred_action)
            loss_action = action_supervision_loss(mapped_action, gt_action_future[:, 0, :])
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
            target_z = z_future[:, step_idx, :, :]
            current_z = teacher_z[:, -1, :, :]
            t = torch.rand(target_z.size(0), 1, 1, device=self.device, dtype=target_z.dtype)
            x_t = (1.0 - t) * current_z + t * target_z
            target_velocity = target_z - current_z
            pred_velocity = self.wm(
                x_t=x_t,
                t=t,
                z_history=teacher_z,
                action_history=teacher_action,
            )
            step_loss = wm_cfm_loss(pred_velocity, target_velocity)
            loss_recon_steps.append(step_loss)
            teacher_z = torch.cat([teacher_z[:, 1:, ...], z_future[:, step_idx, :].unsqueeze(1)], dim=1)
            if step_idx < rollout_horizon - 1:
                teacher_action = torch.cat(
                    [teacher_action[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                    dim=1,
                )
        loss_recon = torch.stack(loss_recon_steps).mean()
        loss_recon_weighted = self.reconstruction_weight * loss_recon
        latent_for_reg = torch.cat([z_history, z_future], dim=1)
        loss_sigreg = torch.tensor(0.0, device=self.device)
        current_sigreg_weight = 0.0
        if self.sigreg_enabled:
            if hasattr(self.wm, "compute_sigreg") and callable(getattr(self.wm, "compute_sigreg")):
                loss_sigreg = self.wm.compute_sigreg(latent_for_reg)
            else:
                loss_sigreg = sigreg_loss(
                    latent_for_reg,
                    num_projections=self.sigreg_num_projections,
                    num_quadrature_points=self.sigreg_num_quadrature_points,
                    t_min=self.sigreg_t_min,
                    t_max=self.sigreg_t_max,
                    kernel_sigma=self.sigreg_kernel_sigma,
                )
            if self.sigreg_target_weight > 0.0 and self.sigreg_warmup_steps > 0:
                current_sigreg_weight = self.sigreg_target_weight * min(
                    1.0, float(self._global_step + 1) / float(self.sigreg_warmup_steps)
                )
        loss_action = torch.tensor(0.0, device=self.device)
        loss_action_weighted = torch.tensor(0.0, device=self.device)
        shared_backward = self.training_mode == "semi_supervised" and (not self.detach_idm_in_wm)
        if self.training_mode == "semi_supervised":
            self.idm_optimizer.zero_grad(set_to_none=True)
            mapped_action = self.action_mapper(pred_action)
            loss_action = action_supervision_loss(mapped_action, gt_action_future[:, 0, :])
            loss_action_weighted = self.semi_supervised_weight * loss_action
            if not shared_backward:
                loss_action_weighted.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.idm.parameters()) + list(self.action_mapper.parameters()),
                    self.grad_clip_norm,
                )
                self.idm_optimizer.step()
        elif self.training_mode == "unsupervised":
            self.idm_optimizer.zero_grad(set_to_none=True)
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