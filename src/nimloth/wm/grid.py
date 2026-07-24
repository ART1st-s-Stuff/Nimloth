"""16-slot spatial state and H-step temporal-spatial world model."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nimloth.wm._vendor_lewm import Embedder, MLP, modulate
from nimloth.wm.lewm import SafeBatchNorm1d
from nimloth.wm.model import WorldModel
from nimloth.wm.value_head import ValueHead


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


class SharedSlotProjector(nn.Module):
    """对每个 query slot 使用同一个 SFT1 projector，保留空间轴。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        *,
        grid_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.qwen_hidden_dim = self.input_dim
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.grid_tokens = int(grid_tokens)
        self.latent_token_count = self.grid_tokens
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_tokens, self.input_dim)
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected:
            raise ValueError(
                "SharedSlotProjector expected hidden shape "
                f"(B, {self.grid_tokens}, {self.input_dim}), "
                f"got {tuple(hidden.shape)}"
            )
        return self.net(hidden.to(dtype=next(self.parameters()).dtype))


def load_sft1_slot_projector(
    checkpoint: str | Path,
    *,
    qwen_hidden_dim: int,
    state_dim: int = 1024,
    grid_tokens: int = 16,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> SharedSlotProjector:
    """加载 SFT1 明确保存的 row-major 16-slot 表征接口。"""

    checkpoint = Path(checkpoint)
    config_path = checkpoint / "grid_state_config.json"
    state_path = checkpoint / "slot_projector.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"SFT1 grid-state interface is incomplete: {checkpoint}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "grid_tokens": int(grid_tokens),
        "qwen_hidden_dim": int(qwen_hidden_dim),
        "state_dim": int(state_dim),
        "shared_slot_projector": True,
        "ordering": "row_major",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"SFT1 grid-state interface mismatch: {mismatches}")

    projector = SharedSlotProjector(
        input_dim=int(config["qwen_hidden_dim"]),
        hidden_dim=int(config["projector_hidden_dim"]),
        output_dim=int(config["state_dim"]),
        grid_tokens=int(config["grid_tokens"]),
    )
    projector.load_state_dict(
        torch.load(state_path, map_location=map_location, weights_only=True)
    )
    projector.requires_grad_(False).eval()
    if dtype is not None:
        projector.to(dtype=dtype)
    return projector


class _LeWMGridMLP(nn.Module):
    """逐 slot 应用 refactor 前权威版本的 LeWM MLP。"""

    def __init__(self, *, emb_dim: int = 1024, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = MLP(
            input_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.emb_dim,
            norm_fn=SafeBatchNorm1d,
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 3 or grid.shape[-1] != self.emb_dim:
            raise ValueError(
                f"expected grid (B, N, {self.emb_dim}), got {tuple(grid.shape)}"
            )
        batch, slots, _ = grid.shape
        weight = next(self.parameters())
        output = self.net(
            grid.reshape(batch * slots, self.emb_dim).to(dtype=weight.dtype)
        )
        return output.reshape(batch, slots, self.emb_dim)


class LeWMGridEncoder(_LeWMGridMLP):
    """DINO-grid SFT2 的 trainable online state encoder。"""


class LeWMGridDecoder(_LeWMGridMLP):
    """把 predicted latent grid 解码到 DINO token 空间。"""


class GridStateProjector(nn.Module):
    """冻结 SFT1 slot projector，训练逐-slot LeWM encoder。"""

    def __init__(
        self,
        slot_projector: SharedSlotProjector,
        online_encoder: LeWMGridEncoder,
    ) -> None:
        super().__init__()
        self.slot_projector = slot_projector.requires_grad_(False)
        self.online_encoder = online_encoder
        self.qwen_hidden_dim = slot_projector.qwen_hidden_dim
        self.input_dim = slot_projector.qwen_hidden_dim
        self.latent_token_count = slot_projector.grid_tokens
        self.grid_tokens = slot_projector.grid_tokens
        self.emb_dim = online_encoder.emb_dim

    def train(self, mode: bool = True) -> "GridStateProjector":
        super().train(mode)
        self.slot_projector.eval()
        return self

    def project_slots(self, hidden: torch.Tensor) -> torch.Tensor:
        """保留对 Qwen hidden 的输入梯度，但不更新 SFT1 projector 权重。"""

        return self.slot_projector(hidden).float()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(self.project_slots(hidden))


class EMATargetGridEncoder(nn.Module):
    """online grid encoder 的 frozen EMA target。"""

    def __init__(self, online_encoder: LeWMGridEncoder, *, decay: float = 0.99) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.encoder = copy.deepcopy(online_encoder)
        self.encoder.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "EMATargetGridEncoder":
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def update(self, online_encoder: LeWMGridEncoder) -> None:
        online = _unwrap(online_encoder)
        for target_parameter, online_parameter in zip(
            self.encoder.parameters(),
            online.parameters(),
            strict=True,
        ):
            target_parameter.mul_(self.decay).add_(
                online_parameter.detach(),
                alpha=1.0 - self.decay,
            )
        for target_buffer, online_buffer in zip(
            self.encoder.buffers(),
            online.buffers(),
            strict=True,
        ):
            if target_buffer.is_floating_point():
                target_buffer.mul_(self.decay).add_(
                    online_buffer.detach(),
                    alpha=1.0 - self.decay,
                )
            else:
                target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def forward(self, projected_grid: torch.Tensor) -> torch.Tensor:
        return self.encoder(projected_grid).detach()


class _GridAttention(nn.Module):
    """与旧 spatial predictor 同参数结构、支持显式时间因果 mask 的 attention。"""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float,
    ) -> None:
        super().__init__()
        inner_dim = int(dim_head) * int(heads)
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(value)
        qkv = self.to_qkv(normalized).chunk(3, dim=-1)
        query, key, encoded_value = (
            rearrange(tensor, "b t (h d) -> b h t d", h=self.heads)
            for tensor in qkv
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            encoded_value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.to_out(rearrange(output, "b h t d -> b t (h d)"))


class _GridFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _TemporalSpatialConditionalBlock(nn.Module):
    """LeWM AdaLN-zero block：空间双向、时间只看当前及更早状态。"""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = _GridAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.mlp = _GridFeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        state = state + gate_msa * self.attn(
            modulate(self.norm1(state), shift_msa, scale_msa),
            attention_mask=attention_mask,
        )
        state = state + gate_mlp * self.mlp(
            modulate(self.norm2(state), shift_mlp, scale_mlp)
        )
        return state


@dataclass(frozen=True)
class GridPredictorConfig:
    grid_tokens: int = 16
    emb_dim: int = 1024
    action_dim: int = 8
    history_size: int = 4
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1


class TemporalSpatialGridPredictor(nn.Module):
    """对每个历史位置联合预测完整 next-state grid。

    同一时间位置的 16 个 slot 双向交互；时间 mask 只允许读取当前和更早位置。
    当 H=1 时，计算图与 refactor 前的 spatial predictor 相同，因而旧 ID33
    predictor 除新增 temporal position 外可以严格 warm start。
    """

    def __init__(self, config: GridPredictorConfig) -> None:
        super().__init__()
        if config.grid_tokens < 1 or config.history_size < 1:
            raise ValueError("grid_tokens and history_size must be positive")
        self.config = config
        self.grid_tokens = config.grid_tokens
        self.emb_dim = config.emb_dim
        self.action_dim = config.action_dim
        self.spatial_position = nn.Parameter(
            torch.randn(1, config.grid_tokens, config.emb_dim)
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, config.history_size, 1, config.emb_dim)
        )
        self.action_encoder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.action_dim,
            emb_dim=config.emb_dim,
        )
        self.layers = nn.ModuleList(
            [
                _TemporalSpatialConditionalBlock(
                    config.emb_dim,
                    heads=config.heads,
                    dim_head=config.dim_head,
                    mlp_dim=config.mlp_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.emb_dim)

    def _attention_mask(
        self,
        time_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        token_times = torch.arange(time_steps, device=device).repeat_interleave(
            self.grid_tokens
        )
        allowed = token_times.unsqueeze(0) <= token_times.unsqueeze(1)
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, float("-inf"))

    def predict_sequence(
        self,
        state_context: torch.Tensor,
        action_context: torch.Tensor,
    ) -> torch.Tensor:
        expected_tail = (self.grid_tokens, self.emb_dim)
        if state_context.ndim != 4 or tuple(state_context.shape[2:]) != expected_tail:
            raise ValueError(
                "grid state context must have shape "
                f"(B,T,{self.grid_tokens},{self.emb_dim}), "
                f"got {tuple(state_context.shape)}"
            )
        if action_context.shape != state_context.shape[:2]:
            raise ValueError(
                "grid action context must have shape (B,T), "
                f"got {tuple(action_context.shape)}"
            )
        time_steps = state_context.shape[1]
        if not 1 <= time_steps <= self.config.history_size:
            raise ValueError(
                f"context T={time_steps} exceeds H={self.config.history_size}"
            )
        if torch.any(action_context < 0) or torch.any(
            action_context >= self.action_dim
        ):
            raise ValueError("action index is outside configured action_dim")

        dtype = self.spatial_position.dtype
        actions = F.one_hot(
            action_context.long(),
            num_classes=self.action_dim,
        ).to(dtype=dtype)
        condition = self.action_encoder(actions).unsqueeze(2).expand(
            -1,
            -1,
            self.grid_tokens,
            -1,
        )
        state = (
            state_context.to(dtype=dtype)
            + self.spatial_position.unsqueeze(1)
            + self.temporal_position[:, :time_steps]
        )
        batch = state.shape[0]
        state = state.reshape(batch, time_steps * self.grid_tokens, self.emb_dim)
        condition = condition.reshape(
            batch,
            time_steps * self.grid_tokens,
            self.emb_dim,
        ).to(dtype=dtype)
        attention_mask = self._attention_mask(
            time_steps,
            device=state.device,
            dtype=state.dtype,
        )
        for block in self.layers:
            state = block(
                state,
                condition,
                attention_mask=attention_mask,
            )
        return self.norm(state).reshape(
            batch,
            time_steps,
            self.grid_tokens,
            self.emb_dim,
        )

    def forward(
        self,
        state: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim == 3 and action_indices.ndim == 1:
            return self.predict_sequence(
                state.unsqueeze(1),
                action_indices.unsqueeze(1),
            )[:, -1]
        return self.predict_sequence(state, action_indices)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "predictor.pt")
        (path / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "TemporalSpatialGridPredictor":
        path = Path(path)
        config_path = path / "config.json"
        state_path = path / "predictor.pt"
        if not config_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"incomplete grid WM checkpoint: {path}")
        module = cls(
            GridPredictorConfig(
                **json.loads(config_path.read_text(encoding="utf-8"))
            )
        )
        module.load_state_dict(
            torch.load(state_path, map_location=map_location, weights_only=True)
        )
        return module


class GridWorldModel(WorldModel):
    """16-slot WM：在线/EMA state、temporal-spatial predictor、DINO decoder。"""

    def __init__(
        self,
        *,
        state_proj: GridStateProjector,
        target_encoder: EMATargetGridEncoder,
        wm_predictor: TemporalSpatialGridPredictor,
        dino_decoder: LeWMGridDecoder,
        value_head: ValueHead,
        train_dino_decoder: bool = True,
        update_target_encoder: bool = True,
    ) -> None:
        super().__init__(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
        )
        self.target_encoder = target_encoder
        self.dino_decoder = dino_decoder
        self.train_dino_decoder = bool(train_dino_decoder)
        self.update_target_encoder = bool(update_target_encoder)
        if not self.train_dino_decoder:
            self.dino_decoder.requires_grad_(False).eval()

    def project_target_state(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        projector = _unwrap(self.state_proj)
        with torch.no_grad():
            projected = projector.project_slots(qwen_hidden)
            return self.target_encoder(projected).float()

    def project_training_state_sequences(
        self,
        qwen_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在线 grid 作为 predictor 输入，冻结 EMA grid 作为 WM 目标。"""

        if qwen_hidden.ndim < 4:
            raise ValueError(
                "grid state sequence hidden must have shape (B,T,N,D), "
                f"got {tuple(qwen_hidden.shape)}"
            )
        batch_size, time_steps = qwen_hidden.shape[:2]
        flat_hidden = qwen_hidden.flatten(0, 1)
        online = self.project_state(flat_hidden)
        target = self.project_target_state(flat_hidden)
        output_shape = (batch_size, time_steps, *online.shape[1:])
        return online.reshape(output_shape), target.reshape(output_shape)

    def sigreg_state(self, state: torch.Tensor) -> torch.Tensor:
        """单个时刻先对 slots 做 mean pooling，交给公共 SFT2 SIGReg。"""

        if state.ndim != 3:
            raise ValueError(
                "grid SIGReg state must have shape (B,N,D), "
                f"got {tuple(state.shape)}"
            )
        return state.mean(dim=-2)

    def sigreg_state_sequence(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """RL sequence 每个时刻先对 slots 做 mean pooling。"""

        if state_sequence.ndim != 4:
            raise ValueError(
                "grid SIGReg state must have shape (B,T,N,D), "
                f"got {tuple(state_sequence.shape)}"
            )
        return state_sequence.mean(dim=-2)

    def decode_prediction(self, predicted_state: torch.Tensor) -> torch.Tensor:
        return self.dino_decoder(predicted_state).float()

    def predict_action_values(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim < 3:
            raise ValueError(
                "grid value input must have shape (...,N,D), "
                f"got {tuple(state.shape)}"
            )
        return self.value_head(state.mean(dim=-2)).float()

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        if not self.update_target_encoder:
            return
        projector = _unwrap(self.state_proj)
        self.target_encoder.update(projector.online_encoder)
        if dist.is_available() and dist.is_initialized():
            for buffer in self.target_encoder.buffers():
                dist.broadcast(buffer, src=0)

    @property
    def trainable_modules(self) -> tuple[nn.Module, ...]:
        modules = (
            self.state_proj,
            self.wm_predictor,
            self.value_head,
        )
        if self.train_dino_decoder:
            return (*modules, self.dino_decoder)
        return modules

    @property
    def synchronized_modules(self) -> tuple[nn.Module, ...]:
        return self.trainable_modules

    def unwrapped(self) -> "GridWorldModel":
        return GridWorldModel(
            state_proj=_unwrap(self.state_proj),
            target_encoder=self.target_encoder,
            wm_predictor=_unwrap(self.wm_predictor),
            dino_decoder=_unwrap(self.dino_decoder),
            value_head=_unwrap(self.value_head),
            train_dino_decoder=self.train_dino_decoder,
            update_target_encoder=self.update_target_encoder,
        )

    def save_checkpoint_extras(self, output_dir: Path) -> None:
        torch.save(
            self.target_encoder.state_dict(),
            output_dir / "target_grid_encoder_ema.pt",
        )
        torch.save(
            _unwrap(self.dino_decoder).state_dict(),
            output_dir / "dino_grid_decoder.pt",
        )
        metadata = {
            "format": "sft2_dino_grid_v1",
            "grid_tokens": int(_unwrap(self.state_proj).grid_tokens),
            "state_dim": int(_unwrap(self.state_proj).emb_dim),
            "ema_decay": float(self.target_encoder.decay),
        }
        (output_dir / "dino_grid_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def load_checkpoint_extras(
        self,
        checkpoint_dir: Path,
        *,
        map_location: torch.device,
    ) -> None:
        config_path = checkpoint_dir / "dino_grid_config.json"
        target_path = checkpoint_dir / "target_grid_encoder_ema.pt"
        decoder_path = checkpoint_dir / "dino_grid_decoder.pt"
        missing = [
            str(path)
            for path in (config_path, target_path, decoder_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"incomplete DINO grid checkpoint extras: {missing}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        projector = _unwrap(self.state_proj)
        expected = {
            "format": "sft2_dino_grid_v1",
            "grid_tokens": int(projector.grid_tokens),
            "state_dim": int(projector.emb_dim),
            "ema_decay": float(self.target_encoder.decay),
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"DINO grid checkpoint mismatch: {mismatches}")
        self.target_encoder.load_state_dict(
            torch.load(target_path, map_location=map_location, weights_only=True)
        )
        _unwrap(self.dino_decoder).load_state_dict(
            torch.load(decoder_path, map_location=map_location, weights_only=True)
        )


def warm_start_legacy_grid_components(
    world_model: GridWorldModel,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """从 ID33 格式加载可证明兼容的 grid 辅助权重。

    旧 predictor 只有一个 spatial step。当前 predictor 复用它的全部参数，
    新增的 H-step temporal position 保持零初始化；这不是 resume。
    """

    checkpoint = Path(checkpoint)
    state_path = checkpoint / "config.json"
    required = (
        state_path,
        checkpoint / "online_encoder.pt",
        checkpoint / "target_encoder_ema.pt",
        checkpoint / "grid_wm.pt",
        checkpoint / "dino_decoder.pt",
        checkpoint / "value_head" / "value_head.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"legacy DINO grid warm start is incomplete: {missing}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    invariants = {
        "grid_tokens": 16,
        "state_dim": 1024,
        "dino_weight": 0.5,
        "sigreg_weight": 0.1,
        "ema_decay": 0.99,
        "encoder_decoder": "LeWM_MLP_per_slot",
        "dino_target": "next_rgb_final_patch_adaptive_pool_4x4",
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in invariants.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"legacy DINO grid warm-start mismatch: {mismatches}")

    projector = _unwrap(world_model.state_proj)
    projector.online_encoder.load_state_dict(
        torch.load(
            checkpoint / "online_encoder.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    world_model.target_encoder.load_state_dict(
        torch.load(
            checkpoint / "target_encoder_ema.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    _unwrap(world_model.dino_decoder).load_state_dict(
        torch.load(
            checkpoint / "dino_decoder.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    predictor = _unwrap(world_model.wm_predictor)
    incompatible = predictor.load_state_dict(
        torch.load(
            checkpoint / "grid_wm.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=False,
    )
    if incompatible.missing_keys != ["temporal_position"] or incompatible.unexpected_keys:
        raise ValueError(
            "legacy grid predictor is not compatible with H-step extension: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    loaded_value = ValueHead.load_checkpoint(
        checkpoint / "value_head",
        emb_dim=projector.emb_dim,
    )
    _unwrap(world_model.value_head).load_state_dict(loaded_value.state_dict())
    return {
        "source": str(checkpoint.resolve()),
        "source_epoch": int(state["epoch"]),
        "source_step": int(state["step"]),
        "predictor_new_parameters": ["temporal_position"],
        "resume": False,
    }


__all__ = [
    "EMATargetGridEncoder",
    "GridPredictorConfig",
    "GridStateProjector",
    "GridWorldModel",
    "LeWMGridDecoder",
    "LeWMGridEncoder",
    "SharedSlotProjector",
    "TemporalSpatialGridPredictor",
    "load_sft1_slot_projector",
    "warm_start_legacy_grid_components",
]
