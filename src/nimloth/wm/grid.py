"""16-slot spatial state and H-context temporal-spatial world model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nimloth.wm._vendor_lewm import Embedder, modulate
from nimloth.wm.model import WorldModel


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
    """加载 SFT1 的 row-major 16-slot projector，供 SFT2 继续训练。"""

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
    if dtype is not None:
        projector.to(dtype=dtype)
    return projector


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

    def rollout_from_history(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """从真实 grid state 历史自回归模拟未来动作序列。"""

        expected_tail = (self.grid_tokens, self.emb_dim)
        if (
            state_history.ndim != 4
            or tuple(state_history.shape[2:]) != expected_tail
        ):
            raise ValueError(
                "grid state_history must have shape "
                f"(B,L,{self.grid_tokens},{self.emb_dim}), "
                f"got {tuple(state_history.shape)}"
            )
        batch_size, history_steps = state_history.shape[:2]
        if not 1 <= history_steps <= self.config.history_size:
            raise ValueError(
                "real grid history length must be in [1, history_size], "
                f"got L={history_steps}, history_size={self.config.history_size}"
            )
        expected_previous = (batch_size, history_steps - 1)
        if previous_actions.shape != expected_previous:
            raise ValueError(
                "previous_actions must align with all but the last grid state, "
                f"got {tuple(previous_actions.shape)}, expected {expected_previous}"
            )
        if (
            future_actions.ndim != 2
            or future_actions.shape[0] != batch_size
            or future_actions.shape[1] < 1
        ):
            raise ValueError(
                "future_actions must have shape (B,P) with P>=1, "
                f"got {tuple(future_actions.shape)}"
            )

        all_states = state_history
        all_actions = torch.cat((previous_actions, future_actions), dim=1)
        predicted: list[torch.Tensor] = []
        for future_step in range(future_actions.shape[1]):
            state_index = history_steps - 1 + future_step
            context_start = max(
                0,
                state_index - self.config.history_size + 1,
            )
            state_context = all_states[:, context_start : state_index + 1]
            action_context = all_actions[:, context_start : state_index + 1]
            next_state = self.predict_sequence(
                state_context,
                action_context,
            )[:, -1]
            predicted.append(next_state)
            all_states = torch.cat((all_states, next_state.unsqueeze(1)), dim=1)
        return torch.stack(predicted, dim=1)

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


class ResidualTemporalSpatialGridPredictor(nn.Module):
    """以严格copy初始化的temporal-spatial grid predictor。

    body保留原有action-conditioned时空建模能力；独立delta head零初始化，
    因而第一次更新前的预测逐值等于输入state。这样T1训练必须从copy baseline
    出发，不能依靠随机初始化制造看似有限但更差的successor state。
    """

    def __init__(self, config: GridPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.body = TemporalSpatialGridPredictor(config)
        self.delta_head = nn.Linear(config.emb_dim, config.emb_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @property
    def grid_tokens(self) -> int:
        return int(self.config.grid_tokens)

    @property
    def emb_dim(self) -> int:
        return int(self.config.emb_dim)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    def is_zero_initialized(self) -> bool:
        return bool(
            torch.count_nonzero(self.delta_head.weight.detach()).item() == 0
            and torch.count_nonzero(self.delta_head.bias.detach()).item() == 0
        )

    def predict_sequence(
        self,
        state_context: torch.Tensor,
        action_context: torch.Tensor,
    ) -> torch.Tensor:
        features = self.body.predict_sequence(state_context, action_context)
        baseline = state_context.to(dtype=self.delta_head.weight.dtype)
        return baseline + self.delta_head(features)

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

    def rollout_from_history(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        expected_tail = (self.grid_tokens, self.emb_dim)
        if state_history.ndim != 4 or tuple(state_history.shape[2:]) != expected_tail:
            raise ValueError(
                "residual grid state_history must have shape "
                f"(B,L,{self.grid_tokens},{self.emb_dim}), got {tuple(state_history.shape)}"
            )
        batch_size, history_steps = state_history.shape[:2]
        if not 1 <= history_steps <= self.config.history_size:
            raise ValueError("residual grid history length is outside configured context")
        if previous_actions.shape != (batch_size, history_steps - 1):
            raise ValueError("previous_actions do not align with residual grid history")
        if (
            future_actions.ndim != 2
            or future_actions.shape[0] != batch_size
            or future_actions.shape[1] < 1
        ):
            raise ValueError("future_actions must have shape (B,P) with P>=1")

        all_states = state_history
        all_actions = torch.cat((previous_actions, future_actions), dim=1)
        predicted: list[torch.Tensor] = []
        for future_step in range(future_actions.shape[1]):
            state_index = history_steps - 1 + future_step
            context_start = max(0, state_index - self.config.history_size + 1)
            next_state = self.predict_sequence(
                all_states[:, context_start : state_index + 1],
                all_actions[:, context_start : state_index + 1],
            )[:, -1]
            predicted.append(next_state)
            all_states = torch.cat((all_states, next_state.unsqueeze(1)), dim=1)
        return torch.stack(predicted, dim=1)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "predictor.pt")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "schema": "nimloth_residual_temporal_spatial_grid_v1",
                    "predictor": asdict(self.config),
                    "delta_head_zero_initialized": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "ResidualTemporalSpatialGridPredictor":
        path = Path(path)
        payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if payload.get("schema") != "nimloth_residual_temporal_spatial_grid_v1":
            raise ValueError("unsupported residual grid predictor checkpoint schema")
        module = cls(GridPredictorConfig(**payload["predictor"]))
        module.load_state_dict(
            torch.load(path / "predictor.pt", map_location=map_location, weights_only=True)
        )
        return module


class GridWorldModel(WorldModel):
    """16-slot WM；state 就是可训练的 SFT1 projector 输出。"""

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

    def predict_action_values(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim < 3:
            raise ValueError(
                "grid value input must have shape (...,N,D), "
                f"got {tuple(state.shape)}"
            )
        return self.value_head(state.mean(dim=-2)).float()

    def predict_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim < 3:
            raise ValueError(
                "grid policy input must have shape (...,N,D), "
                f"got {tuple(state.shape)}"
            )
        if self.planner_policy_head is None:
            raise RuntimeError("grid world model has no PlannerPolicyHead")
        return self.planner_policy_head(state.mean(dim=-2)).float()

    def unwrapped(self) -> "GridWorldModel":
        return GridWorldModel(
            state_proj=_unwrap(self.state_proj),
            wm_predictor=_unwrap(self.wm_predictor),
            value_head=_unwrap(self.value_head),
            planner_policy_head=(
                _unwrap(self.planner_policy_head)
                if self.planner_policy_head is not None
                else None
            ),
        )


__all__ = [
    "GridPredictorConfig",
    "GridWorldModel",
    "SharedSlotProjector",
    "TemporalSpatialGridPredictor",
    "ResidualTemporalSpatialGridPredictor",
    "load_sft1_slot_projector",
]
