"""Thin wrapper: LeWM ARPredictor on Qwen-latent state (no pixel encoder)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from nimloth.wm._vendor_lewm import ARPredictor, Embedder, MLP
from nimloth.wm.lewm import LeWMConfig


class LatentWMPredictor(nn.Module):
    """LeWM ARPredictor + action encoder for Qwen-latent dynamics.

    Mirrors the LeWM paper predictor structure:
    - ARPredictor outputs into ``predictor_hidden_dim`` (not directly ``emb_dim``).
    - ``pred_proj`` (LeWM MLP with BatchNorm1d) maps predictor hidden to ``emb_dim``.
    """

    def __init__(self, config: LeWMConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.action_dim,
            emb_dim=config.emb_dim,
        )
        self.predictor = ARPredictor(
            num_frames=config.history_size,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            mlp_dim=config.predictor_mlp_dim,
            input_dim=config.emb_dim,
            hidden_dim=config.predictor_hidden_dim,
            output_dim=config.predictor_hidden_dim,  # LeWM style: not directly emb_dim
        )
        # LeWM-style projection: predictor_hidden_dim -> hidden -> emb_dim, with BatchNorm
        self.pred_proj = MLP(
            config.predictor_hidden_dim,
            config.predictor_hidden_dim,
            config.emb_dim,
            norm_fn=nn.LayerNorm,  # default in LeWM MLP; avoids BN inplace conflicts
        )

    @property
    def emb_dim(self) -> int:
        return self.config.emb_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_from_context(
        self, state_ctx: torch.Tensor, action_ctx: torch.Tensor
    ) -> torch.Tensor:
        """Predict next state from a (possibly multi-step) context window.

        Args:
            state_ctx:  (B, T, emb_dim)  -- T states (most recent at position T-1).
            action_ctx: (B, T) int64     -- T action indices (paired element-wise with states).

        Returns:
            (B, emb_dim) -- predicted next state after the last context step.
        """
        return self.predict_sequence(state_ctx, action_ctx)[:, -1]

    def predict_sequence(
        self,
        state_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """预测上下文中每个 ``state/action`` 位置对应的下一状态。

        ``history_size`` 是最大上下文长度。训练传入完整的 ``H`` 个位置；episode
        开头和自回归 rollout 可以传入尚未填满的真实短前缀，禁止伪造重复状态或
        zero action 来凑满窗口。
        """

        if state_ctx.ndim != 3:
            raise ValueError(
                "state_ctx must have shape (B, T, D), "
                f"got {tuple(state_ctx.shape)}"
            )
        if action_ctx.ndim != 2 or action_ctx.shape != state_ctx.shape[:2]:
            raise ValueError(
                "action_ctx must have shape (B, T) matching state_ctx, "
                f"got {tuple(action_ctx.shape)} for {tuple(state_ctx.shape)}"
            )
        time_steps = state_ctx.shape[1]
        if not 1 <= time_steps <= self.config.history_size:
            raise ValueError(
                "context length must be in [1, history_size], "
                f"got T={time_steps}, history_size={self.config.history_size}"
            )

        # action_one_hot 会为 (B,) 输入额外插入时间轴；sequence 路径直接构造
        # (B, T, action_dim)，确保 LeWM 的因果时间轴保持不变。
        actions = torch.nn.functional.one_hot(
            action_ctx, num_classes=self.config.action_dim
        ).float()
        act_emb = self.action_encoder(actions)
        preds = self.predictor(state_ctx, act_emb)
        b, t, _ = preds.shape
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=b, t=t)
        return preds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_next_emb(
        self, state_emb: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        """Single-step next-latent prediction.

        Args:
            state_emb:      (B, emb_dim)   -- current WM state.
            action_indices: (B,) int64     -- action taken at current step.

        Returns:
            (B, emb_dim) -- predicted next WM state.
        """
        return self._predict_from_context(
            state_emb.unsqueeze(1),   # (B, 1, emb_dim)
            action_indices.unsqueeze(1),  # (B, 1)
        )

    def rollout_states(
        self,
        state_emb: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive multi-step rollout (purely in WM latent space, no Qwen).

        Args:
            state_emb:        (B, emb_dim)       -- initial WM state (from Qwen slow path).
            action_sequences: (B, num_steps)     -- action indices for each step.

        Returns:
            (B, num_steps, emb_dim) -- predicted states s₁ … s_num_steps.
        """
        if state_emb.ndim != 2:
            raise ValueError(
                "state_emb must have shape (B,D), "
                f"got {tuple(state_emb.shape)}"
            )
        empty_history = action_sequences.new_empty(
            (action_sequences.shape[0], 0)
        )
        return self.rollout_from_history(
            state_emb.unsqueeze(1),
            empty_history,
            action_sequences,
        )

    def rollout_from_history(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """从真实历史上下文出发，自回归模拟未来 action sequence。

        ``state_history`` 与 ``previous_actions`` 分别为 ``(B,L,D)`` 和
        ``(B,L-1)``；动作 ``previous_actions[:, i]`` 与状态
        ``state_history[:, i]`` 配对。``future_actions`` 的第一个动作与
        最后一个真实状态配对。
        """

        if state_history.ndim != 3:
            raise ValueError(
                "state_history must have shape (B,L,D), "
                f"got {tuple(state_history.shape)}"
            )
        batch_size, history_steps = state_history.shape[:2]
        if not 1 <= history_steps <= self.config.history_size:
            raise ValueError(
                "real state history length must be in [1, history_size], "
                f"got L={history_steps}, history_size={self.config.history_size}"
            )
        expected_previous = (batch_size, history_steps - 1)
        if previous_actions.shape != expected_previous:
            raise ValueError(
                "previous_actions must align with all but the last real state, "
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
            next_state = self._predict_from_context(
                state_context,
                action_context,
            )
            predicted.append(next_state)
            all_states = torch.cat((all_states, next_state.unsqueeze(1)), dim=1)
        return torch.stack(predicted, dim=1)

    def forward(self, state_emb: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        """同时支持单步 ``(B,D)`` 与 sequence ``(B,T,D)`` 的模块入口。"""

        if state_emb.ndim == 2:
            return self.predict_next_emb(state_emb, action_indices)
        if state_emb.ndim == 3:
            return self.predict_sequence(state_emb, action_indices)
        raise ValueError(
            "state_emb must have shape (B, D) or (B, T, D), "
            f"got {tuple(state_emb.shape)}"
        )

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "predictor.pt")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def create(cls, config: LeWMConfig | None = None) -> "LatentWMPredictor":
        return cls(config or LeWMConfig())

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "LatentWMPredictor":
        path = Path(path)
        cfg_dict = json.loads((path / "config.json").read_text(encoding="utf-8"))
        cfg = LeWMConfig(**{k: v for k, v in cfg_dict.items() if k in LeWMConfig.__dataclass_fields__})
        module = cls.create(cfg)
        state_path = path / "predictor.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location=map_location, weights_only=True)
            module.load_state_dict(state)
            return module

        raise FileNotFoundError(f"missing LatentWMPredictor checkpoint: {state_path}")
