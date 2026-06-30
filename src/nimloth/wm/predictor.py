"""Thin wrapper: LeWM ARPredictor on Qwen-latent state (no pixel encoder)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from nimloth.wm._vendor_lewm import ARPredictor, Embedder, MLP
from nimloth.wm.lewm import LeWMConfig, SafeBatchNorm1d, action_one_hot

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
            norm_fn=SafeBatchNorm1d,
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
        # action_one_hot adds an extra unsqueeze(1) designed for (B,) input;
        # for multi-step (B, T) we use one_hot directly to get (B, T, num_actions).
        actions = torch.nn.functional.one_hot(
            action_ctx, num_classes=self.config.action_dim
        ).float()
        act_emb = self.action_encoder(actions)
        preds = self.predictor(state_ctx, act_emb)
        b, t, _ = preds.shape
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=b, t=t)
        return preds[:, -1]

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
        B = state_emb.shape[0]
        num_steps = action_sequences.shape[1]
        H = self.config.history_size
        device = state_emb.device

        # all_s grows from (B, 1, emb_dim) to (B, num_steps + 1, emb_dim).
        all_s = state_emb.unsqueeze(1)
        zero_action = torch.zeros(B, dtype=torch.long, device=device)

        for t in range(num_steps):
            s_ctx_list: list[torch.Tensor] = []
            a_ctx_list: list[torch.Tensor] = []

            for h in range(H):
                s_idx = t - H + 1 + h  # state index into s₀…s_t
                if s_idx < 0:
                    s_ctx_list.append(state_emb)
                    a_ctx_list.append(zero_action)
                elif s_idx == 0:
                    s_ctx_list.append(state_emb)
                    a_ctx_list.append(action_sequences[:, s_idx])
                else:
                    s_ctx_list.append(all_s[:, s_idx, :])
                    a_ctx_list.append(action_sequences[:, s_idx])

            s_ctx = torch.stack(s_ctx_list, dim=1)  # (B, H, emb_dim)
            a_ctx = torch.stack(a_ctx_list, dim=1)  # (B, H)

            next_s = self._predict_from_context(s_ctx, a_ctx)  # (B, emb_dim)
            all_s = torch.cat([all_s, next_s.unsqueeze(1)], dim=1)

        return all_s[:, 1:, :]  # drop s₀

    def forward(self, state_emb: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        """DDP-compatible entrypoint for next-latent prediction."""

        return self.predict_next_emb(state_emb, action_indices)

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
