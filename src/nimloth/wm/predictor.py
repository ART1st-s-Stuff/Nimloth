"""Thin wrapper: LeWM ARPredictor on Qwen-latent state (no pixel encoder)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from nimloth.wm._vendor_lewm import ARPredictor, Embedder, MLP
from nimloth.wm.lewm import LeWMConfig, action_one_hot

class LatentWMPredictor(nn.Module):
    """LeWM ARPredictor + action encoder for Qwen-latent dynamics.

    Mirrors the LeWM paper predictor structure:
    - an optional ``dynamics_dim`` factorizes a wide external State before the
      action-conditioned transformer;
    - ARPredictor outputs into ``predictor_hidden_dim`` (not directly ``emb_dim``);
    - ``pred_proj`` maps predictor hidden back to the external ``emb_dim``.
    """

    def __init__(self, config: LeWMConfig) -> None:
        super().__init__()
        self.config = config
        dynamics_dim = int(config.dynamics_dim or config.emb_dim)
        self.dynamics_dim = dynamics_dim
        self.state_input = (
            nn.Identity()
            if dynamics_dim == config.emb_dim
            else nn.Sequential(
                nn.LayerNorm(config.emb_dim),
                nn.Linear(config.emb_dim, dynamics_dim),
            )
        )
        self.action_encoder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.action_dim,
            emb_dim=dynamics_dim,
        )
        self.predictor = ARPredictor(
            num_frames=config.history_size,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            mlp_dim=config.predictor_mlp_dim,
            input_dim=dynamics_dim,
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
        if state_ctx.ndim != 3 or action_ctx.ndim != 2:
            raise ValueError(
                "state/action context must have shapes (B,T,D)/(B,T), got "
                f"{tuple(state_ctx.shape)}/{tuple(action_ctx.shape)}"
            )
        if state_ctx.shape[:2] != action_ctx.shape:
            raise ValueError(
                "state/action context dimensions disagree: "
                f"{tuple(state_ctx.shape)}/{tuple(action_ctx.shape)}"
            )
        expected_t = int(self.config.history_size)
        actual_t = int(state_ctx.shape[1])
        if actual_t != expected_t:
            raise ValueError(
                f"LatentWMPredictor expected exactly T={expected_t} from configured "
                f"history_size={expected_t}, got T={actual_t}; training and inference "
                "history protocols must match"
            )
        # action_one_hot adds an extra unsqueeze(1) designed for (B,) input;
        # for multi-step (B, T) we use one_hot directly to get (B, T, num_actions).
        actions = torch.nn.functional.one_hot(
            action_ctx, num_classes=self.config.action_dim
        ).float()
        act_emb = self.action_encoder(actions)
        if isinstance(self.state_input, nn.Identity):
            dynamics_ctx = state_ctx
        else:
            # StateProjector follows the Qwen BF16 dtype while WM parameters are
            # FP32 by default. Normal training runs under autocast, but DDP's
            # terminal-only dummy path intentionally executes outside it.
            input_dtype = next(self.state_input.parameters()).dtype
            dynamics_ctx = self.state_input(state_ctx.to(dtype=input_dtype))
        preds = self.predictor(dynamics_ctx, act_emb)
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
        if self.config.history_size != 1:
            raise ValueError(
                "predict_next_emb supplies T=1 but this predictor is configured "
                f"history_size={self.config.history_size}; use a predictor trained with "
                "history_size=1 or provide an explicit full history context"
            )
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
        if self.config.history_size != 1:
            raise ValueError(
                "rollout_states(initial_state, actions) only has one initial state and "
                "requires an explicit trained history context when history_size="
                f"{self.config.history_size}; fake repeated-state/action padding is forbidden"
            )
        current = state_emb
        predicted: list[torch.Tensor] = []
        for step in range(action_sequences.shape[1]):
            current = self.predict_next_emb(current, action_sequences[:, step])
            predicted.append(current)
        if not predicted:
            return state_emb.new_empty((state_emb.shape[0], 0, state_emb.shape[-1]))
        return torch.stack(predicted, dim=1)

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
    def load_checkpoint(
        cls,
        path: Path,
        map_location: str | torch.device = "cpu",
        *,
        history_size_override: int | None = None,
    ) -> "LatentWMPredictor":
        path = Path(path)
        cfg_dict = json.loads((path / "config.json").read_text(encoding="utf-8"))
        original_history_size = int(cfg_dict.get("history_size", 1))
        if history_size_override is not None:
            if history_size_override != 1 or original_history_size < 1:
                raise ValueError(
                    "history_size_override only supports explicit migration to the "
                    "historically trained T=1 protocol"
                )
            cfg_dict["history_size"] = 1
        cfg = LeWMConfig(**{k: v for k, v in cfg_dict.items() if k in LeWMConfig.__dataclass_fields__})
        module = cls.create(cfg)
        state_path = path / "predictor.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location=map_location, weights_only=True)
            if history_size_override == 1:
                key = "predictor.pos_embedding"
                if key not in state or state[key].ndim != 3:
                    raise ValueError("checkpoint misses predictor.pos_embedding for T=1 migration")
                state[key] = state[key][:, :1].clone()
            module.load_state_dict(state)
            return module

        raise FileNotFoundError(f"missing LatentWMPredictor checkpoint: {state_path}")
