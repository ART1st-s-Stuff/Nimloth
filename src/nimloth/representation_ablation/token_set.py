"""Token-set world-model modules for representation ablations.

These modules are the Phase-2 building blocks for multi-latent and compressed
vision-token experiments.  They do not extract Qwen features by themselves; they
operate on already encoded token sets with shape ``(B, K, D)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

# Keep this module importable without external/le-wm.  Importing
# nimloth.wm.dataset would execute nimloth.wm.__init__, which imports LeWM.
NUM_NAVIGATION_ACTIONS = 8


@dataclass
class TokenSetPredictorConfig:
    """Config for action-conditioned token-set dynamics."""

    num_tokens: int = 4
    emb_dim: int = 1024
    action_dim: int = NUM_NAVIGATION_ACTIONS
    hidden_dim: int = 1024
    depth: int = 4
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.emb_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("emb_dim and hidden_dim must be positive")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")


class TokenSetWMPredictor(nn.Module):
    """Predict next state token set from current token set and action.

    Input/output shape is ``(B, K, D)``.  The action is represented as one extra
    token prepended to the state tokens; the transformer output state positions
    are projected back to the representation dimension.
    """

    def __init__(self, config: TokenSetPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or TokenSetPredictorConfig()
        cfg = self.config
        self.input_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim)
        self.action_embed = nn.Embedding(cfg.action_dim, cfg.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.num_tokens + 1, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.hidden_dim * cfg.mlp_ratio,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.depth)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)

    @property
    def emb_dim(self) -> int:
        return self.config.emb_dim

    @property
    def num_tokens(self) -> int:
        return self.config.num_tokens

    def predict_next_emb(self, state_tokens: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        """Single-step token-set prediction.

        Args:
            state_tokens: ``(B, K, D)`` current token-set state.
            action_indices: ``(B,)`` action ids.
        Returns:
            ``(B, K, D)`` predicted next token-set state.
        """

        if state_tokens.ndim != 3:
            raise ValueError(f"state_tokens must have shape (B, K, D), got {tuple(state_tokens.shape)}")
        if state_tokens.shape[1] != self.config.num_tokens:
            raise ValueError(f"expected K={self.config.num_tokens}, got {state_tokens.shape[1]}")
        if action_indices.ndim != 1 or action_indices.shape[0] != state_tokens.shape[0]:
            raise ValueError("action_indices must have shape (B,) matching state_tokens")
        action = self.action_embed(action_indices.long()).unsqueeze(1)
        state = self.input_proj(state_tokens)
        x = torch.cat([action, state], dim=1) + self.pos_embed.to(dtype=state.dtype, device=state.device)
        x = self.encoder(x)
        state_out = self.norm(x[:, 1:, :])
        return self.output_proj(state_out)

    def rollout_states(self, state_tokens: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        """Autoregressive rollout in token-set space.

        Args:
            state_tokens: ``(B, K, D)`` initial state.
            action_sequences: ``(B, T)`` action ids.
        Returns:
            ``(B, T, K, D)`` predicted states after every action.
        """

        if action_sequences.ndim != 2:
            raise ValueError(f"action_sequences must have shape (B, T), got {tuple(action_sequences.shape)}")
        cur = state_tokens
        preds: list[torch.Tensor] = []
        for step in range(action_sequences.shape[1]):
            cur = self.predict_next_emb(cur, action_sequences[:, step])
            preds.append(cur)
        if not preds:
            return state_tokens.new_empty(state_tokens.shape[0], 0, *state_tokens.shape[1:])
        return torch.stack(preds, dim=1)

    def forward(self, state_tokens: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        return self.predict_next_emb(state_tokens, action_indices)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "predictor.pt")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "TokenSetWMPredictor":
        path = Path(path)
        cfg = TokenSetPredictorConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        module = cls(cfg)
        state_path = path / "predictor.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing TokenSetWMPredictor checkpoint: {state_path}")
        module.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
        return module


class TokenSetValueHead(nn.Module):
    """Pool token-set states and predict per-action values."""

    def __init__(
        self,
        *,
        emb_dim: int,
        num_tokens: int,
        num_actions: int = NUM_NAVIGATION_ACTIONS,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        hidden = hidden_dim or emb_dim
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_actions = num_actions
        self.token_score = nn.Linear(emb_dim, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_actions),
        )

    def pool(self, state_tokens: torch.Tensor) -> torch.Tensor:
        if state_tokens.ndim != 3:
            raise ValueError(f"state_tokens must have shape (B, K, D), got {tuple(state_tokens.shape)}")
        if state_tokens.shape[1] != self.num_tokens or state_tokens.shape[2] != self.emb_dim:
            raise ValueError(
                f"expected state_tokens shape (B, {self.num_tokens}, {self.emb_dim}), got {tuple(state_tokens.shape)}"
            )
        weights = torch.softmax(self.token_score(state_tokens.float()).squeeze(-1), dim=-1)
        return torch.sum(state_tokens * weights.to(dtype=state_tokens.dtype).unsqueeze(-1), dim=1)

    def forward(self, state_tokens: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(state_tokens)
        weight = self.net[1].weight
        return self.net(pooled.to(dtype=weight.dtype))

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "value_head.pt")
        config = {
            "emb_dim": self.emb_dim,
            "num_tokens": self.num_tokens,
            "num_actions": self.num_actions,
            "hidden_dim": self.net[1].out_features,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "TokenSetValueHead":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        module = cls(**cfg)
        state_path = path / "value_head.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing TokenSetValueHead checkpoint: {state_path}")
        module.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
        return module
