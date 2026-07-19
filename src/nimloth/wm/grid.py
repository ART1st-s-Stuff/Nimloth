"""Spatial 4x4 query-state projection and joint latent world model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class SharedSlotProjector(nn.Module):
    """Apply one shared MLP to every spatial query slot.

    The projector preserves the grid axis: ``(B, N, H_qwen)`` becomes
    ``(B, N, H_state)``. Sharing parameters prevents sixteen independent heads
    from absorbing region-specific teacher errors while query identity and
    spatial position remain represented by the Qwen hidden states.
    """

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
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"SharedSlotProjector expected (B, N, {self.input_dim}), got {tuple(hidden.shape)}"
            )
        dtype = next(self.parameters()).dtype
        return self.net(hidden.to(dtype=dtype))


class GridLatentWMPredictor(nn.Module):
    """Jointly predict the complete next spatial token grid from state+action.

    A single transformer mixes all spatial slots, so camera motion and object
    movement can transfer information between regions. One action embedding is
    broadcast to every slot; learned spatial positions keep row-major slot
    identities stable.
    """

    def __init__(
        self,
        *,
        grid_tokens: int = 16,
        emb_dim: int = 1024,
        action_dim: int = 6,
        depth: int = 4,
        heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if grid_tokens < 1:
            raise ValueError("grid_tokens must be positive")
        if emb_dim % heads != 0:
            raise ValueError("emb_dim must be divisible by heads")
        self.grid_tokens = int(grid_tokens)
        self.emb_dim = int(emb_dim)
        self.action_dim = int(action_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.spatial_position = nn.Parameter(torch.zeros(1, self.grid_tokens, self.emb_dim))
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        self.action_embedding = nn.Embedding(self.action_dim, self.emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=int(heads),
            dim_feedforward=int(mlp_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=int(depth), enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(self.emb_dim)
        self.output_projection = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, state_grid: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_tokens, self.emb_dim)
        if state_grid.ndim != 3 or tuple(state_grid.shape[1:]) != expected:
            raise ValueError(
                "GridLatentWMPredictor expected state shape "
                f"(B, {self.grid_tokens}, {self.emb_dim}), got {tuple(state_grid.shape)}"
            )
        if action_indices.ndim != 1 or action_indices.shape[0] != state_grid.shape[0]:
            raise ValueError(
                f"action_indices must have shape ({state_grid.shape[0]},), got {tuple(action_indices.shape)}"
            )
        if torch.any(action_indices < 0) or torch.any(action_indices >= self.action_dim):
            raise ValueError("action index is outside configured action_dim")
        dtype = next(self.parameters()).dtype
        state = state_grid.to(dtype=dtype)
        action = self.action_embedding(action_indices).unsqueeze(1)
        mixed = self.mixer(state + self.spatial_position + action)
        return self.output_projection(self.output_norm(mixed))

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "predictor.pt")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "grid_tokens": self.grid_tokens,
                    "emb_dim": self.emb_dim,
                    "action_dim": self.action_dim,
                    "depth": self.depth,
                    "heads": self.heads,
                    "mlp_dim": self.mlp_dim,
                    "dropout": self.dropout,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "GridLatentWMPredictor":
        path = Path(path)
        config_path = path / "config.json"
        state_path = path / "predictor.pt"
        if not config_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"incomplete grid WM checkpoint: {path}")
        module = cls(**json.loads(config_path.read_text(encoding="utf-8")))
        module.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
        return module


def load_sft1_slot_projector(
    checkpoint: str | Path,
    *,
    qwen_hidden_dim: int,
    state_dim: int = 1024,
    grid_tokens: int = 16,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> SharedSlotProjector:
    """Load the fail-closed SFT1→SFT2 spatial representation interface."""

    checkpoint = Path(checkpoint)
    config_path = checkpoint / "grid_state_config.json"
    state_path = checkpoint / "slot_projector.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"SFT1 grid-state interface is incomplete: {checkpoint}")
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
    projector.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
    if dtype is not None:
        projector.to(dtype=dtype)
    return projector
