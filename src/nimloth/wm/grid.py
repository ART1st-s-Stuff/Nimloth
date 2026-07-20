"""Spatial 4x4 query-state projection and joint latent world model."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.wm._vendor_lewm import ConditionalBlock, Embedder, MLP, modulate
from nimloth.wm.lewm import SafeBatchNorm1d


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


class _LeWMGridMLP(nn.Module):
    """Apply LeWM's projector/pred-proj MLP independently to spatial slots."""

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
            raise ValueError(f"expected grid (B, N, {self.emb_dim}), got {tuple(grid.shape)}")
        batch, slots, _ = grid.shape
        weight = next(self.parameters())
        encoded = self.net(grid.reshape(batch * slots, self.emb_dim).to(dtype=weight.dtype))
        return encoded.reshape(batch, slots, self.emb_dim)


class LeWMGridEncoder(_LeWMGridMLP):
    """LeWM projector MLP adapted to each SFT1 spatial query slot."""


class LeWMGridDecoder(_LeWMGridMLP):
    """LeWM pred-proj MLP used to decode predicted latents to DINO tokens."""


class EMATargetGridEncoder(nn.Module):
    """Non-trainable EMA copy of the online LeWM grid encoder."""

    def __init__(self, online_encoder: LeWMGridEncoder, *, decay: float = 0.99) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.encoder = copy.deepcopy(online_encoder)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def update(self, online_encoder: LeWMGridEncoder) -> None:
        online = online_encoder.module if hasattr(online_encoder, "module") else online_encoder
        for target_parameter, online_parameter in zip(
            self.encoder.parameters(), online.parameters(), strict=True
        ):
            target_parameter.mul_(self.decay).add_(online_parameter.detach(), alpha=1.0 - self.decay)
        for target_buffer, online_buffer in zip(
            self.encoder.buffers(), online.buffers(), strict=True
        ):
            if target_buffer.is_floating_point():
                target_buffer.mul_(self.decay).add_(online_buffer.detach(), alpha=1.0 - self.decay)
            else:
                target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.encoder(grid).detach()


class _NonCausalConditionalBlock(ConditionalBlock):
    """LeWM AdaLN-zero block with bidirectional spatial attention."""

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            causal=False,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LeWMSpatialPredictor(nn.Module):
    """LeWM action-conditioned predictor adapted from temporal to spatial tokens."""

    def __init__(
        self,
        *,
        grid_tokens: int = 16,
        emb_dim: int = 1024,
        action_dim: int = 6,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if grid_tokens < 1:
            raise ValueError("grid_tokens must be positive")
        self.grid_tokens = int(grid_tokens)
        self.emb_dim = int(emb_dim)
        self.action_dim = int(action_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.spatial_position = nn.Parameter(torch.randn(1, self.grid_tokens, self.emb_dim))
        self.action_encoder = Embedder(
            input_dim=self.action_dim,
            smoothed_dim=self.action_dim,
            emb_dim=self.emb_dim,
        )
        self.layers = nn.ModuleList(
            [
                _NonCausalConditionalBlock(
                    self.emb_dim,
                    heads=self.heads,
                    dim_head=self.dim_head,
                    mlp_dim=self.mlp_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.emb_dim)

    def forward(self, state_grid: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        if state_grid.ndim != 3 or tuple(state_grid.shape[1:]) != (
            self.grid_tokens,
            self.emb_dim,
        ):
            raise ValueError(
                f"expected state grid (B, {self.grid_tokens}, {self.emb_dim}), "
                f"got {tuple(state_grid.shape)}"
            )
        if action_indices.ndim != 1 or action_indices.shape[0] != state_grid.shape[0]:
            raise ValueError(f"action_indices must have shape ({state_grid.shape[0]},)")
        if torch.any(action_indices < 0) or torch.any(action_indices >= self.action_dim):
            raise ValueError("action index is outside configured action_dim")
        dtype = self.spatial_position.dtype
        actions = F.one_hot(action_indices.long(), num_classes=self.action_dim).float()
        actions = actions.unsqueeze(1).expand(-1, self.grid_tokens, -1)
        condition = self.action_encoder(actions).to(dtype=dtype)
        state = state_grid.to(dtype=dtype) + self.spatial_position
        for block in self.layers:
            state = block(state, condition)
        return self.norm(state)


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
