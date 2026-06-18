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
    """LeWM ARPredictor + action encoder for Qwen-latent dynamics."""

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
            output_dim=config.emb_dim,
        )
        self.pred_proj = MLP(config.emb_dim, config.emb_dim, config.emb_dim)

    @property
    def emb_dim(self) -> int:
        return self.config.emb_dim

    def predict_next_emb(self, state_emb: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        ctx = state_emb.unsqueeze(1)
        actions = action_one_hot(action_indices, self.config.action_dim)
        act_emb = self.action_encoder(actions)
        preds = self.predictor(ctx, act_emb)
        b, t, _ = preds.shape
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=b, t=t)
        return preds[:, -1]

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
