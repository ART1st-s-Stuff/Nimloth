"""High-level LeWM wrapper for Nimloth training and inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS
from nimloth.wm.model import ARPredictor, Embedder, JEPA, MLP, NavigationCNNEncoder, SIGReg


@dataclass
class LeWMConfig:
    img_size: int = 96
    emb_dim: int = 128
    action_dim: int = NUM_NAVIGATION_ACTIONS
    action_emb_dim: int = 64
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_mlp_dim: int = 256
    predictor_hidden_dim: int = 128
    history_size: int = 1
    sigreg_weight: float = 0.1


def build_lewm(config: LeWMConfig | None = None) -> JEPA:
    cfg = config or LeWMConfig()
    encoder = NavigationCNNEncoder(img_size=cfg.img_size, emb_dim=cfg.emb_dim)
    predictor = ARPredictor(
        num_frames=cfg.history_size,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        input_dim=cfg.emb_dim,
        hidden_dim=cfg.predictor_hidden_dim,
        cond_dim=cfg.action_emb_dim,
        output_dim=cfg.emb_dim,
    )
    action_encoder = Embedder(
        input_dim=cfg.action_dim,
        smoothed_dim=cfg.action_dim,
        emb_dim=cfg.action_emb_dim,
    )
    projector = MLP(cfg.emb_dim, cfg.emb_dim, cfg.emb_dim)
    pred_proj = MLP(cfg.emb_dim, cfg.emb_dim, cfg.emb_dim)
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        emb_dim=cfg.emb_dim,
    )


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def action_one_hot(indices: torch.Tensor, num_actions: int = NUM_NAVIGATION_ACTIONS) -> torch.Tensor:
    """indices: (B,) int64 -> (B, 1, num_actions) float."""

    one_hot = torch.nn.functional.one_hot(indices.long(), num_classes=num_actions).float()
    return one_hot.unsqueeze(1)


class LeWMWrapper(nn.Module):
    """Frozen/trainable facade around JEPA for SFT2 alignment."""

    def __init__(self, model: JEPA, config: LeWMConfig, sigreg: SIGReg | None = None) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.sigreg = sigreg or SIGReg()

    @classmethod
    def create(cls, config: LeWMConfig | None = None) -> "LeWMWrapper":
        cfg = config or LeWMConfig()
        return cls(build_lewm(cfg), cfg, SIGReg())

    @property
    def emb_dim(self) -> int:
        return self.config.emb_dim

    def freeze(self) -> None:
        freeze_module(self.model)

    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, C, H, W) -> (B, emb_dim)."""

        return self.model.encode_pixels(pixels)

    def encode_next_targets(self, next_pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encode_image(next_pixels)

    def predict_next_emb(self, state_emb: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        """state_emb: (B, D); action_indices: (B,) -> pred (B, D)."""

        ctx = state_emb.unsqueeze(1)
        actions = action_one_hot(action_indices, self.config.action_dim)
        act_emb = self.model.action_encoder(actions)
        pred = self.model.predict(ctx, act_emb)[:, -1]
        return pred

    def alignment_loss(
        self,
        state_emb: torch.Tensor,
        action_indices: torch.Tensor,
        next_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = self.predict_next_emb(state_emb, action_indices)
        tgt = self.encode_next_targets(next_pixels)
        mse = torch.nn.functional.mse_loss(pred, tgt)
        return mse, {"wm_mse": float(mse.detach().item())}

    def pretrain_step(self, current_pixels: torch.Tensor, next_pixels: torch.Tensor, action_indices: torch.Tensor):
        pixels = torch.stack([current_pixels, next_pixels], dim=1)
        actions = action_one_hot(action_indices, self.config.action_dim)
        loss, metrics = self.model.pretrain_loss(
            pixels,
            actions,
            history_size=self.config.history_size,
            sigreg=self.sigreg,
            sigreg_weight=self.config.sigreg_weight,
        )
        out = {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
        return loss, out

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "LeWMWrapper":
        path = Path(path)
        cfg_dict = json.loads((path / "config.json").read_text(encoding="utf-8"))
        cfg = LeWMConfig(**cfg_dict)
        wrapper = cls.create(cfg)
        state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
        wrapper.model.load_state_dict(state)
        return wrapper
