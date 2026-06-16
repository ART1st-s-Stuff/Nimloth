"""LeWM-compatible JEPA modules (adapted from https://github.com/lucas-maes/le-wm)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (T, B, D)
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: type[nn.Module] = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Embedder(nn.Module):
    def __init__(self, input_dim: int = 8, smoothed_dim: int = 8, emb_dim: int = 64, mlp_scale: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ARPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        cond_dim: int | None = None,
        output_dim: int | None = None,
        dim_head: int = 32,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        out_dim = output_dim or input_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for layer in self.layers:
            x = layer(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class NavigationCNNEncoder(nn.Module):
    """Lightweight pixel encoder for EB-Nav WM pretraining."""

    def __init__(self, img_size: int = 96, emb_dim: int = 128) -> None:
        super().__init__()
        self.img_size = img_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, emb_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, C, H, W)
        feat = self.backbone(pixels).flatten(1)
        return self.proj(feat)


class JEPA(nn.Module):
    """Joint-embedding predictive architecture."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: ARPredictor,
        action_encoder: Embedder,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.emb_dim = emb_dim

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to embeddings. pixels: (B, C, H, W)."""

        b = pixels.size(0)
        flat = pixels.reshape(b, *pixels.shape[-3:])
        emb = self.encoder(flat)
        return self.projector(emb)

    def encode_batch(self, pixels: torch.Tensor, actions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Encode pixels shaped (B, T, C, H, W) and optional actions (B, T, A)."""

        b, t = pixels.shape[:2]
        flat = rearrange(pixels, "b t ... -> (b t) ...")
        emb = self.encode_pixels(flat)
        emb = rearrange(emb, "(b t) d -> b t d", b=b, t=t)
        out: dict[str, torch.Tensor] = {"emb": emb}
        if actions is not None:
            out["act_emb"] = self.action_encoder(actions)
        return out

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        b, t, _ = preds.shape
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=b, t=t)

    def pretrain_loss(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        *,
        history_size: int = 1,
        sigreg: SIGReg | None = None,
        sigreg_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Standard LeWM-style single-step prediction loss on transition batches.

        pixels: (B, 2, C, H, W) with [o_t, o_{t+1}]
        actions: (B, 1, A) one-hot action at t
        """

        encoded = self.encode_batch(pixels, actions)
        emb = encoded["emb"]
        act_emb = encoded["act_emb"]
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        tgt_emb = emb[:, history_size:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        metrics = {"pred_loss": pred_loss.detach()}
        if sigreg is not None:
            sigreg_loss = sigreg(emb.transpose(0, 1))
            metrics["sigreg_loss"] = sigreg_loss.detach()
            total = pred_loss + sigreg_weight * sigreg_loss
        else:
            total = pred_loss
        metrics["loss"] = total.detach()
        return total, metrics
