"""Latent world-model predictors over projected Qwen State."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from nimloth.wm._vendor_lewm import ARPredictor, Embedder, MLP
from nimloth.wm.lewm import LeWMConfig


class _MaskedTokenAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        if heads * dim_head != dim:
            raise ValueError(
                f"tokenized WM requires heads*dim_head==hidden: {heads}*{dim_head}!={dim}"
            )
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.to_qkv(hidden).view(batch, length, 3, self.heads, self.dim_head)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        bias = torch.zeros(
            (batch, 1, length, length), device=hidden.device, dtype=hidden.dtype
        )
        bias.masked_fill_(~allowed[:, None], torch.finfo(hidden.dtype).min)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0
        )
        return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))


class _MaskedConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = _MaskedTokenAttention(dim, heads, dim_head)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(
        hidden: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return hidden * (1 + scale) + shift

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        allowed: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=-1)
        attended = self.attention(
            self._modulate(self.norm_attention(hidden), shift_a, scale_a), allowed
        )
        hidden = hidden + gate_a * attended
        mlp = self.mlp(self._modulate(self.norm_mlp(hidden), shift_m, scale_m))
        hidden = hidden + gate_m * mlp
        return hidden * token_valid.unsqueeze(-1).to(dtype=hidden.dtype)


class _TokenizedHistoryPredictor(nn.Module):
    """T-frame predictor over K State tokens with block-causal padding masks."""

    def __init__(self, config: LeWMConfig) -> None:
        super().__init__()
        if config.emb_dim % config.state_token_count:
            raise ValueError("emb_dim must be divisible by state_token_count")
        self.history_size = int(config.history_size)
        self.token_count = int(config.state_token_count)
        self.action_dim = int(config.action_dim)
        self.token_dim = config.emb_dim // config.state_token_count
        if config.predictor_hidden_dim != self.token_dim:
            raise ValueError(
                "tokenized WM keeps full capacity only when predictor_hidden_dim equals "
                f"emb_dim/state_token_count: {config.predictor_hidden_dim}!={self.token_dim}"
            )
        if not config.residual_prediction:
            raise ValueError("tokenized history WM requires residual_prediction=true")
        if config.dynamics_dim not in (None, config.emb_dim):
            raise ValueError("tokenized history WM does not permit a dynamics_dim bottleneck")
        self.state_norm = nn.LayerNorm(self.token_dim)
        self.time_position = nn.Parameter(
            torch.randn(1, self.history_size, 1, self.token_dim) * 0.02
        )
        self.state_position = nn.Parameter(
            torch.randn(1, 1, self.token_count, self.token_dim) * 0.02
        )
        self.action_encoder = Embedder(
            input_dim=config.action_dim,
            smoothed_dim=config.action_dim,
            emb_dim=self.token_dim,
        )
        self.blocks = nn.ModuleList(
            [
                _MaskedConditionalBlock(
                    self.token_dim,
                    config.predictor_heads,
                    config.predictor_dim_head,
                    config.predictor_mlp_dim,
                )
                for _ in range(config.predictor_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(self.token_dim, config.predictor_mlp_dim),
            nn.GELU(),
            nn.Linear(config.predictor_mlp_dim, self.token_dim),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _allowed_attention(self, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = valid_mask.shape[0]
        length = self.history_size * self.token_count
        token_valid = valid_mask[:, :, None].expand(-1, -1, self.token_count).reshape(batch, length)
        frame = torch.arange(self.history_size, device=valid_mask.device).repeat_interleave(
            self.token_count
        )
        causal = frame[None, :, None] >= frame[None, None, :]
        allowed = causal & token_valid[:, None, :]
        # Invalid queries are discarded after every block, but need one allowed key
        # so scaled-dot-product attention never receives an all-masked row.
        identity = torch.eye(length, device=valid_mask.device, dtype=torch.bool)[None]
        allowed = torch.where(token_valid[:, :, None], allowed, identity)
        return allowed, token_valid

    def forward(
        self,
        state_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, history, external_dim = state_ctx.shape
        if history != self.history_size:
            raise ValueError(f"expected T={self.history_size}, got {history}")
        if external_dim != self.token_count * self.token_dim:
            raise ValueError(f"unexpected external State width: {external_dim}")
        if action_ctx.shape != (batch, history) or valid_mask.shape != (batch, history):
            raise ValueError("action_ctx and valid_mask must match State context B,T")
        if not valid_mask[:, -1].all():
            raise ValueError("the final history slot must always be a real current State/action")
        states = state_ctx.view(batch, history, self.token_count, self.token_dim)
        hidden = self.state_norm(states.float()) + self.time_position + self.state_position
        actions = F.one_hot(action_ctx, num_classes=self.action_dim).float()
        action_hidden = self.action_encoder(actions)
        action_hidden = action_hidden[:, :, None, :].expand(-1, -1, self.token_count, -1)
        action_hidden = action_hidden * valid_mask[:, :, None, None]
        hidden = hidden * valid_mask[:, :, None, None]
        hidden = hidden.reshape(batch, history * self.token_count, self.token_dim)
        action_hidden = action_hidden.reshape_as(hidden)
        allowed, token_valid = self._allowed_attention(valid_mask)
        for block in self.blocks:
            hidden = block(hidden, action_hidden, allowed, token_valid)
        last = hidden.view(batch, history, self.token_count, self.token_dim)[:, -1]
        delta = self.delta_head(self.output_norm(last)).reshape(batch, external_dim)
        return state_ctx[:, -1].float() + delta


class LatentWMPredictor(nn.Module):
    """LeWM predictor with strict history contracts and optional tokenized T>1 residual WM."""

    def __init__(self, config: LeWMConfig) -> None:
        super().__init__()
        self.config = config
        if config.state_token_count > 1:
            self.tokenized_predictor = _TokenizedHistoryPredictor(config)
            self.dynamics_dim = config.emb_dim
            return

        dynamics_dim = int(config.dynamics_dim or config.emb_dim)
        self.dynamics_dim = dynamics_dim
        self.state_input = (
            nn.Identity()
            if dynamics_dim == config.emb_dim
            else nn.Sequential(
                nn.LayerNorm(config.emb_dim), nn.Linear(config.emb_dim, dynamics_dim)
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
            output_dim=config.predictor_hidden_dim,
            dim_head=config.predictor_dim_head,
        )
        self.pred_proj = MLP(
            config.predictor_hidden_dim,
            config.predictor_hidden_dim,
            config.emb_dim,
            norm_fn=nn.LayerNorm,
        )

    @property
    def emb_dim(self) -> int:
        return self.config.emb_dim

    def _validate_context(
        self,
        state_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if state_ctx.ndim != 3 or action_ctx.ndim != 2:
            raise ValueError("state/action context must have shapes (B,T,D)/(B,T)")
        if state_ctx.shape[:2] != action_ctx.shape:
            raise ValueError("state/action context dimensions disagree")
        expected = int(self.config.history_size)
        if state_ctx.shape[1] != expected:
            raise ValueError(
                f"LatentWMPredictor expected exactly T={expected} from configured "
                f"history_size={expected}, got T={state_ctx.shape[1]}"
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(action_ctx, dtype=torch.bool)
        if valid_mask.shape != action_ctx.shape:
            raise ValueError("valid_mask must match action context B,T")
        return valid_mask.bool()

    def _predict_from_context(
        self,
        state_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = self._validate_context(state_ctx, action_ctx, valid_mask)
        if self.config.state_token_count > 1:
            return self.tokenized_predictor(state_ctx, action_ctx, valid_mask)
        if not valid_mask.all():
            raise ValueError("legacy vector WM does not implement masked history padding")
        actions = F.one_hot(action_ctx, num_classes=self.config.action_dim).float()
        act_emb = self.action_encoder(actions)
        if isinstance(self.state_input, nn.Identity):
            dynamics_ctx = state_ctx
        else:
            input_dtype = next(self.state_input.parameters()).dtype
            dynamics_ctx = self.state_input(state_ctx.to(dtype=input_dtype))
        preds = self.predictor(dynamics_ctx, act_emb)
        batch, history, _ = preds.shape
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=batch, t=history)
        return preds[:, -1]

    def predict_next_from_history(
        self,
        state_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._predict_from_context(state_ctx, action_ctx, valid_mask)

    def predict_next_emb(
        self, state_emb: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        if self.config.history_size != 1:
            raise ValueError(
                "predict_next_emb supplies T=1 but this predictor is configured "
                f"history_size={self.config.history_size}"
            )
        return self._predict_from_context(
            state_emb.unsqueeze(1), action_indices.unsqueeze(1)
        )

    def rollout_states(
        self, state_emb: torch.Tensor, action_sequences: torch.Tensor
    ) -> torch.Tensor:
        if self.config.history_size == 1:
            current = state_emb
            predicted: list[torch.Tensor] = []
            for step in range(action_sequences.shape[1]):
                current = self.predict_next_emb(current, action_sequences[:, step])
                predicted.append(current)
            if not predicted:
                return state_emb.new_empty((state_emb.shape[0], 0, state_emb.shape[-1]))
            return torch.stack(predicted, dim=1)
        if self.config.state_token_count == 1:
            raise ValueError(
                "single-state rollout requires a tokenized predictor trained with masked history"
            )
        history_size = self.config.history_size
        all_states = [state_emb]
        outputs: list[torch.Tensor] = []
        for step in range(action_sequences.shape[1]):
            real_count = min(history_size, step + 1)
            state_ctx = state_emb.new_zeros(
                (state_emb.shape[0], history_size, state_emb.shape[-1])
            )
            action_ctx = torch.zeros(
                (state_emb.shape[0], history_size),
                device=action_sequences.device,
                dtype=torch.long,
            )
            valid = torch.zeros_like(action_ctx, dtype=torch.bool)
            first = step + 1 - real_count
            state_ctx[:, -real_count:] = torch.stack(all_states[first : step + 1], dim=1)
            action_ctx[:, -real_count:] = action_sequences[:, first : step + 1]
            valid[:, -real_count:] = True
            next_state = self.predict_next_from_history(state_ctx, action_ctx, valid)
            all_states.append(next_state)
            outputs.append(next_state)
        if not outputs:
            return state_emb.new_empty((state_emb.shape[0], 0, state_emb.shape[-1]))
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        state_emb: torch.Tensor,
        action_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_emb.ndim == 3:
            return self._predict_from_context(state_emb, action_indices, valid_mask)
        if valid_mask is not None:
            raise ValueError("valid_mask is only valid with explicit B,T State context")
        return self.predict_next_emb(state_emb, action_indices)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "predictor.pt")
        (path / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )

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
                    "history_size_override only supports migration to trained T=1"
                )
            cfg_dict["history_size"] = 1
        cfg = LeWMConfig(
            **{
                key: value
                for key, value in cfg_dict.items()
                if key in LeWMConfig.__dataclass_fields__
            }
        )
        module = cls.create(cfg)
        state_path = path / "predictor.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location=map_location, weights_only=True)
            if history_size_override == 1:
                key = "predictor.pos_embedding"
                if key not in state or state[key].ndim != 3:
                    raise ValueError(
                        "checkpoint misses predictor.pos_embedding for T=1 migration"
                    )
                state[key] = state[key][:, :1].clone()
            module.load_state_dict(state)
            return module
        raise FileNotFoundError(f"missing LatentWMPredictor checkpoint: {state_path}")
