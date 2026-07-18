"""Actor-side Nimloth world-model auxiliary loss for VERL PPO updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nimloth.training.sft2.loss import compute_wm_latent_loss
from nimloth.training.sft2.qwen_latent import _capture_last_hidden
from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector


class NimlothWMAuxiliaryModules(nn.Module):
    """Replicated/DDP WM heads optimized jointly with one actor PPO minibatch."""

    def __init__(
        self,
        state_projector: StateProjector,
        predictor: LatentWMPredictor,
    ) -> None:
        super().__init__()
        self.state_projector = state_projector
        self.predictor = predictor

    def forward(
        self,
        current_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        wm_loss, _sigreg, _metrics = compute_wm_latent_loss(
            qwen_hidden_at_latent=current_hidden,
            qwen_hidden_at_next_latent=next_hidden,
            action_indices=action_indices,
            state_proj=self.state_projector,
            wm_predictor=self.predictor,
            sigreg_module=None,
        )
        return wm_loss


def build_nimloth_wm_auxiliary_modules(
    model_config,
    config,
) -> NimlothWMAuxiliaryModules:
    """Build checkpoint-compatible WM modules for one VERL actor worker."""
    latent_token_count = int(config.get("latent_token_count", 0))
    projector_hidden_dim = int(config.get("projector_hidden_dim", 2048))
    if latent_token_count < 1 or projector_hidden_dim < 1:
        raise ValueError("invalid Nimloth WM latent/projector dimensions")

    checkpoint_value = config.get("checkpoint_dir")
    allow_random_init = bool(config.get("allow_random_init", False))
    if not checkpoint_value and not allow_random_init:
        raise ValueError(
            "Nimloth WM auxiliary requires checkpoint_dir unless "
            "allow_random_init is explicitly enabled for a mechanics gate"
        )

    checkpoint_dir = Path(checkpoint_value).expanduser().resolve() if checkpoint_value else None
    if checkpoint_dir is not None:
        state_path = checkpoint_dir / "training_state.pt"
        predictor_config_path = checkpoint_dir / "wm_predictor" / "config.json"
        if not state_path.is_file() or not predictor_config_path.is_file():
            raise FileNotFoundError(
                f"incomplete Nimloth WM checkpoint: {checkpoint_dir}"
            )
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_k = int(training_state.get("latent_token_count", -1))
        saved_mode = training_state.get("latent_query_mode")
        if saved_k != latent_token_count or saved_mode != "inject":
            raise ValueError(
                "Nimloth WM checkpoint protocol mismatch: "
                f"k={saved_k}, mode={saved_mode!r}"
            )
        predictor_config = json.loads(
            predictor_config_path.read_text(encoding="utf-8")
        )
        lewm_config = LeWMConfig(
            **{
                key: value
                for key, value in predictor_config.items()
                if key in LeWMConfig.__dataclass_fields__
            }
        )
    else:
        lewm_config = LeWMConfig(
            emb_dim=int(config.get("emb_dim", 64)),
            action_dim=int(config.get("action_dim", 8)),
            history_size=int(config.get("history_size", 1)),
            predictor_depth=int(config.get("predictor_depth", 1)),
            predictor_heads=int(config.get("predictor_heads", 1)),
            predictor_mlp_dim=int(config.get("predictor_mlp_dim", 128)),
            predictor_hidden_dim=int(config.get("predictor_hidden_dim", 64)),
        )

    text_config = getattr(model_config, "text_config", model_config)
    qwen_hidden_dim = int(text_config.hidden_size)
    state_projector = StateProjector(
        qwen_hidden_dim=qwen_hidden_dim,
        lewm_emb_dim=int(lewm_config.emb_dim),
        projector_hidden_dim=projector_hidden_dim,
        latent_token_count=latent_token_count,
    )
    predictor = LatentWMPredictor(lewm_config)

    if checkpoint_dir is not None:
        state_projector.load_state_dict(
            torch.load(
                checkpoint_dir / "state_proj.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        loaded_predictor = LatentWMPredictor.load_checkpoint(
            checkpoint_dir / "wm_predictor", map_location="cpu"
        )
        predictor.load_state_dict(loaded_predictor.state_dict(), strict=True)

    return NimlothWMAuxiliaryModules(state_projector, predictor)


def _multimodal_inputs(micro_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    values = micro_batch.get("multi_modal_inputs")
    if values is None:
        return {}
    if len(values) != int(micro_batch["input_ids"].shape[0]):
        raise ValueError("WM auxiliary multimodal inputs do not match batch size")
    keys = tuple(values[0].keys())
    if any(tuple(item.keys()) != keys for item in values):
        raise ValueError("WM auxiliary multimodal input keys differ across rows")
    return {
        key: torch.cat([item[key] for item in values], dim=0)
        for key in keys
    }


def compute_verl_wm_auxiliary_loss(
    actor_module,
    wm_auxiliary_module,
    micro_batch: dict[str, Any],
    *,
    latent_token_count: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute next-latent MSE for all eligible transitions in a PPO microbatch."""
    if latent_token_count < 1:
        raise ValueError("WM auxiliary latent_token_count must be positive")
    required = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "wm_latent_positions",
        "wm_action_indices",
        "wm_transition_mask",
    )
    missing = [key for key in required if key not in micro_batch]
    if missing:
        raise ValueError(f"WM auxiliary batch is missing {missing}")

    if wm_auxiliary_module is None:
        raise RuntimeError("actor worker is missing Nimloth WM auxiliary modules")

    input_ids = micro_batch["input_ids"]
    attention_mask = micro_batch["attention_mask"]
    position_ids = micro_batch["position_ids"]
    if position_ids.ndim == 3:
        position_ids = position_ids.transpose(0, 1)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **_multimodal_inputs(micro_batch),
        "use_cache": False,
    }
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16):
        hidden, model_output = _capture_last_hidden(actor_module, model_inputs)
    # Latents come from a forward hook, outside the FSDP root's returned output
    # tree. Keep a zero-valued dependency on returned logits so FSDP's root
    # pre-backward hook enters FORWARD_BACKWARD before parameter hooks fire.
    fsdp_backward_anchor = model_output.logits.sum() * 0.0

    positions = micro_batch["wm_latent_positions"].long()
    actions = micro_batch["wm_action_indices"].long()
    transition_mask = micro_batch["wm_transition_mask"].bool()
    if positions.ndim != 3 or positions.shape[-1] != latent_token_count:
        raise ValueError(
            "WM latent positions must have shape [B, T, k], got "
            f"{tuple(positions.shape)}"
        )
    if actions.shape != positions.shape[:2] or transition_mask.shape != actions.shape:
        raise ValueError("WM action/transition metadata shape mismatch")

    current_rows = []
    next_rows = []
    action_rows = []
    sequence_length = int(hidden.shape[1])
    for batch_index, turn_index in transition_mask.nonzero(as_tuple=False).tolist():
        if turn_index + 1 >= int(positions.shape[1]):
            raise ValueError("WM transition mask selects a missing next turn")
        current_positions = positions[batch_index, turn_index]
        next_positions = positions[batch_index, turn_index + 1]
        if bool((current_positions < 0).any()) or bool((next_positions < 0).any()):
            raise ValueError("WM transition selects padded latent positions")
        if bool((current_positions >= sequence_length).any()) or bool(
            (next_positions >= sequence_length).any()
        ):
            raise ValueError("WM transition latent position exceeds sequence length")
        current_rows.append(hidden[batch_index, current_positions])
        # Match SFT2: the target Qwen hidden state is stop-gradient, while the
        # shared projector still receives target-side gradient.
        next_rows.append(hidden[batch_index, next_positions].detach())
        action_rows.append(actions[batch_index, turn_index])

    if not current_rows:
        dummy_hidden = hidden[:1, :latent_token_count]
        if dummy_hidden.shape[1] != latent_token_count:
            raise ValueError("WM auxiliary cannot construct its zero-transition dummy")
        dummy_action = torch.zeros(1, dtype=torch.long, device=hidden.device)
        dummy_loss = wm_auxiliary_module(
            dummy_hidden, dummy_hidden.detach(), dummy_action
        )
        zero = hidden.sum() * 0.0 + dummy_loss * 0.0 + fsdp_backward_anchor
        return zero, {"wm_mse": 0.0, "wm_transitions": 0.0}

    current_hidden = torch.stack(current_rows, dim=0)
    next_hidden = torch.stack(next_rows, dim=0)
    action_tensor = torch.stack(action_rows).to(device=hidden.device, dtype=torch.long)
    wm_loss = (
        wm_auxiliary_module(current_hidden, next_hidden, action_tensor)
        + fsdp_backward_anchor
    )
    metrics = {
        "wm_mse": float(wm_loss.detach().item()),
        "wm_transitions": float(len(current_rows)),
    }
    return wm_loss, metrics
