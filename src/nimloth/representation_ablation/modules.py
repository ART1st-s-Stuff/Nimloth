"""Factories for Phase-1 representation ablation modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.representation_ablation.config import AblationConfig
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.reconstruction import WMImageDecoder
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def load_qwen_processor_and_model(cfg: AblationConfig, device: torch.device):
    """Load frozen Qwen checkpoint for Phase-1 `qwen_latent` extraction."""

    if cfg.init.qwen_checkpoint is None:
        raise ValueError("init.qwen_checkpoint is required")
    processor = AutoProcessor.from_pretrained(cfg.init.qwen_checkpoint, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = cfg.eval.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.init.qwen_checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=cfg.eval.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device)
    freeze_module(model)
    return processor, token_id_map, model


def load_predictor(cfg: AblationConfig, device: torch.device) -> LatentWMPredictor:
    if cfg.init.wm_predictor_checkpoint is None:
        raise ValueError("init.wm_predictor_checkpoint is required")
    predictor = LatentWMPredictor.load_checkpoint(cfg.init.wm_predictor_checkpoint, map_location=device).to(device)
    return freeze_module(predictor)  # type: ignore[return-value]


def load_state_projector(cfg: AblationConfig, *, qwen_hidden_size: int, emb_dim: int, device: torch.device) -> StateProjector:
    if cfg.init.state_proj_checkpoint is None:
        raise ValueError("init.state_proj_checkpoint is required")
    state_proj = StateProjector(qwen_hidden_size, emb_dim).to(device)
    state_proj.load_state_dict(torch.load(cfg.init.state_proj_checkpoint, map_location=device, weights_only=True))
    return freeze_module(state_proj)  # type: ignore[return-value]


def load_value_head(cfg: AblationConfig, *, emb_dim: int, device: torch.device) -> ValueHead | None:
    if cfg.init.value_head_checkpoint is None:
        return None
    value_state = cfg.init.value_head_checkpoint / "value_head.pt"
    if not value_state.is_file():
        raise FileNotFoundError(
            "value_head_checkpoint must contain value_head.pt; "
            f"missing {value_state}. Refusing to use random-initialized ValueHead."
        )
    value_head = ValueHead.load_checkpoint(
        cfg.init.value_head_checkpoint,
        emb_dim=emb_dim,
        hidden_dim=cfg.value_head.hidden_dim,
        map_location=device,
    ).to(device)
    return freeze_module(value_head)  # type: ignore[return-value]


def load_decoder(cfg: AblationConfig, device: torch.device) -> WMImageDecoder | None:
    if cfg.init.decoder_checkpoint is None:
        return None
    decoder = WMImageDecoder.load_checkpoint(cfg.init.decoder_checkpoint, map_location=device).to(device)
    return freeze_module(decoder)  # type: ignore[return-value]


def checkpoint_path_from_sft2_dir(checkpoint_dir: Path) -> dict[str, Path]:
    """Return standard aux checkpoint paths under an SFT2 checkpoint directory."""

    return {
        "qwen_checkpoint": checkpoint_dir,
        "state_proj_checkpoint": checkpoint_dir / "state_proj.pt",
        "wm_predictor_checkpoint": checkpoint_dir / "wm_predictor",
        "value_head_checkpoint": checkpoint_dir / "value_head",
    }


def module_metadata(cfg: AblationConfig) -> dict[str, Any]:
    return {
        "representation": {
            "type": cfg.representation.type,
            "num_tokens": cfg.representation.num_tokens,
            "dim": cfg.representation.dim,
            "source": cfg.representation.source,
        },
        "predictor": {
            "type": cfg.predictor.type,
            "history_size": cfg.predictor.history_size,
        },
        "value_head": {
            "type": cfg.value_head.type,
            "use_semantic_embedding": cfg.value_head.use_semantic_embedding,
        },
        "reconstruction": {
            "enabled": cfg.reconstruction.enabled,
            "type": cfg.reconstruction.type,
        },
    }
