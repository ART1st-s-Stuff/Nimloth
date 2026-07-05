"""Factories for Phase-1 representation ablation modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.representation_ablation.config import AblationConfig

if TYPE_CHECKING:
    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.reconstruction import WMImageDecoder
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _adapter_base_model_path(checkpoint: Path) -> Path | None:
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        return None
    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    base = data.get("base_model_name_or_path")
    return Path(base) if base else None


def _lora_args_from_adapter(checkpoint: Path) -> argparse.Namespace:
    data = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    return argparse.Namespace(
        lora=False,
        llm_tune="lora",
        vision_tune="full" if (checkpoint / "vision_full_state.pt").is_file() else "freeze",
        lora_r=int(data.get("r", 64)),
        lora_alpha=int(data.get("lora_alpha", 128)),
        lora_dropout=float(data.get("lora_dropout", 0.0)),
        gradient_checkpointing=False,
    )


def load_qwen_processor_and_model(cfg: AblationConfig, device: torch.device):
    """Load frozen Qwen checkpoint for Phase-1 `qwen_latent` extraction.

    Supports both full HF checkpoints and SFT2 LoRA+vision-full checkpoint dirs.
    Adapter-only checkpoints are loaded by first loading their recorded base model,
    then applying the adapter and `vision_full_state.pt`.
    """

    from nimloth.backbone.qwen_tuning import configure_qwen_tuning
    from nimloth.training.sft2.checkpoint import load_lora_adapter_state

    if cfg.init.qwen_checkpoint is None:
        raise ValueError("init.qwen_checkpoint is required")
    qwen_checkpoint = cfg.init.qwen_checkpoint
    adapter_base = _adapter_base_model_path(qwen_checkpoint)
    model_path = adapter_base or qwen_checkpoint
    if adapter_base is not None and not model_path.is_dir():
        raise FileNotFoundError(f"adapter base_model_name_or_path does not exist: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = cfg.eval.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=cfg.eval.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.vocab_size = len(processor.tokenizer)
    if hasattr(model, "generation_config"):
        model.generation_config.vocab_size = len(processor.tokenizer)
    if adapter_base is not None:
        model = configure_qwen_tuning(model, _lora_args_from_adapter(qwen_checkpoint))
        load_lora_adapter_state(model, qwen_checkpoint)
    model.to(device)
    freeze_module(model)
    return processor, token_id_map, model


def qwen_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    hidden = getattr(config, "hidden_size", None)
    if hidden is not None:
        return int(hidden)
    base = getattr(model, "base_model", None)
    base_model = getattr(base, "model", None)
    base_config = getattr(base_model, "config", None)
    hidden = getattr(base_config, "hidden_size", None)
    if hidden is not None:
        return int(hidden)
    raise AttributeError(f"could not resolve Qwen hidden_size from {type(model)}")


def load_predictor(cfg: AblationConfig, device: torch.device) -> LatentWMPredictor:
    from nimloth.wm.predictor import LatentWMPredictor

    if cfg.init.wm_predictor_checkpoint is None:
        raise ValueError("init.wm_predictor_checkpoint is required")
    predictor = LatentWMPredictor.load_checkpoint(cfg.init.wm_predictor_checkpoint, map_location=device).to(device)
    return freeze_module(predictor)  # type: ignore[return-value]


def load_state_projector(cfg: AblationConfig, *, qwen_hidden_size: int, emb_dim: int, device: torch.device) -> StateProjector:
    from nimloth.wm.state_proj import StateProjector

    if cfg.init.state_proj_checkpoint is None:
        raise ValueError("init.state_proj_checkpoint is required")
    state_proj = StateProjector(qwen_hidden_size, emb_dim).to(device)
    state_proj.load_state_dict(torch.load(cfg.init.state_proj_checkpoint, map_location=device, weights_only=True))
    return freeze_module(state_proj)  # type: ignore[return-value]


def load_value_head(cfg: AblationConfig, *, emb_dim: int, device: torch.device) -> ValueHead | None:
    from nimloth.wm.value_head import ValueHead

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
    from nimloth.wm.reconstruction import WMImageDecoder

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
