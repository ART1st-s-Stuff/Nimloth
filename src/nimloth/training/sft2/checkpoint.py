"""SFT2 checkpoint save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from nimloth.training.common.dist import is_main
from nimloth.backbone.vision_ema import VisionEncoderEMA
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


def read_checkpoint_step(ckpt_dir: Path) -> int:
    state_path = ckpt_dir / "training_state.pt"
    if not state_path.is_file():
        return -1
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return int(state.get("step", -1))


def is_trainable_checkpoint_dir(ckpt_dir: Path) -> bool:
    if not (ckpt_dir / "training_state.pt").is_file():
        return False
    return (ckpt_dir / "config.json").is_file() or (ckpt_dir / "adapter_config.json").is_file()


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    """Pick the saved checkpoint with the highest step (latest progress)."""
    candidates: list[tuple[int, Path]] = []
    for name in ("latest", "best"):
        ckpt_dir = output_dir / name
        if is_trainable_checkpoint_dir(ckpt_dir):
            candidates.append((read_checkpoint_step(ckpt_dir), ckpt_dir))
    for epoch_dir in sorted(output_dir.glob("epoch_*")):
        if is_trainable_checkpoint_dir(epoch_dir):
            candidates.append((read_checkpoint_step(epoch_dir), epoch_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_resume_checkpoint_dir(output_dir: Path, resume_from: Path | None) -> Path:
    if resume_from is not None:
        ckpt_dir = resume_from if resume_from.is_absolute() else output_dir / resume_from
    else:
        found = find_resume_checkpoint(output_dir)
        if found is None:
            raise FileNotFoundError(f"no trainable checkpoint under {output_dir}")
        ckpt_dir = found
    if not is_trainable_checkpoint_dir(ckpt_dir):
        raise FileNotFoundError(f"incomplete checkpoint dir: {ckpt_dir}")
    return ckpt_dir


def load_lora_adapter_state(model: torch.nn.Module, adapter_dir: Path) -> None:
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if adapter_file.is_file():
        from safetensors.torch import load_file

        state = load_file(str(adapter_file))
    else:
        bin_file = adapter_dir / "adapter_model.bin"
        if not bin_file.is_file():
            raise FileNotFoundError(f"missing adapter weights in {adapter_dir}")
        state = torch.load(bin_file, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if is_main():
        print(
            json.dumps(
                {
                    "resume_load": {
                        "adapter_dir": str(adapter_dir),
                        "missing_keys": len(incompatible.missing_keys),
                        "unexpected_keys": len(incompatible.unexpected_keys),
                    }
                }
            )
        )


def save_checkpoint(
    model,
    state_proj,
    processor,
    out_dir: Path,
    *,
    wm_predictor: LatentWMPredictor | None = None,
    value_head: ValueHead | None = None,
    vision_ema: VisionEncoderEMA | None = None,
    optimizer=None,
    step: int = 0,
    epoch: int = 0,
    best_val_success_rate: float = -1.0,
    best_val_wm_mse: float = float("inf"),
    lora: bool = False,
    base_model_path: Path | None = None,
    llm_tune: str = "freeze",
    vision_tune: str = "freeze",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    proj = state_proj.module if hasattr(state_proj, "module") else state_proj
    torch.save(proj.state_dict(), out_dir / "state_proj.pt")
    if wm_predictor is not None:
        pred = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
        pred.save_checkpoint(out_dir / "wm_predictor")
    if value_head is not None:
        head = value_head.module if hasattr(value_head, "module") else value_head
        head.save_checkpoint(out_dir / "value_head")
    if vision_ema is not None and vision_ema.shadow:
        vision_ema.save_checkpoint(out_dir / "vision_ema.pt")
    state: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "best_val_success_rate": best_val_success_rate,
        "best_val_wm_mse": best_val_wm_mse,
        "best_val": best_val_wm_mse,
        "lora": lora,
        "llm_tune": llm_tune,
        "vision_tune": vision_tune,
        "vision_ema": vision_ema is not None and bool(vision_ema.shadow),
    }
    if base_model_path is not None:
        state["base_model_path"] = str(base_model_path)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, out_dir / "training_state.pt")


def load_aux_checkpoint(
    ckpt_dir: Path,
    state_proj,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    device: torch.device,
) -> None:
    sp_path = ckpt_dir / "state_proj.pt"
    if sp_path.is_file():
        proj = state_proj.module if hasattr(state_proj, "module") else state_proj
        proj.load_state_dict(torch.load(sp_path, map_location=device, weights_only=True))
    pred_path = ckpt_dir / "wm_predictor"
    if pred_path.is_dir():
        pred = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
        loaded = LatentWMPredictor.load_checkpoint(pred_path, map_location=device)
        pred.load_state_dict(loaded.state_dict())
    head_path = ckpt_dir / "value_head"
    if head_path.is_dir():
        head = value_head.module if hasattr(value_head, "module") else value_head
        loaded_head = ValueHead.load_checkpoint(
            head_path,
            emb_dim=head.net[0].in_features,
            map_location=device,
        )
        head.load_state_dict(loaded_head.state_dict())
