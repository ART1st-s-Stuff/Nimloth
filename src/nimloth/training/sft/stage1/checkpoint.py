"""SFT1/SFT2 checkpoint 保存、阶段恢复校验和 LoRA 权重恢复。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from .distributed import is_main


def save_checkpoint(
    model,
    processor,
    out_dir: Path,
    name: str,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    epoch: int = 0,
    best_val: float = float("inf"),
    *,
    lora: bool = False,
    base_model_path: Path | None = None,
    merge_for_eval: bool = False,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    latent_query_mode: str = "inject",
) -> None:
    ckpt = out_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.config.nimloth_latent_token_count = int(latent_token_count)
    module.config.nimloth_latent_query_mode = latent_query_mode
    module.save_pretrained(ckpt, safe_serialization=True)
    processor.save_pretrained(ckpt)
    state = {
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "lora": lora,
        "latent_token_count": int(latent_token_count),
        "latent_query_mode": latent_query_mode,
        "mask_latent_query_labels": bool(mask_latent_query_labels),
        "training_stage": getattr(module.config, "nimloth_training_stage", "format"),
    }
    if base_model_path is not None:
        state["base_model_path"] = str(base_model_path)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, ckpt / "training_state.pt")
    if lora and merge_for_eval and is_main() and base_model_path is not None:
        merge_peft_checkpoint(base_model_path, ckpt, ckpt / "hf_merged", processor)


def validate_resume_stage(
    state: dict[str, Any], checkpoint: Path, expected: str
) -> None:
    """Only identified early-stage checkpoints may restore optimizer/cursor state."""
    wm_keys = {
        "best_val_wm_mse",
        "state_proj_input_dim",
        "training_invariants",
        "value_objective",
    }
    if wm_keys.intersection(state) or (checkpoint / "wm_predictor").exists():
        raise ValueError("WM/value checkpoint cannot resume an early training stage")
    saved_stage = state.get("training_stage")
    if saved_stage is None:
        legacy_keys = {"step", "epoch", "best_val", "lora"}
        if (
            not legacy_keys.issubset(state)
            or (checkpoint / "grid_state_config.json").exists()
        ):
            raise ValueError("checkpoint has no recognized early-stage identity")
        saved_stage = "format"
    if saved_stage != expected:
        raise ValueError(
            f"checkpoint training stage mismatch: {saved_stage} != {expected}"
        )


def find_latest_resume_dir(output_dir: Path) -> Path | None:
    """Return the newest epoch_* checkpoint dir, else best/ if present."""
    latest_epoch = -1
    latest_dir: Path | None = None
    for p in output_dir.glob("epoch_*"):
        if not (p / "training_state.pt").is_file():
            continue
        try:
            ep = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if ep > latest_epoch:
            latest_epoch = ep
            latest_dir = p
    if latest_dir is not None:
        return latest_dir
    best = output_dir / "best"
    if (best / "training_state.pt").is_file():
        return best
    return None


def merge_peft_checkpoint(
    base_model_path: Path,
    adapter_path: Path,
    out_path: Path,
    processor: AutoProcessor,
) -> None:
    from .checkpoint_export import merge_checkpoint

    merge_checkpoint(base_model_path, adapter_path, out_path, processor)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    # The server's PEFT expects a newer Transformers TP symbol. This model has
    # no tensor-parallel plan, so a sentinel only satisfies PEFT's lazy import;
    # the TP sharding branch remains unreachable.
    import transformers.integrations.tensor_parallel as transformers_tp

    if not hasattr(transformers_tp, "EmbeddingParallel"):

        class _EmbeddingParallelSentinel:
            pass

        transformers_tp.EmbeddingParallel = _EmbeddingParallelSentinel

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    incompatible = set_peft_model_state_dict(model, dict(state), adapter_name="default")
    loaded_state = get_peft_model_state_dict(
        model, adapter_name="default", save_embedding_layers=True
    )
    missing_saved = sorted(set(state) - set(loaded_state))
    shape_mismatches = sorted(
        key
        for key in state.keys() & loaded_state.keys()
        if state[key].shape != loaded_state[key].shape
    )
    value_mismatches = sorted(
        key
        for key in state.keys() & loaded_state.keys()
        if state[key].shape == loaded_state[key].shape
        and not torch.equal(state[key], loaded_state[key].cpu())
    )
    allowed_modules_to_save_keys = {
        key
        for key in incompatible.unexpected_keys
        if key in state and key.endswith(".modules_to_save.weight")
    }
    unexplained_unexpected = sorted(
        set(incompatible.unexpected_keys) - allowed_modules_to_save_keys
    )
    if missing_saved or shape_mismatches or value_mismatches or unexplained_unexpected:
        raise RuntimeError(
            "PEFT resume adapter verification failed: "
            f"missing_saved={missing_saved[:3]} shape={shape_mismatches[:3]} "
            f"values={value_mismatches[:3]} unexpected={unexplained_unexpected[:3]}"
        )
    if is_main():
        print(
            json.dumps(
                {
                    "resume_load": {
                        "adapter_dir": str(adapter_dir),
                        "saved_tensors_verified": len(state),
                        "missing_base_keys": len(incompatible.missing_keys),
                        "verified_modules_to_save_compat_keys": len(
                            allowed_modules_to_save_keys
                        ),
                        "unexpected_keys": 0,
                    }
                }
            )
        )
