"""RL checkpoint save/load helpers (WM + Qwen).

Covers both the WM modules (state_proj, predictor, value_head) and Qwen
model state (LoRA adapters, full-finetune weights, vision EMA).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.training.common.dist import is_main
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def _is_fsdp(module: torch.nn.Module | None) -> bool:
    if module is None:
        return False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return False
    return isinstance(module, FSDP)


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_rl_checkpoint(
    out_dir: Path,
    *,
    # WM modules
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    # Qwen model (may be DDP-wrapped or PeftModel)
    model: torch.nn.Module | None = None,
    processor: Any = None,
    vision_ema: Any = None,
    # Training state
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
    global_step: int = 0,
    best_value_loss: float = float("inf"),
    # Tune metadata
    lora: bool = False,
    llm_tune: str = "freeze",
    vision_tune: str = "freeze",
    base_model_path: str = "",
    latent_query_mode: str = "inject",
    rollout_policy: str = "qwen",
    fast_path_horizon: int = 0,
    predictor_rollout_steps: int = 1,
    predictor_rollout_loss_decay: float = 1.0,
) -> None:
    rank, world = _rank_world()
    fsdp_model = _is_fsdp(model)

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    # FSDP full-state collection is collective: every rank must enter it.
    full_model_state = None
    if fsdp_model:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
            full_model_state = model.state_dict()

    # Raw optimizer states are rank-local FSDP shards. Saving one file per rank
    # supports exact same-world-size resume while also covering the local WM heads.
    if optimizer is not None and fsdp_model:
        torch.save(optimizer.state_dict(), out_dir / f"optimizer_rank_{rank:05d}.pt")

    if rank == 0:
        query_token_ids: list[int] | None = None
        if processor is not None:
            from nimloth.latent.extraction import (
                LatentActionTokens,
                latent_state_tokens,
                special_token_ids,
            )

            token_count = int(getattr(_unwrap(state_proj), "latent_token_count", 1))
            tokens = LatentActionTokens()
            token_map = special_token_ids(
                processor.tokenizer, tokens, latent_token_count=token_count
            )
            query_token_ids = [
                int(token_map[name]) for name in latent_state_tokens(token_count, tokens)
            ]

        # WM modules
        torch.save(_unwrap(state_proj).state_dict(), out_dir / "state_proj.pt")
        _unwrap(wm_predictor).save_checkpoint(out_dir / "wm_predictor")
        _unwrap(value_head).save_checkpoint(out_dir / "value_head")

        # Qwen model
        if model is not None:
            m = _unwrap(model)
            m.config.nimloth_latent_token_count = int(
                getattr(_unwrap(state_proj), "latent_token_count", 1)
            )
            m.config.nimloth_latent_query_mode = latent_query_mode
            if query_token_ids is not None:
                m.config.nimloth_latent_query_token_ids = query_token_ids
            if fsdp_model:
                m.save_pretrained(
                    out_dir,
                    state_dict=full_model_state,
                    safe_serialization=True,
                )
            else:
                m.save_pretrained(out_dir, safe_serialization=True)
        if processor is not None:
            processor.save_pretrained(out_dir)
        if vision_ema is not None:
            ema = _unwrap(vision_ema) if hasattr(vision_ema, "module") else vision_ema
            if hasattr(ema, "save_checkpoint"):
                ema.save_checkpoint(out_dir / "vision_ema.pt")
            elif hasattr(ema, "shadow") and ema.shadow:
                torch.save({"shadow": {k: v.cpu() for k, v in ema.shadow.items()}},
                           out_dir / "vision_ema.pt")

        # Training state
        state: dict[str, Any] = {
            "iteration": iteration,
            "global_step": global_step,
            "best_value_loss": best_value_loss,
            "lora": lora,
            "llm_tune": llm_tune,
            "vision_tune": vision_tune,
            "optimizer_world_size": world if fsdp_model else 1,
            "latent_token_count": int(
                getattr(_unwrap(state_proj), "latent_token_count", 1)
            ),
            "latent_query_mode": latent_query_mode,
            "qwen_hidden_dim": int(
                getattr(_unwrap(state_proj), "qwen_hidden_dim", -1)
            ),
            "state_proj_input_dim": int(
                getattr(_unwrap(state_proj), "input_dim", -1)
            ),
            "rollout_policy": rollout_policy,
            "fast_path_horizon": int(fast_path_horizon),
            "predictor_rollout_steps": int(predictor_rollout_steps),
            "predictor_rollout_loss_decay": float(predictor_rollout_loss_decay),
        }
        if query_token_ids is not None:
            state["latent_query_token_ids"] = query_token_ids
        if base_model_path:
            state["base_model_path"] = str(base_model_path)
        if optimizer is not None and not fsdp_model:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, out_dir / "rl_state.pt")

    if world > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_rl_wm_checkpoint(
    ckpt_dir: Path,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    device: torch.device,
) -> dict:
    """Load *only* the WM components from an RL checkpoint.

    Returns the training-state dict (iteration, global_step, best_value_loss,
    optimizer, …).
    """
    sp_path = ckpt_dir / "state_proj.pt"
    if sp_path.is_file():
        _unwrap(state_proj).load_state_dict(
            torch.load(sp_path, map_location=device, weights_only=True)
        )
    pred_dir = ckpt_dir / "wm_predictor"
    if pred_dir.is_dir():
        loaded_pred = LatentWMPredictor.load_checkpoint(pred_dir, map_location=device)
        _unwrap(wm_predictor).load_state_dict(loaded_pred.state_dict())
    head_dir = ckpt_dir / "value_head"
    if head_dir.is_dir():
        head = _unwrap(value_head)
        loaded_head = ValueHead.load_checkpoint(
            head_dir, emb_dim=head.net[0].in_features, map_location=device
        )
        head.load_state_dict(loaded_head.state_dict())

    state_path = ckpt_dir / "rl_state.pt"
    if state_path.is_file():
        return torch.load(state_path, map_location="cpu", weights_only=False)
    return {}


def validate_rl_checkpoint_metadata(
    state: dict[str, Any],
    *,
    state_proj: StateProjector,
    latent_query_mode: str,
    rollout_policy: str,
    fast_path_horizon: int,
    predictor_rollout_steps: int,
    predictor_rollout_loss_decay: float,
    latent_query_token_ids: list[int] | None = None,
) -> None:
    """Fail fast when a resumed RL checkpoint changes protocol semantics."""

    proj = _unwrap(state_proj)
    expected: dict[str, Any] = {
        "latent_token_count": int(getattr(proj, "latent_token_count", 1)),
        "latent_query_mode": latent_query_mode,
        "qwen_hidden_dim": int(getattr(proj, "qwen_hidden_dim", -1)),
        "state_proj_input_dim": int(getattr(proj, "input_dim", -1)),
        "rollout_policy": rollout_policy,
        "fast_path_horizon": int(fast_path_horizon),
        "predictor_rollout_steps": int(predictor_rollout_steps),
        "predictor_rollout_loss_decay": float(predictor_rollout_loss_decay),
    }
    if latent_query_token_ids is not None:
        expected["latent_query_token_ids"] = [int(value) for value in latent_query_token_ids]
    for key, expected_value in expected.items():
        if key not in state:
            continue
        actual = state[key]
        if isinstance(expected_value, float):
            matches = float(actual) == expected_value
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"RL checkpoint {key} mismatch: checkpoint={actual!r}, "
                f"current={expected_value!r}"
            )


def load_lora_adapter_state(model: torch.nn.Module, adapter_dir: Path) -> None:
    """Load LoRA adapter weights into *model* (which must already be PeftModel).

    Mirrors :func:`nimloth.training.sft2.checkpoint.load_lora_adapter_state`.
    """
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
            json.dumps({
                "resume_load": {
                    "adapter_dir": str(adapter_dir),
                    "missing_keys": len(incompatible.missing_keys),
                    "unexpected_keys": len(incompatible.unexpected_keys),
                }
            })
        )


# Alias for backward compatibility
load_rl_checkpoint = load_rl_wm_checkpoint
