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

from nimloth.agent import Agent
from nimloth.backbone import BackboneEMA
from nimloth.util.distributed import is_main
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead
from nimloth.wm.model import WorldModel
from nimloth.wm.grid import TemporalSpatialGridPredictor
from nimloth.training.rl.token_value import TokenValueHead


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
    agent: Agent,
    processor: Any,
    vision_ema: BackboneEMA | None,
    save_llm: bool = True,
    # Training state
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
    global_step: int = 0,
    best_eval_metric: float = float("-inf"),
    checkpoint_metric: str = "success_rate",
    # Tune metadata
    lora: bool = False,
    llm_tune: str = "freeze",
    vision_tune: str = "freeze",
    base_model_path: str = "",
    token_value_head: torch.nn.Module | None = None,
    credit_assignment: str = "action",
    token_credit_config: dict[str, Any] | None = None,
    truncated_bootstrap: str | None = None,
    planner_config: dict[str, Any] | None = None,
    planner_distillation_weight: float | None = None,
    reference_kl_config: dict[str, Any] | None = None,
    train_world_model: bool = True,
) -> None:
    model = agent.backbone.model
    state_proj = agent.wm.state_proj
    wm_predictor = agent.wm.wm_predictor
    value_head = agent.wm.value_head
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
        # WM modules
        torch.save(_unwrap(state_proj).state_dict(), out_dir / "state_proj.pt")
        _unwrap(wm_predictor).save_checkpoint(out_dir / "wm_predictor")
        _unwrap(value_head).save_checkpoint(out_dir / "value_head")
        if token_value_head is not None:
            token_head = _unwrap(token_value_head)
            if not isinstance(token_head, TokenValueHead):
                raise TypeError("token_value_head must unwrap to TokenValueHead")
            token_head.save_checkpoint(out_dir / "token_value_head")
        agent.wm.save_checkpoint_extras(out_dir)

        # Qwen model
        if save_llm:
            agent.backbone.save_pretrained(
                out_dir,
                state_dict=full_model_state if fsdp_model else None,
            )
            processor.save_pretrained(out_dir)
            if vision_ema is not None and vision_ema.shadow:
                vision_ema.save_checkpoint(out_dir / "vision_ema.pt")

        # Training state
        state: dict[str, Any] = {
            "iteration": iteration,
            "global_step": global_step,
            "best_eval_metric": best_eval_metric,
            "checkpoint_metric": checkpoint_metric,
            "lora": lora,
            "llm_tune": llm_tune,
            "vision_tune": vision_tune,
            "optimizer_world_size": world if fsdp_model else 1,
            "training_world_size": world,
            "optimizer_state_layout": (
                "rank_sharded_fsdp" if fsdp_model else "replicated"
            ),
            "credit_assignment": credit_assignment,
            "token_credit_config": token_credit_config,
            "truncated_bootstrap": truncated_bootstrap,
            "planner_config": planner_config,
            "planner_distillation_weight": planner_distillation_weight,
            "reference_kl_config": reference_kl_config,
            "train_world_model": bool(train_world_model),
        }
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
    wm: WorldModel,
    device: torch.device,
) -> dict:
    """Load *only* the WM components from an RL checkpoint.

    返回训练状态字典，包括 iteration、global_step、best_eval_metric 和
    optimizer 等可恢复信息。
    """
    state_proj = wm.state_proj
    wm_predictor = wm.wm_predictor
    value_head = wm.value_head
    sp_path = ckpt_dir / "state_proj.pt"
    if sp_path.is_file():
        _unwrap(state_proj).load_state_dict(
            torch.load(sp_path, map_location=device, weights_only=True)
        )
    pred_dir = ckpt_dir / "wm_predictor"
    if pred_dir.is_dir():
        predictor = _unwrap(wm_predictor)
        if isinstance(predictor, TemporalSpatialGridPredictor):
            loaded_pred = TemporalSpatialGridPredictor.load_checkpoint(
                pred_dir,
                map_location=device,
            )
        else:
            loaded_pred = LatentWMPredictor.load_checkpoint(
                pred_dir,
                map_location=device,
            )
        _unwrap(wm_predictor).load_state_dict(loaded_pred.state_dict())
    head_dir = ckpt_dir / "value_head"
    if head_dir.is_dir():
        head = _unwrap(value_head)
        loaded_head = ValueHead.load_checkpoint(
            head_dir, emb_dim=head.net[0].in_features, map_location=device
        )
        head.load_state_dict(loaded_head.state_dict())
    wm.load_checkpoint_extras(ckpt_dir, map_location=device)

    state_path = ckpt_dir / "rl_state.pt"
    if state_path.is_file():
        return torch.load(state_path, map_location="cpu", weights_only=False)
    return {}


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
