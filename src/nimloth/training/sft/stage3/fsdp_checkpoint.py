"""Collective FSDP checkpoint boundary with ordinary HF component exports.

All ranks participate in collection. Only rank zero serializes CPU full state;
loading the named optimizer state uses the same Agent root that owns every group.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)

from nimloth.backbone.selected_token_rows import materialize_selected_state_dict

_ROW_FIELDS = {"nimloth_query_rows", "nimloth_protocol_rows", "nimloth_query_ids", "nimloth_protocol_ids"}


def is_fsdp_agent(agent: Any) -> bool:
    return isinstance(getattr(getattr(agent, "backbone", None), "model", None), FSDP)


def collect_fsdp_checkpoint(agent, optimizer) -> dict[str, Any]:
    """Collect CPU full Qwen and full named optimizer state on rank zero."""
    if not is_fsdp_agent(agent):
        raise ValueError("FSDP checkpoint collection requires a wrapped Qwen backbone")
    with FSDP.state_dict_type(
        agent, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        backbone_state = agent.backbone.model.state_dict()
        optimizer_state = FSDP.optim_state_dict(agent, optimizer) if optimizer is not None else None
    return {"backbone": backbone_state, "optimizer": optimizer_state}


def optimizer_state_to_load(agent, optimizer, state):
    """Transform CPU named full state to rank-local optimizer parameter IDs."""
    if not is_fsdp_agent(agent):
        return state
    with FSDP.state_dict_type(
        agent, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        return FSDP.optim_state_dict_to_load(agent, optimizer, state)


def save_collected_backbone(agent, output_dir: Path, state: dict, metadata: dict) -> None:
    """Export gathered CPU tensors without invoking rank-zero FSDP collectives.

    Bypass Backbone.save_pretrained, which otherwise reads live selected rows
    from a sharded module. The explicit gathered state is the sole tensor source.
    """
    if not state or any(value.device.type != "cpu" for value in state.values()):
        raise ValueError("FSDP export requires nonempty CPU full backbone state")
    model = agent.backbone.model.module
    for key, value in metadata.items():
        setattr(model.config, key, value)
    row_state = {key: value.clone() for key, value in state.items() if key.rsplit(".", 1)[-1] in _ROW_FIELDS}
    dense_state = materialize_selected_state_dict(state) if row_state else state
    model.save_pretrained(output_dir, state_dict=dense_state, safe_serialization=True)
    if row_state:
        torch.save(row_state, output_dir / "selected_token_rows.pt")
