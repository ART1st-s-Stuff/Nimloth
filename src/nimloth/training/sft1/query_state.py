"""DeepSight-aligned direct K16 state objective and parameter ownership.

This is a new SFT1 contract.  It deliberately does not reuse or reinterpret the
legacy state-interface-v2 projector/readout objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    require_archived_assistant_response,
)
from nimloth.backbone.qwen25vl.tuning import is_llm_param, is_vision_param
from nimloth.latent import LatentActionTokens
from nimloth.wm.grid import (
    DIRECT_SLOT_PROJECTOR_ARTIFACT_SCHEMA,
    DirectSlotProjector,
)


QUERY_STATE_SCHEMA = "nimloth_sft1_query_state_v1"
QUERY_STATE_OBJECTIVE_VERSION = "direct_query_state_dino_lm_v1"
DIRECT_STATE_ARTIFACT_SCHEMA = DIRECT_SLOT_PROJECTOR_ARTIFACT_SCHEMA


@dataclass(frozen=True)
class QueryStateTargets:
    """Original-observation DINO targets and real-row validity."""

    dino_regions: torch.Tensor
    sample_valid: torch.Tensor


@dataclass(frozen=True)
class QueryStateNormalization:
    """Update-global denominators for framework-averaged distributed gradients."""

    global_state_valid_element_count: int
    global_lm_valid_token_count: int
    gradient_average_world_size: int

    def __post_init__(self) -> None:
        if self.global_state_valid_element_count < 1:
            raise ValueError("global state valid-element count must be positive")
        if self.global_lm_valid_token_count < 1:
            raise ValueError("global LM valid-token count must be positive")
        if self.gradient_average_world_size < 1:
            raise ValueError("gradient-average world size must be positive")


@dataclass(frozen=True)
class QueryStateObjectiveOutput:
    """Raw Query hidden, unique canonical state, and the two active losses."""

    raw_query_hidden: torch.Tensor
    state: torch.Tensor
    action_logits: torch.Tensor
    losses: Mapping[str, torch.Tensor]
    total_loss: torch.Tensor
    loss_sums: Mapping[str, torch.Tensor]
    local_valid_counts: Mapping[str, int]


class SFT1QueryStateObjective(nn.Module):
    """Fixed ``2 * direct DINO MSE + 1 * real assistant CE`` objective."""

    STATE_WEIGHT = 2.0
    LM_WEIGHT = 1.0

    def __init__(self, *, projector: DirectSlotProjector) -> None:
        super().__init__()
        if not isinstance(projector, DirectSlotProjector):
            raise TypeError(
                "Query-State requires the canonical no-bias DirectSlotProjector"
            )
        self.projector = projector

    def forward(
        self,
        query_hidden: torch.Tensor,
        action_logits: torch.Tensor,
        lm_loss_sum: torch.Tensor,
        lm_valid_token_count: int,
        targets: QueryStateTargets,
        normalization: QueryStateNormalization,
    ) -> QueryStateObjectiveOutput:
        if query_hidden.ndim != 3 or tuple(query_hidden.shape[1:]) != (16, 2048):
            raise ValueError(
                "Query-State query hidden must have shape (B,16,2048), "
                f"got {tuple(query_hidden.shape)}"
            )
        batch_size = int(query_hidden.shape[0])
        if action_logits.shape != (batch_size, 8):
            raise ValueError("Query-State action logits must have shape (B,8)")
        if targets.dino_regions.shape != (batch_size, 16, 1024):
            raise ValueError("Query-State DINO target must have shape (B,16,1024)")
        if targets.sample_valid.shape != (batch_size,):
            raise ValueError("Query-State sample_valid must have shape (B,)")
        if targets.sample_valid.dtype != torch.bool:
            raise ValueError("Query-State sample_valid must be boolean")
        if not torch.isfinite(query_hidden).all():
            raise ValueError("Query-State query hidden must be finite")
        if not torch.isfinite(action_logits).all():
            raise ValueError("Query-State action logits must be finite")
        if not torch.isfinite(targets.dino_regions).all():
            raise ValueError("Query-State DINO target must be finite")
        if not isinstance(lm_valid_token_count, int) or lm_valid_token_count < 0:
            raise ValueError("local LM valid-token count must be a non-negative integer")
        if lm_valid_token_count > normalization.global_lm_valid_token_count:
            raise ValueError("local LM valid-token count exceeds the global count")
        if lm_loss_sum.ndim != 0 or not torch.isfinite(lm_loss_sum):
            raise ValueError("same-forward LM loss sum must be a finite scalar")
        detached_lm_sum = float(lm_loss_sum.detach().item())
        if detached_lm_sum < 0.0:
            raise ValueError("same-forward LM loss sum must be non-negative")
        if lm_valid_token_count == 0 and detached_lm_sum != 0.0:
            raise ValueError("zero-token LM loss sum must be exactly zero")

        state = self.projector(query_hidden)
        if not torch.isfinite(state).all():
            raise ValueError("Query-State canonical state must be finite")
        dino = targets.dino_regions.detach().to(
            device=state.device,
            dtype=torch.float32,
        )
        valid = targets.sample_valid.detach().to(device=state.device)
        state_square = (state.float() - dino).square()
        state_loss_sum = (
            state_square * valid[:, None, None].to(dtype=state_square.dtype)
        ).sum()
        local_state_count = int(valid.sum().item()) * 16 * 1024
        if local_state_count > normalization.global_state_valid_element_count:
            raise ValueError("local state valid-element count exceeds the global count")

        # FSDP/DDP averages parameter gradients.  world_size/global_count makes
        # each local sum become one global valid-element/token mean afterwards.
        state_loss = state_loss_sum * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_state_valid_element_count)
        )
        lm_loss = lm_loss_sum.float() * (
            float(normalization.gradient_average_world_size)
            / float(normalization.global_lm_valid_token_count)
        )
        total = self.STATE_WEIGHT * state_loss + self.LM_WEIGHT * lm_loss
        losses = {
            "direct_state_mse": state_loss,
            "lm_ce": lm_loss,
        }
        return QueryStateObjectiveOutput(
            raw_query_hidden=query_hidden,
            state=state,
            action_logits=action_logits.float(),
            losses=losses,
            total_loss=total,
            loss_sums={
                "direct_state_mse": state_loss_sum,
                "lm_ce": lm_loss_sum.float(),
            },
            local_valid_counts={
                "direct_state_mse": local_state_count,
                "lm_ce": lm_valid_token_count,
            },
        )


@dataclass(frozen=True)
class QueryStateParameterInventory:
    language_trainable: tuple[str, ...]
    direct_state_trainable: tuple[str, ...]
    visual_frozen: tuple[str, ...]
    other_frozen: tuple[str, ...]
    other_trainable: tuple[str, ...]


@dataclass(frozen=True)
class QueryStateTrainableParameterGroup:
    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


def _is_lm_head_param(name: str) -> bool:
    return name.startswith("lm_head.") or ".lm_head." in name


def query_state_parameter_inventory(
    root: "SFT1QueryStateTrainingRoot",
) -> QueryStateParameterInventory:
    """Enumerate every parameter and fail on any ownership ambiguity."""

    if not isinstance(root, SFT1QueryStateTrainingRoot):
        raise TypeError("Query-State inventory requires SFT1QueryStateTrainingRoot")
    language_trainable: list[str] = []
    visual_frozen: list[str] = []
    other_frozen: list[str] = []
    other_trainable: list[str] = []
    errors: list[str] = []
    lm_head_names: list[str] = []

    for local_name, parameter in root.backbone.named_parameters():
        name = f"backbone.{local_name}"
        if "nimloth_query_embedding_adapter" in local_name:
            errors.append(f"query additive adapter is forbidden: {name}")
            if parameter.requires_grad:
                other_trainable.append(name)
            else:
                other_frozen.append(name)
            continue
        if is_vision_param(local_name):
            if parameter.requires_grad:
                errors.append(f"visual parameter is trainable: {name}")
                other_trainable.append(name)
            else:
                visual_frozen.append(name)
            continue
        if is_llm_param(local_name):
            if _is_lm_head_param(local_name):
                lm_head_names.append(name)
            if not parameter.requires_grad:
                if _is_lm_head_param(local_name):
                    errors.append(f"LM head parameter is frozen: {name}")
                else:
                    errors.append(f"language parameter is frozen: {name}")
                other_frozen.append(name)
            else:
                language_trainable.append(name)
            continue
        if parameter.requires_grad:
            errors.append(f"unowned backbone parameter is trainable: {name}")
            other_trainable.append(name)
        else:
            other_frozen.append(name)

    direct_state_trainable: list[str] = []
    expected_direct_names = {"projector.linear.weight"}
    actual_direct_names = {
        name for name, _parameter in root.objective.named_parameters()
    }
    if actual_direct_names != expected_direct_names:
        errors.append(
            "direct objective parameter set mismatch: "
            f"expected={sorted(expected_direct_names)}, "
            f"actual={sorted(actual_direct_names)}"
        )
    for local_name, parameter in root.objective.named_parameters():
        name = f"objective.{local_name}"
        if not parameter.requires_grad:
            errors.append(f"direct state parameter is frozen: {name}")
        else:
            direct_state_trainable.append(name)

    if not language_trainable:
        errors.append("full language path has no trainable parameters")
    if not lm_head_names:
        errors.append("top-level LM head parameter is absent")
    if not visual_frozen:
        errors.append("frozen visual parameter inventory is empty")
    if not direct_state_trainable:
        errors.append("direct state head has no trainable parameter")
    if errors:
        raise ValueError("; ".join(errors))

    inventory = QueryStateParameterInventory(
        language_trainable=tuple(language_trainable),
        direct_state_trainable=tuple(direct_state_trainable),
        visual_frozen=tuple(visual_frozen),
        other_frozen=tuple(other_frozen),
        other_trainable=tuple(other_trainable),
    )
    trainable_names = set(inventory.language_trainable).union(
        inventory.direct_state_trainable
    )
    actual_trainable_names = {
        name for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if trainable_names != actual_trainable_names:
        raise ValueError(
            "Query-State trainable inventory is not exhaustive: "
            f"missing={sorted(actual_trainable_names - trainable_names)}, "
            f"extra={sorted(trainable_names - actual_trainable_names)}"
        )
    return inventory


def query_state_trainable_parameter_groups(
    root: "SFT1QueryStateTrainingRoot",
) -> tuple[QueryStateTrainableParameterGroup, ...]:
    """Return disjoint pre-FSDP language/direct-state parameter groups."""

    inventory = query_state_parameter_inventory(root)
    named = dict(root.named_parameters())
    groups = (
        QueryStateTrainableParameterGroup(
            name="language",
            parameter_names=inventory.language_trainable,
            parameters=tuple(named[name] for name in inventory.language_trainable),
        ),
        QueryStateTrainableParameterGroup(
            name="direct_state",
            parameter_names=inventory.direct_state_trainable,
            parameters=tuple(named[name] for name in inventory.direct_state_trainable),
        ),
    )
    parameter_ids = [
        id(parameter)
        for group in groups
        for parameter in group.parameters
    ]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("Query-State trainable parameter groups overlap")
    return groups


def _require_complete_archived_response(response: str, *, source: str) -> None:
    require_archived_assistant_response(response, source=source)
    tokens = LatentActionTokens()
    if response.count(tokens.action_start) != 1 or response.count(tokens.action_end) != 1:
        raise ValueError("Query-State requires one complete archived action boundary")
    action_blocks = [
        tokens.action_start + action + tokens.action_end
        for action in tokens.action_tokens
    ]
    if sum(response.count(block) for block in action_blocks) != 1:
        raise ValueError("Query-State requires one actual archived action token")


class SFT1QueryStateTrainingRoot(nn.Module):
    """One wrapped call owning Qwen plus the unique direct canonical state head."""

    def __init__(
        self,
        backbone: nn.Module,
        objective: SFT1QueryStateObjective,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.objective = objective

    def forward(
        self,
        batch: QwenStateTrainingBatch,
        targets: QueryStateTargets,
        normalization: QueryStateNormalization,
    ) -> QueryStateObjectiveOutput:
        if "labels" not in batch.backbone_batch.tensors:
            raise ValueError("Query-State requires final-assistant LM labels")
        for response, source in zip(
            batch.archived_assistant_responses,
            batch.response_sources,
            strict=True,
        ):
            _require_complete_archived_response(response, source=source)
        forward = getattr(self.backbone, "forward_state_training", None)
        if forward is None:
            raise TypeError("Query-State backbone must implement forward_state_training")
        student = forward(batch)
        if student.lm_loss_sum is None:
            raise RuntimeError("same-forward Qwen output omitted its LM loss sum")
        return self.objective(
            student.query_hidden,
            student.action_logits,
            student.lm_loss_sum,
            student.lm_valid_token_count,
            targets,
            normalization,
        )

    def assert_trainable_contract(self) -> QueryStateParameterInventory:
        return query_state_parameter_inventory(self)


__all__ = [
    "DIRECT_STATE_ARTIFACT_SCHEMA",
    "QUERY_STATE_OBJECTIVE_VERSION",
    "QUERY_STATE_SCHEMA",
    "QueryStateNormalization",
    "QueryStateObjectiveOutput",
    "QueryStateParameterInventory",
    "QueryStateTargets",
    "QueryStateTrainableParameterGroup",
    "SFT1QueryStateObjective",
    "SFT1QueryStateTrainingRoot",
    "query_state_parameter_inventory",
    "query_state_trainable_parameter_groups",
]
