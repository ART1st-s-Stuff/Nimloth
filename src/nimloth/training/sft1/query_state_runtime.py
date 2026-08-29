"""Production-preparation constructor and FSDP ownership for Query-State SFT1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import nn

from nimloth.backbone.base import LoadedBackbone
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.training.sft1.query_state import (
    QueryStateParameterInventory,
    SFT1QueryStateObjective,
    SFT1QueryStateTrainingRoot,
    query_state_trainable_parameter_groups,
)
from nimloth.training.verl.runtime import (
    MixedPrecisionConfig,
    TrainingAssembly,
    assemble_training_root,
    assert_complete_module_device,
    wrap_complete_fsdp,
)
from nimloth.training.verl.source import verify_pinned_vagen_verl_source
from nimloth.wm.grid import DirectSlotProjector


@dataclass(frozen=True)
class QueryStateProductionContract:
    llm_tune: str = "full"
    vision_tune: str = "freeze"
    query_tune: str = "freeze"
    latent_token_count: int = 16
    state_dim: int = 1024

    def __post_init__(self) -> None:
        actual = (
            self.llm_tune,
            self.vision_tune,
            self.query_tune,
            self.latent_token_count,
            self.state_dim,
        )
        expected = ("full", "freeze", "freeze", 16, 1024)
        if actual != expected:
            raise ValueError(
                "Query-State production contract requires full language, frozen "
                "vision, no query adapter, K16, and state_dim=1024"
            )


@dataclass(frozen=True)
class QueryStateConstructedRoot:
    root: SFT1QueryStateTrainingRoot
    inventory: QueryStateParameterInventory
    contract: QueryStateProductionContract


@dataclass(frozen=True)
class QueryStatePreWrapOwnership:
    # Preserve named-parameter order. Optimizer state_dict loading is positional
    # within each group, so object-id set iteration would break exact resume in a
    # fresh process even when the parameter sets happen to match.
    language_parameter_ids: tuple[int, ...]
    direct_state_parameter_ids: tuple[int, ...]
    parameter_names: Mapping[int, str]


@dataclass(frozen=True)
class QueryStateWorkerAssembly:
    root: nn.Module
    optimizer: torch.optim.Optimizer
    ownership: QueryStatePreWrapOwnership


def _require_single_floating_parameter_dtype(
    module: nn.Module,
    *,
    owner: str,
) -> torch.dtype:
    by_dtype: dict[torch.dtype, list[str]] = {}
    for name, parameter in module.named_parameters():
        if parameter.is_floating_point():
            by_dtype.setdefault(parameter.dtype, []).append(name)
    if not by_dtype:
        raise ValueError(f"Query-State {owner} has no floating parameters")
    if len(by_dtype) != 1:
        summary = ", ".join(
            f"{dtype}: {names[0]}"
            for dtype, names in sorted(by_dtype.items(), key=lambda item: str(item[0]))
        )
        raise ValueError(
            f"Query-State {owner} has mixed floating parameter dtypes: {summary}"
        )
    return next(iter(by_dtype))


def construct_query_state_production_root(
    loaded: LoadedBackbone,
    *,
    contract: QueryStateProductionContract | None = None,
) -> QueryStateConstructedRoot:
    """Construct only the new direct-state root from an already loaded Qwen."""

    resolved = contract or QueryStateProductionContract()
    backbone = loaded.backbone
    if not isinstance(backbone, Qwen25VLBackbone):
        raise TypeError("Query-State production constructor requires Qwen25VLBackbone")
    if backbone.latent_token_count != resolved.latent_token_count:
        raise ValueError("Query-State loaded backbone does not use K16")
    if backbone.vision_tune != resolved.vision_tune:
        raise ValueError("Query-State loaded backbone vision mode is not frozen")
    if backbone.lora:
        raise ValueError("Query-State full-language production path rejects LoRA")
    if loaded.query_adapter is not None:
        raise ValueError("Query-State production path forbids a query additive adapter")
    if loaded.pair_parallel:
        raise ValueError(
            "Query-State complete-root FSDP path rejects pre-sharded model-parallel Qwen"
        )
    model_dtype = _require_single_floating_parameter_dtype(
        backbone,
        owner="loaded Qwen",
    )
    objective = SFT1QueryStateObjective(
        projector=DirectSlotProjector().to(dtype=model_dtype)
    )
    root = SFT1QueryStateTrainingRoot(backbone, objective)
    inventory = root.assert_trainable_contract()
    # The objective type and exhaustive inventory make SharedSlotProjector or a
    # hidden second state owner impossible at construction time.
    if tuple(inventory.direct_state_trainable) != (
        "objective.projector.linear.weight",
    ):
        raise RuntimeError("Query-State direct state ownership changed unexpectedly")
    return QueryStateConstructedRoot(root=root, inventory=inventory, contract=resolved)


def capture_query_state_pre_wrap_ownership(
    root: SFT1QueryStateTrainingRoot,
) -> QueryStatePreWrapOwnership:
    """Capture exact original-parameter identities before official FSDP wrapping."""

    groups = query_state_trainable_parameter_groups(root)
    by_name = dict(root.named_parameters())
    names: dict[int, str] = {}
    ids_by_group: dict[str, list[int]] = {"language": [], "direct_state": []}
    for group in groups:
        for name, parameter in zip(group.parameter_names, group.parameters, strict=True):
            if by_name.get(name) is not parameter:
                raise ValueError("Query-State parameter group name/identity mismatch")
            identity = id(parameter)
            if identity in names:
                raise ValueError("Query-State pre-wrap parameter identity is duplicated")
            names[identity] = name
            ids_by_group[group.name].append(identity)
    actual = {id(parameter) for parameter in root.parameters() if parameter.requires_grad}
    language_ids = tuple(ids_by_group["language"])
    direct_state_ids = tuple(ids_by_group["direct_state"])
    expected = set(language_ids) | set(direct_state_ids)
    if actual != expected or set(language_ids) & set(direct_state_ids):
        raise ValueError("Query-State pre-wrap optimizer ownership is not exhaustive/disjoint")
    return QueryStatePreWrapOwnership(
        language_parameter_ids=language_ids,
        direct_state_parameter_ids=direct_state_ids,
        parameter_names=names,
    )


def _query_state_adamw_factory(
    ownership: QueryStatePreWrapOwnership,
    *,
    language_learning_rate: float,
    direct_state_learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    epsilon: float,
) -> Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]:
    if language_learning_rate <= 0.0 or direct_state_learning_rate <= 0.0:
        raise ValueError("Query-State learning rates must be positive")
    if weight_decay < 0.0 or epsilon <= 0.0:
        raise ValueError("Query-State AdamW weight decay/epsilon are invalid")
    if not all(0.0 <= value < 1.0 for value in betas):
        raise ValueError("Query-State AdamW betas must lie in [0,1)")

    def build(parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        by_id = {id(parameter): parameter for parameter in trainable}
        if len(by_id) != len(trainable):
            raise ValueError("Query-State post-wrap trainables contain duplicate identities")
        expected = set(ownership.language_parameter_ids) | set(
            ownership.direct_state_parameter_ids
        )
        actual = set(by_id)
        if actual != expected:
            missing = [
                ownership.parameter_names.get(value, str(value))
                for value in expected - actual
            ]
            unexpected = [
                ownership.parameter_names.get(value, str(value))
                for value in actual - expected
            ]
            raise ValueError(
                "Query-State post-FSDP ownership mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [by_id[value] for value in ownership.language_parameter_ids],
                    "lr": float(language_learning_rate),
                    "group_name": "language",
                },
                {
                    "params": [by_id[value] for value in ownership.direct_state_parameter_ids],
                    "lr": float(direct_state_learning_rate),
                    "group_name": "direct_state",
                },
            ],
            weight_decay=float(weight_decay),
            betas=betas,
            eps=float(epsilon),
        )
        if tuple(group.get("group_name") for group in optimizer.param_groups) != (
            "language",
            "direct_state",
        ):
            raise RuntimeError("Query-State optimizer group identity is invalid")
        return optimizer

    return build


def assemble_query_state_training_root(
    *,
    constructed: QueryStateConstructedRoot,
    device: torch.device,
    repo_root: Path,
    wrap_policy: Mapping[str, Any],
    mixed_precision: MixedPrecisionConfig,
    language_learning_rate: float,
    direct_state_learning_rate: float,
    weight_decay: float,
    adam_betas: tuple[float, float],
    adam_epsilon: float,
    wrap: Callable[[nn.Module], nn.Module] | None = None,
) -> QueryStateWorkerAssembly:
    """Fail closed before wrapping, then create exactly two optimizer groups.

    ``wrap`` exists for CPU ownership tests.  Production callers leave it unset
    and use the repository's official complete-root FULL_SHARD wrapper.
    """

    verify_pinned_vagen_verl_source(repo_root)
    constructed.root.assert_trainable_contract()
    backbone_device = getattr(constructed.root.backbone, "device", None)
    if backbone_device is None or torch.device(backbone_device) != device:
        raise ValueError(
            "Query-State assembly device must match the Qwen input-forward device"
        )
    # Capture identities only after the complete root (including the fresh direct
    # head) has moved. Module.to() is allowed to replace Parameter objects under
    # framework future flags; pre-move object IDs cannot own optimizer resume.
    constructed.root.train()
    constructed.root.to(device)
    assert_complete_module_device(constructed.root, device)
    _require_single_floating_parameter_dtype(
        constructed.root,
        owner="complete root",
    )
    ownership = capture_query_state_pre_wrap_ownership(constructed.root)
    wrapper = wrap or (
        lambda module: wrap_complete_fsdp(
            module,
            device=device,
            wrap_policy=wrap_policy,
            mixed_precision=mixed_precision,
            repo_root=repo_root,
        )
    )
    assembly: TrainingAssembly = assemble_training_root(
        constructed.root,
        device=device,
        wrap=wrapper,
        optimizer_factory=_query_state_adamw_factory(
            ownership,
            language_learning_rate=language_learning_rate,
            direct_state_learning_rate=direct_state_learning_rate,
            weight_decay=weight_decay,
            betas=adam_betas,
            epsilon=adam_epsilon,
        ),
    )
    return QueryStateWorkerAssembly(
        root=assembly.root,
        optimizer=assembly.optimizer,
        ownership=ownership,
    )


__all__ = [
    "QueryStateConstructedRoot",
    "QueryStatePreWrapOwnership",
    "QueryStateProductionContract",
    "QueryStateWorkerAssembly",
    "assemble_query_state_training_root",
    "capture_query_state_pre_wrap_ownership",
    "construct_query_state_production_root",
]
